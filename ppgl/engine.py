"""Training and inference engine.

Device handling is resolved at runtime, never hard-coded: CUDA (any number of
devices), Apple MPS, or CPU. Mixed precision is enabled only on CUDA.

The prediction path is built around the routing measurement. For a routed model,
:meth:`Trainer.predict` scores the *same trained weights* under several routing
rules in one pass - soft router probabilities, hard top-1, and ground-truth
"oracle" routing - so the oracle-minus-predicted gap is measured directly rather
than inferred from two separately trained models.
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ppgl import REGIONS
from ppgl.labels import active_classes
from ppgl.losses import PPGLLoss, class_prior_from_labels, inverse_frequency_weights
from ppgl.metrics import PRIMARY_METRIC, compute_all_metrics, region_metrics
from ppgl.mixing import mix_batch, permuted_batch
from ppgl.models import layer_wise_param_groups, uses_contrastive
from ppgl.scheduler import build_scheduler
from ppgl.utils import distributed as dist_utils
from ppgl.utils.device import supports_amp
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

# Routing rules evaluated for routed models. "oracle" needs ground-truth regions.
ROUTING_EVAL_MODES = ("soft", "argmax", "oracle")


def _build_grad_scaler(enabled: bool):
    """A gradient scaler that works across the supported torch range.

    ``torch.cuda.amp.GradScaler`` is deprecated from torch 2.4 and emits a
    FutureWarning on every fold, which is a lot of noise across ~1000 folds. The
    replacement ``torch.amp.GradScaler("cuda", ...)`` only exists from 2.3, and
    requirements.txt allows torch>=2.1, so both spellings are kept.
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch 2.1/2.2
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class Predictions:
    """Everything needed to score a split and to analyse routing, after the fact."""

    y_true: np.ndarray
    y_prob: np.ndarray
    slice_uids: List[str] = field(default_factory=list)
    patient_uids: List[str] = field(default_factory=list)
    region_true: Optional[np.ndarray] = None
    region_prob: Optional[np.ndarray] = None
    router_probs: Optional[np.ndarray] = None
    routing_weights: Optional[np.ndarray] = None
    expert_probs: Optional[np.ndarray] = None
    # Multiple-instance models only: a compact summary of the attention
    # distribution over each patient's slices. Stored per row rather than the full
    # variable-length vector so the predictions stay a flat CSV. A concentrated
    # distribution (high max, low entropy) means one slice carried the call.
    attention_max: Optional[np.ndarray] = None
    attention_entropy: Optional[np.ndarray] = None
    bag_size: Optional[np.ndarray] = None

    def metrics(self) -> Dict[str, Any]:
        out = compute_all_metrics(
            self.y_true, self.y_prob, patient_uids=self.patient_uids or None, class_names=active_classes()
        )
        if self.region_true is not None and self.region_prob is not None:
            out.update(region_metrics(self.region_true, self.region_prob, REGIONS))
        return out

    def to_frame(self):
        import pandas as pd

        data: Dict[str, Any] = {
            "slice_uid": self.slice_uids,
            "patient_uid": self.patient_uids,
            "y_true": self.y_true,
        }
        for idx, name in enumerate(active_classes()[: self.y_prob.shape[1]]):
            data[f"prob_{name}"] = self.y_prob[:, idx]
        if self.region_true is not None:
            data["region_true"] = self.region_true
        if self.region_prob is not None:
            for idx, name in enumerate(REGIONS[: self.region_prob.shape[1]]):
                data[f"region_prob_{name}"] = self.region_prob[:, idx]
        if self.router_probs is not None:
            for idx, name in enumerate(REGIONS[: self.router_probs.shape[1]]):
                data[f"router_{name}"] = self.router_probs[:, idx]
        if self.expert_probs is not None:
            for r_idx, region in enumerate(REGIONS[: self.expert_probs.shape[1]]):
                for c_idx, cluster in enumerate(active_classes()[: self.expert_probs.shape[2]]):
                    data[f"expert_{region}_{cluster}"] = self.expert_probs[:, r_idx, c_idx]
        for name, values in (
            ("attention_max", self.attention_max),
            ("attention_entropy", self.attention_entropy),
            ("bag_size", self.bag_size),
        ):
            if values is not None:
                data[name] = values
        return pd.DataFrame(data)


class Trainer:
    """Trains one model on one fold."""

    def __init__(
        self,
        model: nn.Module,
        cfg,
        device: torch.device,
        train_labels: Optional[Sequence[int]] = None,
        train_regions: Optional[Sequence[int]] = None,
        is_routed: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.is_routed = bool(is_routed)
        self.model = model.to(device)

        class_weights = None
        if cfg.loss.class_weighting == "inverse" and train_labels is not None:
            class_weights = inverse_frequency_weights(train_labels, len(active_classes())).to(device)

        # Logit adjustment needs the training class prior. Estimated per fold, from
        # the training rows only.
        class_prior = None
        logit_adjust_tau = float(getattr(cfg.loss, "logit_adjust_tau", 0.0))
        if logit_adjust_tau > 0:
            if train_labels is None:
                raise ValueError(
                    "loss.logit_adjust_tau > 0 requires train_labels to estimate the class prior"
                )
            class_prior = class_prior_from_labels(train_labels, len(active_classes())).to(device)
            if class_weights is not None:
                log.warning(
                    "loss.logit_adjust_tau=%.3g and loss.class_weighting='inverse' are both set. "
                    "Both correct the class prior, so the cross-entropy term will use logit "
                    "adjustment ONLY and ignore the class weights. Tversky still uses them. "
                    "Set loss.class_weighting='none' to make this explicit.",
                    logit_adjust_tau,
                )

        region_prior = None
        if train_regions is not None:
            regions = np.asarray(train_regions, dtype=int)
            regions = regions[regions >= 0]
            if regions.size:
                counts = np.bincount(regions, minlength=len(REGIONS)).astype(np.float64)
                region_prior = torch.tensor(counts / counts.sum(), dtype=torch.float32)

        consistency_weight = (
            float(cfg.model.consistency_weight)
            if self.is_routed and cfg.model.routing == "consistency"
            else 0.0
        )
        contrastive_weight = (
            float(cfg.loss.contrastive_weight) if uses_contrastive(cfg.model.name) else 0.0
        )
        if uses_contrastive(cfg.model.name) and contrastive_weight <= 0:
            # The ViT+CL baseline is meaningless without its contrastive term.
            contrastive_weight = 1.0
            log.info("vit_cl baseline: loss.contrastive_weight was 0, defaulting to 1.0")

        self.criterion = PPGLLoss(
            cfg.loss,
            n_classes=len(active_classes()),
            n_regions=len(REGIONS),
            class_weights=class_weights,
            region_prior=region_prior,
            class_prior=class_prior,
            consistency_weight=consistency_weight,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=float(cfg.loss.contrastive_temperature),
        ).to(device)

        self.optimizer = self._build_optimizer()
        self.scheduler = None  # built in fit(), needs steps_per_epoch
        self.use_amp = bool(cfg.optim.amp) and supports_amp(device)
        self.scaler = _build_grad_scaler(self.use_amp)

        self.model_for_forward = dist_utils.wrap_model(self.model, device)

        # Not every model takes clinical covariates or bag masks, so the extra
        # arguments are passed only to those whose forward() declares them.
        # Inspected once here rather than branching on model name at every call.
        import inspect

        parameters = inspect.signature(dist_utils.unwrap_model(self.model).forward).parameters
        self._accepts_clinical = "clinical" in parameters
        self._accepts_mask = "mask" in parameters
        self._accepts_lesion_mask = "lesion_mask" in parameters

        # Mixup / CutMix state. The generator is seeded per fold so a run is
        # reproducible; it is separate from the dataloader's so that changing the
        # mixing settings does not reshuffle the batches.
        self.mixup_alpha = float(getattr(cfg.optim, "mixup_alpha", 0.0))
        self.cutmix_alpha = float(getattr(cfg.optim, "cutmix_alpha", 0.0))
        self.mix_prob = float(getattr(cfg.optim, "mix_prob", 0.5))
        self._mixing_enabled = (self.mixup_alpha > 0 or self.cutmix_alpha > 0) and self.mix_prob > 0
        self._mix_rng = np.random.default_rng(int(cfg.runtime.seed) + 7919)
        self._warned_mix_skipped = False
        if self._mixing_enabled:
            log.info(
                "input mixing enabled: mixup_alpha=%.3g cutmix_alpha=%.3g prob=%.2f",
                self.mixup_alpha,
                self.cutmix_alpha,
                self.mix_prob,
            )

        # Test-time augmentation: 1 = identity only, 2 = identity + horizontal flip.
        self.tta_views = max(1, int(getattr(cfg.data, "tta", 1)))
        if self.tta_views > 2:
            log.warning(
                "data.tta=%d but only 2 anatomically plausible views exist "
                "(identity, horizontal flip); clamping to 2",
                self.tta_views,
            )
            self.tta_views = 2

        self.history: List[Dict[str, float]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_metric = -math.inf
        self.best_epoch = -1

    # ------------------------------------------------------------- optimiser
    def _build_optimizer(self):
        core = dist_utils.unwrap_model(self.model)
        head_modules = (
            core.trainable_head_modules()
            if hasattr(core, "trainable_head_modules")
            else [m for n, m in core.named_children() if n != "backbone"]
        )

        if self.cfg.model.finetune == "llrd":
            groups = layer_wise_param_groups(
                core.backbone,
                head_modules,
                base_lr=float(self.cfg.optim.lr),
                decay=float(self.cfg.optim.llrd),
                weight_decay=float(self.cfg.optim.weight_decay),
            )
        else:
            # LoRA and linear probe: only adapters/heads train, so a flat LR is
            # appropriate and LLRD has nothing to decay through.
            decay, no_decay = [], []
            for param in core.parameters():
                if not param.requires_grad:
                    continue
                (no_decay if param.ndim <= 1 else decay).append(param)
            groups = []
            if decay:
                groups.append(
                    {
                        "params": decay,
                        "lr": float(self.cfg.optim.lr),
                        "weight_decay": float(self.cfg.optim.weight_decay),
                        "name": "trainable",
                    }
                )
            if no_decay:
                groups.append(
                    {
                        "params": no_decay,
                        "lr": float(self.cfg.optim.lr),
                        "weight_decay": 0.0,
                        "name": "trainable_nodecay",
                    }
                )
            if not groups:
                raise RuntimeError("no trainable parameters; check model.finetune")

        betas = tuple(self.cfg.optim.betas)
        if self.cfg.optim.optimizer == "adamw":
            return torch.optim.AdamW(groups, betas=betas)
        if self.cfg.optim.optimizer == "adam":
            return torch.optim.Adam(groups, betas=betas)
        if self.cfg.optim.optimizer == "sgd":
            return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
        raise ValueError(f"unknown optimizer {self.cfg.optim.optimizer!r}")

    # ------------------------------------------------------------------- fit
    def fit(self, train_loader, val_loader=None) -> List[Dict[str, float]]:
        steps_per_epoch = max(
            1, math.ceil(len(train_loader) / max(1, int(self.cfg.optim.grad_accum_steps)))
        )
        self.scheduler = build_scheduler(self.optimizer, self.cfg.optim, steps_per_epoch)

        epochs = int(self.cfg.optim.epochs)
        patience = int(self.cfg.optim.early_stop_patience)
        min_epochs = int(self.cfg.optim.early_stop_min_epochs)
        since_improved = 0

        for epoch in range(epochs):
            # Router temperature annealing position.
            core = dist_utils.unwrap_model(self.model)
            progress = epoch / max(1, epochs - 1)
            if hasattr(core, "set_progress"):
                core.set_progress(progress)
            # The objective can also be scheduled (attention supervision warm-down).
            if hasattr(self.criterion, "set_progress"):
                self.criterion.set_progress(progress)

            started = time.time()
            train_stats = self._train_one_epoch(train_loader, epoch)
            record: Dict[str, float] = {"epoch": epoch, **train_stats}

            if val_loader is not None:
                predictions = self.predict(val_loader, routing_modes=("soft",))
                val_metrics = predictions["soft"].metrics()
                monitor = val_metrics.get(f"patient_{PRIMARY_METRIC}")
                if monitor is None or (isinstance(monitor, float) and math.isnan(monitor)):
                    monitor = val_metrics.get(PRIMARY_METRIC, -math.inf)
                record[f"val_patient_{PRIMARY_METRIC}"] = float(val_metrics.get(f"patient_{PRIMARY_METRIC}", float("nan")))
                record[f"val_{PRIMARY_METRIC}"] = float(val_metrics.get(PRIMARY_METRIC, float("nan")))
                record["val_monitor"] = float(monitor)

                if monitor > self.best_metric + 1e-6:
                    self.best_metric = float(monitor)
                    self.best_epoch = epoch
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in dist_utils.unwrap_model(self.model).state_dict().items()
                    }
                    since_improved = 0
                else:
                    since_improved += 1

            record["lr"] = float(self.scheduler.get_last_lr()[0]) if self.scheduler else 0.0
            record["seconds"] = round(time.time() - started, 2)
            self.history.append(record)

            if epoch % 5 == 0 or epoch == epochs - 1:
                log.info(
                    "epoch %d/%d loss=%.4f%s (%.1fs)",
                    epoch + 1,
                    epochs,
                    record.get("loss", float("nan")),
                    f" val_monitor={record['val_monitor']:.4f}" if "val_monitor" in record else "",
                    record["seconds"],
                )

            if (
                val_loader is not None
                and patience > 0
                and epoch + 1 >= min_epochs
                and since_improved >= patience
            ):
                log.info(
                    "early stop at epoch %d (best %.4f at epoch %d)",
                    epoch + 1,
                    self.best_metric,
                    self.best_epoch + 1,
                )
                break

        if self.best_state is not None:
            dist_utils.unwrap_model(self.model).load_state_dict(self.best_state)
            log.info("restored best weights from epoch %d", self.best_epoch + 1)
        return self.history

    def _train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model_for_forward.train()
        # A frozen backbone must stay in eval mode so its BN/dropout do not drift.
        if self.cfg.model.finetune == "linear_probe":
            dist_utils.unwrap_model(self.model).backbone.eval()

        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        accum = max(1, int(self.cfg.optim.grad_accum_steps))
        totals: Dict[str, float] = {}
        n_batches = 0
        n_mixed = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            batch = self._to_device(batch)

            mix = self._maybe_mix(batch)
            if mix is not None:
                batch = mix.batch
                n_mixed += 1

            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self._forward(batch)
                components = self.criterion(outputs, batch)
                if mix is not None:
                    # Blend the two hard-target evaluations rather than building a
                    # soft label, so the Tversky and auxiliary terms stay active.
                    other = self.criterion(outputs, permuted_batch(batch, mix.permutation))
                    components = {
                        key: mix.lam * value + (1.0 - mix.lam) * other[key]
                        if key in other
                        else value
                        for key, value in components.items()
                    }
                loss = components["total"] / accum

            if not torch.isfinite(loss):
                log.warning("non-finite loss at epoch %d step %d; skipping batch", epoch, step)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                if float(self.cfg.optim.grad_clip) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg.optim.grad_clip),
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            n_batches += 1
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())

        stats = {k: v / max(n_batches, 1) for k, v in totals.items()}
        stats["loss"] = stats.pop("total", float("nan"))
        if self._mixing_enabled:
            stats["mix_rate"] = n_mixed / max(n_batches, 1)
        return stats

    def _maybe_mix(self, batch: Dict[str, Any]):
        """Apply mixup/cutmix to a training batch, or return ``None``.

        Returns ``None`` when mixing is disabled, does not fire, or is not
        applicable - which includes bag-shaped (5-D) input, where blending across
        patients would mix instances between bags and corrupt the attention.
        """
        if not self._mixing_enabled:
            return None

        images = batch.get("image")
        if not torch.is_tensor(images):
            return None

        if images.dim() != 4:
            if not self._warned_mix_skipped:
                log.info(
                    "input mixing requested but the input is %d-D (bag model); "
                    "mixing is skipped for the whole run",
                    images.dim(),
                )
                self._warned_mix_skipped = True
            return None

        return mix_batch(
            batch,
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            prob=self.mix_prob,
            rng=self._mix_rng,
        )

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            out[key] = value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
        return out

    def _forward(
        self,
        batch: Dict[str, Any],
        force_routing: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Call the model with whichever optional inputs it declares."""
        kwargs: Dict[str, Any] = {
            "region": batch.get("region"),
            "force_routing": force_routing,
        }
        if self._accepts_clinical:
            kwargs["clinical"] = batch.get("clinical")
        if self._accepts_mask:
            kwargs["mask"] = batch.get("mask")
        if self._accepts_lesion_mask:
            kwargs["lesion_mask"] = batch.get("lesion_mask")
        return self.model_for_forward(batch["image"], **kwargs)

    def _forward_tta(
        self,
        batch: Dict[str, Any],
        force_routing: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass averaged over test-time views.

        Returns the identity view's raw outputs together with the view-averaged
        cluster and region probabilities. Averaging happens in probability space,
        not logit space, because the views are separate observations of the same
        slice rather than an ensemble of scores on one input.

        Only the cluster and region distributions are averaged. Router
        probabilities, expert probabilities and MIL attention are taken from the
        identity view alone: those are diagnostics tied to a specific input, and
        the routing analysis compares them against that slice's region label.
        """
        views: List[Dict[str, Any]] = [batch]
        if self.tta_views >= 2 and torch.is_tensor(batch.get("image")):
            flipped = dict(batch)
            # Left-right flip. Works for both (B,C,H,W) and (B,K,C,H,W).
            flipped["image"] = torch.flip(batch["image"], dims=(-1,))
            views.append(flipped)

        base_outputs: Optional[Dict[str, torch.Tensor]] = None
        prob_sum: Optional[torch.Tensor] = None
        region_sum: Optional[torch.Tensor] = None

        for view in views:
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self._forward(view, force_routing=force_routing)
            probs = torch.softmax(outputs["logits"].float(), dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
            if "region_logits" in outputs:
                region_probs = torch.softmax(outputs["region_logits"].float(), dim=1)
                region_sum = region_probs if region_sum is None else region_sum + region_probs
            if base_outputs is None:
                base_outputs = outputs

        divisor = float(len(views))
        assert base_outputs is not None and prob_sum is not None
        return (
            base_outputs,
            prob_sum / divisor,
            (region_sum / divisor) if region_sum is not None else None,
        )

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(
        self,
        loader,
        routing_modes: Sequence[str] = ("soft",),
    ) -> Dict[str, Predictions]:
        """Score ``loader`` under each routing rule.

        Non-routed models ignore ``routing_modes`` and return a single ``"soft"``
        entry, so callers can treat every model uniformly.
        """
        self.model_for_forward.eval()

        modes = list(routing_modes) if self.is_routed else ["soft"]
        if not self.is_routed:
            modes = ["soft"]

        collected: Dict[str, Dict[str, List[Any]]] = {
            mode: {
                "y_true": [],
                "y_prob": [],
                "slice_uid": [],
                "patient_uid": [],
                "region_true": [],
                "region_prob": [],
                "router": [],
                "weights": [],
                "expert": [],
                "attn_max": [],
                "attn_entropy": [],
                "bag_size": [],
            }
            for mode in modes
        }

        for batch in loader:
            batch = self._to_device(batch)
            region = batch.get("region")

            for mode in modes:
                if mode == "oracle" and (region is None or not bool((region >= 0).any())):
                    # No ground-truth regions available; oracle is undefined.
                    continue
                outputs, probs, region_probs = self._forward_tta(
                    batch, force_routing=mode if self.is_routed else None
                )

                bucket = collected[mode]
                bucket["y_true"].append(batch["cluster"].detach().cpu().numpy())
                bucket["y_prob"].append(probs.detach().cpu().numpy())
                bucket["slice_uid"].extend(batch["slice_uid"])
                bucket["patient_uid"].extend(batch["patient_uid"])
                if region is not None:
                    bucket["region_true"].append(region.detach().cpu().numpy())
                if region_probs is not None:
                    bucket["region_prob"].append(region_probs.detach().cpu().numpy())
                if "router_probs" in outputs:
                    bucket["router"].append(outputs["router_probs"].float().detach().cpu().numpy())
                if "routing_weights" in outputs:
                    bucket["weights"].append(
                        outputs["routing_weights"].float().detach().cpu().numpy()
                    )
                if "expert_logits" in outputs:
                    bucket["expert"].append(
                        torch.softmax(outputs["expert_logits"].float(), dim=-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                if "attention" in outputs:
                    # Summarise the per-instance attention: how peaked it is, and
                    # over how many real slices. Stored instead of the full
                    # variable-length vector to keep predictions a flat table.
                    attention = outputs["attention"].float().detach()
                    valid = batch.get("mask")
                    if valid is None:
                        valid = torch.ones_like(attention, dtype=torch.bool)
                    safe = attention.clamp_min(1e-12)
                    entropy = -(safe.log() * attention * valid).sum(dim=1)
                    bucket["attn_max"].append(
                        attention.masked_fill(~valid, 0.0).max(dim=1).values.cpu().numpy()
                    )
                    bucket["attn_entropy"].append(entropy.cpu().numpy())
                    bucket["bag_size"].append(
                        valid.sum(dim=1).to(torch.int64).cpu().numpy()
                    )

        out: Dict[str, Predictions] = {}
        for mode, bucket in collected.items():
            if not bucket["y_true"]:
                continue
            out[mode] = Predictions(
                y_true=np.concatenate(bucket["y_true"]),
                y_prob=np.concatenate(bucket["y_prob"]),
                slice_uids=list(bucket["slice_uid"]),
                patient_uids=list(bucket["patient_uid"]),
                region_true=np.concatenate(bucket["region_true"]) if bucket["region_true"] else None,
                region_prob=np.concatenate(bucket["region_prob"]) if bucket["region_prob"] else None,
                router_probs=np.concatenate(bucket["router"]) if bucket["router"] else None,
                routing_weights=np.concatenate(bucket["weights"]) if bucket["weights"] else None,
                expert_probs=np.concatenate(bucket["expert"]) if bucket["expert"] else None,
                attention_max=np.concatenate(bucket["attn_max"]) if bucket["attn_max"] else None,
                attention_entropy=(
                    np.concatenate(bucket["attn_entropy"]) if bucket["attn_entropy"] else None
                ),
                bag_size=np.concatenate(bucket["bag_size"]) if bucket["bag_size"] else None,
            )
        return out


# --------------------------------------------------------------------------- #
# DataLoader construction
# --------------------------------------------------------------------------- #
def build_loader(
    dataset,
    cfg,
    train: bool,
    labels: Optional[Sequence[int]] = None,
    seed: int = 0,
    collate: Optional[Any] = None,
):
    """Build a DataLoader.

    Class imbalance is handled with a ``WeightedRandomSampler`` on the training
    split (SDHx is over half the data) unless disabled. Under DDP a
    ``DistributedSampler`` is used instead, since the two cannot be combined
    without a custom sampler.
    """
    from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

    from ppgl.data.dataset import collate_batch
    from ppgl.utils.seed import make_generator, worker_init_fn

    collate_fn = collate or collate_batch
    batch_size = int(cfg.optim.batch_size)
    num_workers = int(cfg.data.num_workers)
    pin_memory = bool(cfg.data.pin_memory) and torch.cuda.is_available()

    sampler = None
    shuffle = False
    if dist_utils.is_distributed():
        sampler = DistributedSampler(dataset, shuffle=train, drop_last=False)
    elif train:
        if cfg.data.balanced_sampler and labels is not None and len(labels):
            counts = np.bincount(
                np.asarray(labels, dtype=int), minlength=len(active_classes())
            ).astype(np.float64)
            per_class = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
            weights = per_class[np.asarray(labels, dtype=int)]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
                generator=make_generator(seed),
            )
        else:
            shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        # BatchNorm in the MLP heads fails on a batch of one.
        drop_last=bool(train and len(dataset) > batch_size),
        persistent_workers=bool(num_workers > 0),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=make_generator(seed) if shuffle else None,
    )


__all__ = ["Predictions", "ROUTING_EVAL_MODES", "Trainer", "build_loader"]
