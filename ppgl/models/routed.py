"""Anatomy-routed mixture of expert classifiers. A comparison model.

Retained as a comparator and for the historical record. It is NOT the method the
current paper proposes - that is :mod:`ppgl.models.attn_lora` (Attention Mass
Guidance). This class also requires per-slice anatomical region labels, which the
archive does not contain, so its results are not reportable; see the README.

Architecture: a shared backbone produces pooled features ``z``; a router maps
``z`` to a distribution over the three anatomical regions; one expert classifier
per region maps ``z`` to cluster logits; the outputs are combined by the routing
weights.

The contribution this class is built to measure
-----------------------------------------------
Anatomy-conditioned MoE routing already exists in medical imaging (Shi et al.,
ICIC 2025; CT-Agent 2026; HarMoE 2026), and dense-to-sparse gate annealing dates
to EvoMoE (2021). Neither is claimed as novel here.

What is unmeasured in the prior work on *this* task is the **train/inference
routing mismatch**. The earlier version of this model selected an expert with the
**ground-truth one-hot region** during training but combined all experts with
**soft router probabilities** at inference. Each expert was therefore only ever
trained on hard-routed samples and then evaluated on blended ones - textbook
exposure bias, never quantified. :attr:`routing` selects among:

``hard``
    Ground-truth one-hot routing while training, soft at inference. Reproduces
    the mismatch; kept so the gap can be measured rather than assumed.
``soft``
    Router probabilities at both training and inference. The mismatch is removed.
``anneal``
    Router temperature decays from ``anneal_start_temp`` to ``anneal_end_temp``,
    so routing starts dense and sharpens (EvoMoE-style).
``consistency``
    Soft routing plus a router entropy/utilisation penalty applied in
    :class:`ppgl.losses.RouterConsistencyLoss`.

For evaluation, :meth:`forward` accepts ``force_routing`` so the same weights can
be scored under different routing rules without retraining:

``"oracle"``
    Ground-truth region one-hot. Upper bound given a perfect region estimator.
``"argmax"``
    Hard top-1 routing from the predicted region.
``"soft"``
    Router probabilities.

The oracle-minus-soft difference is the headline measurement of the paper.

Capacity note
-------------
Head & neck contributes 64 slices and chest 80. Three independent
``768 -> 256 -> 4`` experts is roughly 200k parameters each, which those
subsets cannot support. ``share_expert_trunk=True`` shares the trunk and keeps
only a per-region output layer, cutting per-expert parameters by ~99%. Expert
utilisation and effective sample size are recorded so this can be reported
rather than discovered by a reviewer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppgl import REGIONS
from ppgl.labels import n_active_classes
from ppgl.models.backbones import create_backbone
from ppgl.models.baselines import make_mlp_head
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


class AnatomyRoutedClassifier(nn.Module):
    """Shared backbone, per-region expert classifiers, router-weighted aggregation."""

    def __init__(
        self,
        model_cfg,
        n_classes: Optional[int] = None,
        n_regions: int = len(REGIONS),
        aggregate: str = "logit",
    ) -> None:
        super().__init__()
        if aggregate not in ("logit", "prob"):
            raise ValueError(f"aggregate must be 'logit' or 'prob', got {aggregate!r}")

        self.backbone, self.feature_dim, self.mean, self.std = create_backbone(model_cfg)
        self.n_classes = int(n_classes or n_active_classes())
        self.n_regions = int(n_regions)
        self.routing = model_cfg.routing
        self.aggregate = aggregate
        self.share_expert_trunk = bool(model_cfg.share_expert_trunk)

        # The routed model keeps the MLP experts of the prior architecture
        # unchanged, so model.head_type does not apply to it. Warn rather than
        # silently ignore the setting.
        if str(getattr(model_cfg, "head_type", "mlp")) != "mlp":
            log.warning(
                "model.head_type=%r has no effect on the routed model: its experts are "
                "kept as the prior work's MLP heads so the routing comparison is not "
                "confounded. Use a non-routed model to evaluate the prototype head.",
                model_cfg.head_type,
            )

        # Router: a single linear layer, matching the prior architecture.
        self.router = nn.Linear(self.feature_dim, self.n_regions)

        if self.share_expert_trunk:
            self.expert_trunk = nn.Sequential(
                nn.Linear(self.feature_dim, model_cfg.hidden_dim),
                nn.BatchNorm1d(model_cfg.hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(model_cfg.dropout),
            )
            self.experts = nn.ModuleList(
                nn.Linear(model_cfg.hidden_dim, self.n_classes) for _ in range(self.n_regions)
            )
        else:
            self.expert_trunk = None
            self.experts = nn.ModuleList(
                make_mlp_head(
                    self.feature_dim, model_cfg.hidden_dim, self.n_classes, model_cfg.dropout
                )
                for _ in range(self.n_regions)
            )

        self.base_temperature = float(model_cfg.router_temperature)
        self.anneal_start_temp = float(model_cfg.anneal_start_temp)
        self.anneal_end_temp = float(model_cfg.anneal_end_temp)
        # Current annealing position in [0, 1]; updated by the trainer each epoch.
        self.register_buffer("_progress", torch.zeros(1), persistent=False)

    # ------------------------------------------------------------------ setup
    def trainable_head_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [self.router]
        if self.expert_trunk is not None:
            modules.append(self.expert_trunk)
        modules.extend(self.experts)
        return modules

    def set_progress(self, progress: float) -> None:
        """Set training progress in [0, 1]; drives temperature annealing."""
        self._progress.fill_(float(min(max(progress, 0.0), 1.0)))

    @property
    def temperature(self) -> float:
        if self.routing != "anneal":
            return self.base_temperature
        progress = float(self._progress.item())
        return self.anneal_start_temp + progress * (self.anneal_end_temp - self.anneal_start_temp)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        region_logits = self.router(features)

        temperature = max(self.temperature, 1e-3)
        router_probs = F.softmax(region_logits / temperature, dim=1)

        trunk = self.expert_trunk(features) if self.expert_trunk is not None else features
        # (B, R, C)
        expert_logits = torch.stack([expert(trunk) for expert in self.experts], dim=1)

        mode = force_routing or self._active_mode()
        weights = self._routing_weights(mode, router_probs, region)

        if self.aggregate == "prob":
            expert_probs = F.softmax(expert_logits, dim=-1)
            mixed = (weights.unsqueeze(-1) * expert_probs).sum(dim=1)
            # Return log-probabilities so downstream code can treat the output
            # as logits uniformly (softmax of a log-prob is the prob itself).
            logits = torch.log(mixed.clamp_min(1e-8))
        else:
            logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)

        return {
            "logits": logits,
            "region_logits": region_logits,
            "router_probs": router_probs,
            "routing_weights": weights,
            "expert_logits": expert_logits,
            "features": features,
            "router_temperature": torch.tensor(temperature, device=features.device),
        }

    def _active_mode(self) -> str:
        """Routing rule in force for the current phase."""
        if self.training:
            # "hard" is the only mode that consumes the ground-truth region
            # during training; the others are soft and so identical at inference.
            return "hard" if self.routing == "hard" else "soft"
        return "soft"

    def _routing_weights(
        self,
        mode: str,
        router_probs: torch.Tensor,
        region: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mode in ("soft", "anneal", "consistency"):
            return router_probs

        if mode == "argmax":
            index = router_probs.argmax(dim=1)
            return F.one_hot(index, num_classes=self.n_regions).to(router_probs.dtype)

        if mode in ("hard", "oracle"):
            if region is None:
                if mode == "oracle":
                    raise ValueError("force_routing='oracle' requires ground-truth region labels")
                return router_probs
            valid = region >= 0
            onehot = F.one_hot(region.clamp_min(0), num_classes=self.n_regions).to(
                router_probs.dtype
            )
            # Slices without a region label fall back to soft routing rather
            # than being silently assigned to region 0.
            return torch.where(valid.unsqueeze(1), onehot, router_probs)

        raise ValueError(f"unknown routing mode {mode!r}")

    # ------------------------------------------------------------- diagnostics
    @torch.no_grad()
    def expert_utilisation(self, router_probs: torch.Tensor) -> Dict[str, float]:
        """Mean routing mass and top-1 share per expert, for the utilisation table."""
        mean_mass = router_probs.mean(dim=0)
        top1 = router_probs.argmax(dim=1)
        counts = torch.bincount(top1, minlength=self.n_regions).to(router_probs.dtype)
        share = counts / max(router_probs.shape[0], 1)
        out: Dict[str, float] = {}
        for idx, name in enumerate(REGIONS[: self.n_regions]):
            out[f"router_mass_{name}"] = float(mean_mass[idx])
            out[f"router_top1_share_{name}"] = float(share[idx])
        return out


__all__ = ["AnatomyRoutedClassifier"]
