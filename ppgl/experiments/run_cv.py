"""Cross-validation runner for the deep models.

Trains one model configuration across every requested fold and writes per-fold
predictions and metrics into the results store. Nothing is aggregated here; the
tables and statistics read the store afterwards.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ppgl import REGIONS
from ppgl.labels import active_classes
from ppgl.data.splits import Fold, fingerprint
from ppgl.engine import ROUTING_EVAL_MODES, Trainer, build_loader
from ppgl.experiments import results as store
from ppgl.experiments.data_prep import build_folds, prepare_frame
from ppgl.models import build_model
from ppgl.models.names import is_bag_model, is_tabular, needs_clinical, needs_lesion_mask
from ppgl.utils.device import describe_hardware, resolve_device
from ppgl.utils.logging import get_logger, setup_logging
from ppgl.utils.seed import job_seed, seed_everything

log = get_logger(__name__)


def run_fold(
    cfg,
    frame: pd.DataFrame,
    fold: Fold,
    device_index: Optional[int] = None,
    fold_fingerprint: str = "",
) -> Dict[str, Any]:
    """Train and evaluate one fold. Returns the per-mode metrics."""
    from ppgl.data.dataset import PPGLSliceDataset

    if is_tabular(cfg.model.name):
        raise ValueError(f"{cfg.model.name!r} is a tabular control; use run_tabular")

    fold_directory = store.fold_dir(cfg.output_dir, cfg.experiment, fold.repeat, fold.fold)
    os.makedirs(fold_directory, exist_ok=True)
    setup_logging(log_file=os.path.join(fold_directory, "log.txt"), job_tag=f"{cfg.experiment}/{fold.tag}")

    seed = job_seed(int(cfg.runtime.seed), fold.repeat, fold.fold)
    seed_everything(seed, deterministic=bool(cfg.runtime.deterministic))

    device = resolve_device(cfg.runtime.device, device_index=device_index)
    log.info(
        "fold %s on %s | model=%s backbone=%s finetune=%s routing=%s",
        fold.tag,
        device,
        cfg.model.name,
        cfg.model.backbone,
        cfg.model.finetune,
        cfg.model.routing,
    )

    train_frame = frame.iloc[fold.train_idx].reset_index(drop=True)
    val_frame = frame.iloc[fold.val_idx].reset_index(drop=True)
    test_frame = frame.iloc[fold.test_idx].reset_index(drop=True)

    # Clinical covariates, if this model uses them. The preprocessor is fit on the
    # training rows of THIS fold only, so imputation and scaling cannot leak.
    clinical_matrices: Dict[str, Any] = {"train": None, "val": None, "test": None}
    n_clinical = 0
    if needs_clinical(cfg.model.name) or cfg.data.clinical_csv:
        from ppgl.data.clinical import build_fold_matrices, clinical_columns

        if clinical_columns(frame):
            clinical_matrices, preprocessor = build_fold_matrices(
                frame,
                fold.train_idx,
                {"train": fold.train_idx, "val": fold.val_idx, "test": fold.test_idx},
            )
            n_clinical = preprocessor.n_features
            log.info("clinical block: %d features from %d columns",
                     n_clinical, len(preprocessor.columns))
        elif needs_clinical(cfg.model.name):
            raise ValueError(
                f"model {cfg.model.name!r} needs clinical covariates but none were attached; "
                "set data.clinical_csv"
            )

    extra_model_kwargs: Dict[str, Any] = {}
    if needs_clinical(cfg.model.name):
        extra_model_kwargs["n_clinical_features"] = n_clinical

    model = build_model(
        cfg.model,
        n_classes=len(active_classes()),
        n_regions=len(REGIONS),
        **extra_model_kwargs,
    )
    mean = getattr(model, "mean", None)
    std = getattr(model, "std", None)

    # Multiple-instance models consume one bag per patient; everything else is
    # one item per slice. The only difference is the dataset and the collate.
    collate = None
    if is_bag_model(cfg.model.name):
        from ppgl.data.bags import PPGLBagDataset, collate_bags

        collate = collate_bags
        train_ds = PPGLBagDataset(
            train_frame, cfg.data, train=True, mean=mean, std=std,
            clinical=clinical_matrices["train"], max_slices=int(cfg.data.bag_max_slices), seed=seed,
        )
        val_ds = PPGLBagDataset(
            val_frame, cfg.data, train=False, mean=mean, std=std,
            clinical=clinical_matrices["val"], seed=seed,
        )
        test_ds = PPGLBagDataset(
            test_frame, cfg.data, train=False, mean=mean, std=std,
            clinical=clinical_matrices["test"], seed=seed,
        )
    else:
        # Attention-supervised models also need the lesion footprint, transformed
        # under the same geometry as the image.
        wants_mask = needs_lesion_mask(cfg.model.name)
        train_ds = PPGLSliceDataset(
            train_frame, cfg.data, train=True, mean=mean, std=std,
            clinical=clinical_matrices["train"], return_mask=wants_mask,
        )
        val_ds = PPGLSliceDataset(
            val_frame, cfg.data, train=False, mean=mean, std=std,
            clinical=clinical_matrices["val"], return_mask=wants_mask,
        )
        test_ds = PPGLSliceDataset(
            test_frame, cfg.data, train=False, mean=mean, std=std,
            clinical=clinical_matrices["test"], return_mask=wants_mask,
        )

    train_labels = train_ds.cluster_labels
    train_regions = train_ds.region_labels

    train_loader = build_loader(
        train_ds, cfg, train=True, labels=train_labels, seed=seed, collate=collate
    )
    val_loader = build_loader(val_ds, cfg, train=False, seed=seed, collate=collate)
    test_loader = build_loader(test_ds, cfg, train=False, seed=seed, collate=collate)

    is_routed = cfg.model.name == "routed"
    trainer = Trainer(
        model,
        cfg,
        device,
        train_labels=train_labels,
        train_regions=train_regions,
        is_routed=is_routed,
    )
    history = trainer.fit(train_loader, val_loader if len(val_ds) else None)

    modes = ROUTING_EVAL_MODES if is_routed else ("soft",)
    predictions = trainer.predict(test_loader, routing_modes=modes)

    metrics_by_mode: Dict[str, Dict[str, Any]] = {}
    frames_by_mode: Dict[str, pd.DataFrame] = {}
    for mode, bundle in predictions.items():
        metrics = bundle.metrics()
        if bundle.router_probs is not None:
            core = model
            if hasattr(core, "expert_utilisation"):
                import torch

                metrics.update(
                    core.expert_utilisation(torch.as_tensor(bundle.router_probs))
                )
        metrics_by_mode[mode] = metrics
        frames_by_mode[mode] = bundle.to_frame()

    extra = {
        "fold_fingerprint": fold_fingerprint,
        "config_fingerprint": store.config_fingerprint(cfg),
        "seed": seed,
        "device": str(device),
        "best_epoch": trainer.best_epoch,
        "best_val_monitor": trainer.best_metric,
        "epochs_run": len(history),
        "n_train_slices": int(len(train_frame)),
        "n_val_slices": int(len(val_frame)),
        "n_test_slices": int(len(test_frame)),
        "n_train_patients": int(train_frame["patient_uid"].nunique()),
        "n_test_patients": int(test_frame["patient_uid"].nunique()),
        "train_class_counts": {
            cluster: int((train_frame["cluster"] == cluster).sum()) for cluster in active_classes()
        },
        "test_class_counts": {
            cluster: int((test_frame["cluster"] == cluster).sum()) for cluster in active_classes()
        },
        "expert_effective_n": _expert_sample_sizes(train_frame),
    }

    store.save_fold_result(
        cfg.output_dir,
        cfg.experiment,
        fold.repeat,
        fold.fold,
        metrics_by_mode=metrics_by_mode,
        predictions_by_mode=frames_by_mode,
        history=history,
        extra=extra,
    )

    # Weights, only when explicitly asked for. Trainer.fit() has already restored the
    # best-epoch state into the model, so this saves the same weights that produced the
    # metrics above rather than the last epoch's.
    if bool(getattr(cfg.runtime, "save_checkpoint", False)):
        import torch

        from ppgl.utils import distributed as dist_utils

        target = os.path.join(
            store.fold_dir(cfg.output_dir, cfg.experiment, fold.repeat, fold.fold),
            "model.pt",
        )
        torch.save(
            {
                "model": dist_utils.unwrap_model(model).state_dict(),
                "experiment": cfg.experiment,
                "repeat": fold.repeat,
                "fold": fold.fold,
                "best_epoch": trainer.best_epoch,
                "fold_fingerprint": fold_fingerprint,
                "config_fingerprint": store.config_fingerprint(cfg),
            },
            target,
        )
        log.info("saved checkpoint %s", target)

    primary = metrics_by_mode.get("soft", {})
    log.info(
        "fold %s done | patient macro-F1=%.4f slice macro-F1=%.4f AUC=%.4f",
        fold.tag,
        primary.get("patient_macro_f1", float("nan")),
        primary.get("macro_f1", float("nan")),
        primary.get("macro_auc_ovr", float("nan")),
    )
    return metrics_by_mode


def _expert_sample_sizes(train_frame: pd.DataFrame) -> Dict[str, Any]:
    """Training slices available per anatomical expert, and per (region, cluster) cell.

    Reported because head & neck and chest are small enough that a dedicated
    expert cannot be well estimated, and because head & neck contains no
    kinase-signalling cases at all. Stating this pre-empts the obvious
    mixture-of-experts criticism.
    """
    if "region_idx" not in train_frame.columns:
        return {}
    out: Dict[str, Any] = {}
    for idx, region in enumerate(REGIONS):
        subset = train_frame[train_frame["region_idx"] == idx]
        out[region] = {
            "n_slices": int(len(subset)),
            "n_patients": int(subset["patient_uid"].nunique()) if len(subset) else 0,
            "per_cluster": {
                cluster: int((subset["cluster"] == cluster).sum()) for cluster in active_classes()
            },
        }
    return out


def run_cv(
    cfg,
    repeats: Optional[Sequence[int]] = None,
    fold_ids: Optional[Sequence[int]] = None,
    device_index: Optional[int] = None,
) -> pd.DataFrame:
    """Run every requested fold for one configuration, in this process."""
    frame = prepare_frame(cfg)
    folds = build_folds(cfg, frame)
    fold_fingerprint = fingerprint(folds)

    store.save_run_metadata(cfg.output_dir, cfg.experiment, cfg, describe_hardware())

    selected = [
        fold
        for fold in folds
        if (repeats is None or fold.repeat in set(repeats))
        and (fold_ids is None or fold.fold in set(fold_ids))
    ]
    log.info("running %d of %d folds for experiment %r", len(selected), len(folds), cfg.experiment)

    expected = store.config_fingerprint(cfg)
    for fold in selected:
        if cfg.runtime.resume and store.is_fold_complete(
            cfg.output_dir, cfg.experiment, fold.repeat, fold.fold, expected_fingerprint=expected
        ):
            log.info("skipping completed fold %s", fold.tag)
            continue
        run_fold(cfg, frame, fold, device_index=device_index, fold_fingerprint=fold_fingerprint)

    for mode in store.available_modes(cfg.output_dir, cfg.experiment):
        store.summarise_experiment(cfg.output_dir, cfg.experiment, mode=mode)
    return store.load_fold_metrics(cfg.output_dir, cfg.experiment, mode="soft")


__all__ = ["build_folds", "prepare_frame", "run_cv", "run_fold"]
