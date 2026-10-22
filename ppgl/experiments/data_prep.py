"""Manifest loading, region attachment and fold construction.

Separated from the runners so that both the deep and the tabular paths share one
implementation of "which slices, in which folds", and so that neither the folds
nor the tabular controls require torch to be installed.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from ppgl.data.inventory import load_manifest
from ppgl.data.location import attach_regions
from ppgl.data.splits import Fold, make_folds
from ppgl.labels import apply_label_space, set_active_label_space
from ppgl.models.names import needs_clinical, needs_region_labels
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


def prepare_frame(cfg) -> pd.DataFrame:
    """Load the manifest, map to the active label space, attach side information.

    Order matters: the label space is applied before folds are built, so
    stratification and the all-classes-present assertion operate on the target
    the model will actually be trained on.
    """
    frame = load_manifest(cfg.data.manifest)

    # Process-wide, so metrics, tables and figures name columns consistently.
    set_active_label_space(cfg.data.label_space)
    frame = apply_label_space(frame, cfg.data.label_space)

    needs_regions = needs_region_labels(cfg)
    if needs_regions or cfg.data.location_source != "none":
        frame = attach_regions(frame, cfg.data)
        if needs_regions and bool(frame.get("provisional", pd.Series([False])).any()):
            log.warning(
                "model %r depends on anatomical region labels and is using PROVISIONAL "
                "heuristic labels. Results are for pipeline validation only; supply "
                "radiologist labels via data.location_csv before reporting numbers.",
                cfg.model.name,
            )
    else:
        frame = frame.copy()
        frame["region_idx"] = -1
        frame["region_confidence"] = 0.0

    if needs_clinical(cfg.model.name) or cfg.data.clinical_csv:
        from ppgl.data.clinical import attach_clinical

        frame = attach_clinical(frame, cfg.data, required=needs_clinical(cfg.model.name))

    if cfg.smoke_test:
        frame = smoke_subset(frame)
    return frame


def smoke_subset(frame: pd.DataFrame, per_class: int = 12) -> pd.DataFrame:
    """A tiny class-balanced subset for wiring checks.

    Whole patients are taken, never individual slices, so the smoke path cannot
    accidentally introduce the patient-level leakage the real splits avoid.
    """
    keep_patients: List[str] = []
    for _, group in frame.sort_values("patient_uid").groupby("cluster", sort=True):
        patients = list(dict.fromkeys(group["patient_uid"].tolist()))
        keep_patients.extend(patients[:per_class])
    subset = frame[frame["patient_uid"].isin(keep_patients)].reset_index(drop=True)
    log.warning(
        "SMOKE TEST: %d of %d slices from %d of %d patients",
        len(subset),
        len(frame),
        subset["patient_uid"].nunique(),
        frame["patient_uid"].nunique(),
    )
    return subset


def build_folds(cfg, frame: pd.DataFrame) -> List[Fold]:
    return make_folds(
        frame,
        n_splits=int(cfg.cv.n_splits),
        n_repeats=int(cfg.cv.n_repeats),
        seed=int(cfg.cv.seed),
        inner_val_fraction=float(cfg.cv.inner_val_fraction),
        require_all_classes=bool(cfg.cv.require_all_classes_per_fold),
    )


__all__ = ["build_folds", "prepare_frame", "smoke_subset"]
