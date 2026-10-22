"""Cross-validation runner for the non-deep controls.

Covers ``location_only``, ``region_majority``, ``radiomics_gbm`` and
``radiomics_gbm_region``. These use the identical folds, the identical metric
implementation, and the identical results store as the deep models, so their
numbers are directly comparable and can enter the same paired tests.

``location_only`` is the decision gate for the paper: it sees only which
anatomical region the lesion is in, and no image content at all. Run it first.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ppgl import REGIONS
from ppgl.labels import active_classes
from ppgl.data.radiomics import build_feature_table
from ppgl.data.splits import Fold, fingerprint
from ppgl.experiments import results as store
from ppgl.experiments.data_prep import build_folds, prepare_frame
from ppgl.metrics import compute_all_metrics, region_metrics
from ppgl.models.tabular import (
    LocationOnlyClassifier,
    RadiomicsGBMClassifier,
    RegionMajorityClassifier,
    TabularLogisticClassifier,
)
from ppgl.utils.device import describe_hardware
from ppgl.utils.logging import get_logger, setup_logging
from ppgl.utils.seed import job_seed, seed_everything

log = get_logger(__name__)

TABULAR_RUNNERS = (
    "location_only",
    "region_majority",
    "radiomics_gbm",
    "radiomics_gbm_region",
    "clinical_only",
    "clinical_radiomics",
)


def _build_estimator(name: str, seed: int, n_classes: int):
    if name == "location_only":
        return LocationOnlyClassifier(n_classes=n_classes, random_state=seed)
    if name == "region_majority":
        return RegionMajorityClassifier(n_classes=n_classes)
    if name == "radiomics_gbm":
        return RadiomicsGBMClassifier(n_classes=n_classes, random_state=seed, include_region=False)
    if name == "radiomics_gbm_region":
        return RadiomicsGBMClassifier(n_classes=n_classes, random_state=seed, include_region=True)
    if name == "clinical_only":
        return TabularLogisticClassifier(n_classes=n_classes, random_state=seed)
    if name == "clinical_radiomics":
        return RadiomicsGBMClassifier(n_classes=n_classes, random_state=seed, include_region=False)
    raise ValueError(f"unknown tabular model {name!r}. Known: {TABULAR_RUNNERS}")


def _needs_features(name: str) -> bool:
    """Whether this estimator consumes the radiomics feature table."""
    return name in ("radiomics_gbm", "radiomics_gbm_region", "clinical_radiomics")


def _needs_clinical_matrix(name: str) -> bool:
    return name in ("clinical_only", "clinical_radiomics")


def run_fold(
    cfg,
    frame: pd.DataFrame,
    fold: Fold,
    features: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    fold_fingerprint: str = "",
) -> Dict[str, Any]:
    """Fit and score one fold for a tabular control."""
    name = cfg.model.name
    directory = store.fold_dir(cfg.output_dir, cfg.experiment, fold.repeat, fold.fold)
    os.makedirs(directory, exist_ok=True)
    setup_logging(log_file=os.path.join(directory, "log.txt"), job_tag=f"{cfg.experiment}/{fold.tag}")

    seed = job_seed(int(cfg.runtime.seed), fold.repeat, fold.fold)
    seed_everything(seed, deterministic=True)

    # The inner validation split is not needed by these estimators, so it is
    # folded back into training. The test split is untouched and identical to
    # the deep models' test split.
    train_idx = np.concatenate([fold.train_idx, fold.val_idx])
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[fold.test_idx].reset_index(drop=True)

    y_train = train_frame["cluster_idx"].to_numpy(dtype=int)
    y_test = test_frame["cluster_idx"].to_numpy(dtype=int)
    regions_train = train_frame.get("region_idx", pd.Series(np.full(len(train_frame), -1))).to_numpy(dtype=int)
    regions_test = test_frame.get("region_idx", pd.Series(np.full(len(test_frame), -1))).to_numpy(dtype=int)

    estimator = _build_estimator(name, seed, n_classes=len(active_classes()))

    if _needs_features(name) or _needs_clinical_matrix(name):
        blocks_train: List[np.ndarray] = []
        blocks_test: List[np.ndarray] = []

        if _needs_features(name):
            if features is None or feature_columns is None:
                raise ValueError(f"{name!r} requires a precomputed feature table")
            blocks_train.append(features.iloc[train_idx][feature_columns].to_numpy(dtype=np.float64))
            blocks_test.append(
                features.iloc[fold.test_idx][feature_columns].to_numpy(dtype=np.float64)
            )

        if _needs_clinical_matrix(name):
            # Fit imputation and scaling on this fold's training rows only.
            from ppgl.data.clinical import build_fold_matrices, clinical_columns

            if not clinical_columns(frame):
                raise ValueError(
                    f"{name!r} needs clinical covariates but none were attached; "
                    "set data.clinical_csv"
                )
            matrices, preprocessor = build_fold_matrices(
                frame, train_idx, {"train": train_idx, "test": fold.test_idx}
            )
            blocks_train.append(matrices["train"].astype(np.float64))
            blocks_test.append(matrices["test"].astype(np.float64))
            log.info("clinical block: %d features", preprocessor.n_features)

        x_train = np.hstack(blocks_train)
        x_test = np.hstack(blocks_test)
        estimator.fit(x_train, y_train, regions=regions_train)
        y_prob = estimator.predict_proba(x_test, regions=regions_test)
    else:
        if (regions_train < 0).all():
            raise ValueError(
                f"{name!r} needs anatomical region labels but none are available. "
                "Set data.location_source=csv with data.location_csv, or =heuristic "
                "for a provisional run."
            )
        estimator.fit(regions_train, y_train)
        y_prob = estimator.predict_proba(regions_test)

    metrics = compute_all_metrics(
        y_test,
        y_prob,
        patient_uids=test_frame["patient_uid"].tolist(),
        class_names=active_classes(),
    )
    if (regions_test >= 0).any():
        # A tabular control has no learned region estimator, but recording the
        # region distribution of the test fold keeps the metrics files uniform.
        onehot = np.zeros((len(regions_test), len(REGIONS)), dtype=np.float64)
        valid = regions_test >= 0
        onehot[np.arange(len(regions_test))[valid], regions_test[valid]] = 1.0
        metrics.update(region_metrics(regions_test, onehot, REGIONS))

    predictions = pd.DataFrame(
        {
            "slice_uid": test_frame["slice_uid"],
            "patient_uid": test_frame["patient_uid"],
            "y_true": y_test,
            "region_true": regions_test,
        }
    )
    for idx, cluster in enumerate(active_classes()):
        predictions[f"prob_{cluster}"] = y_prob[:, idx]

    extra = {
        "fold_fingerprint": fold_fingerprint,
        "config_fingerprint": store.config_fingerprint(cfg),
        "seed": seed,
        "device": "cpu",
        "n_train_slices": int(len(train_frame)),
        "n_test_slices": int(len(test_frame)),
        "n_train_patients": int(train_frame["patient_uid"].nunique()),
        "n_test_patients": int(test_frame["patient_uid"].nunique()),
        "n_features": int(len(feature_columns)) if feature_columns else 0,
    }

    store.save_fold_result(
        cfg.output_dir,
        cfg.experiment,
        fold.repeat,
        fold.fold,
        metrics_by_mode={"soft": metrics},
        predictions_by_mode={"soft": predictions},
        history=None,
        extra=extra,
    )

    log.info(
        "fold %s [%s] patient macro-F1=%.4f slice macro-F1=%.4f AUC=%.4f",
        fold.tag,
        name,
        metrics.get("patient_macro_f1", float("nan")),
        metrics.get("macro_f1", float("nan")),
        metrics.get("macro_auc_ovr", float("nan")),
    )
    return {"soft": metrics}


def run_cv(
    cfg,
    repeats: Optional[Sequence[int]] = None,
    fold_ids: Optional[Sequence[int]] = None,
    device_index: Optional[int] = None,
) -> pd.DataFrame:
    """Run all requested folds for one tabular control."""
    del device_index  # CPU only

    frame = prepare_frame(cfg)
    folds = build_folds(cfg, frame)
    fold_fingerprint = fingerprint(folds)

    store.save_run_metadata(cfg.output_dir, cfg.experiment, cfg, describe_hardware())

    features, feature_columns = None, None
    if _needs_features(cfg.model.name):
        cache = os.path.join(cfg.output_dir, "radiomics_features.csv")
        features, feature_columns = build_feature_table(frame, cfg.data, cache_csv=cache)
        log.info("radiomics feature table: %d slices x %d features", len(features), len(feature_columns))

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
        run_fold(
            cfg,
            frame,
            fold,
            features=features,
            feature_columns=feature_columns,
            fold_fingerprint=fold_fingerprint,
        )

    store.summarise_experiment(cfg.output_dir, cfg.experiment, mode="soft")
    return store.load_fold_metrics(cfg.output_dir, cfg.experiment, mode="soft")


__all__ = ["TABULAR_RUNNERS", "run_cv", "run_fold"]
