"""Evaluation metrics. One definition, applied to every model.

This module exists because of a specific defect in an earlier version of this
work. Its main table reported a per-class "balanced accuracy" that equalled per-class
recall for the proposed model (0.65/0.65, 0.49/0.49, 0.59/0.59, 0.59/0.59) but
*not* for the baselines (ResNet-101 SDHx: BA 0.55, recall 0.80). Two different
quantities were printed in one column, so the comparison was invalid. Here there
is exactly one implementation of each metric and every model - deep, radiomics,
or location-only - is scored through :func:`compute_metrics`.

Metric choices follow current guidance for small, imbalanced, multi-class
medical imaging (Mosquera et al., European Radiology 2024; Imani et al. 2026,
which finds MCC ranks classifiers more consistently than F1 or balanced
accuracy):

* ``macro_f1``  - primary
* ``mcc``       - secondary
* ``balanced_accuracy``, ``macro_auc_ovr`` - reported for all models
* per-class precision / recall / F1 / one-vs-rest AUC

Everything is computed both at slice level and at patient level (mean predicted
probability across a patient's slices), because 152 patients contribute one
slice while one contributes 22, so slice-level means are patient-weighted in a
way that needs to be visible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ppgl.labels import active_classes

PRIMARY_METRIC = "macro_f1"
# Metrics where a larger value is better. Used by early stopping and by the
# statistics module when deciding the sign of a difference.
HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "macro_f1": True,
    "macro_precision": True,
    "macro_recall": True,
    "mcc": True,
    "macro_auc_ovr": True,
    "cohen_kappa": True,
    "loss": False,
}
SUMMARY_METRICS = (
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "macro_auc_ovr",
    "macro_precision",
    "macro_recall",
    "accuracy",
)


def compute_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Full metric set for one set of predictions.

    Parameters
    ----------
    y_true
        Integer labels, shape ``(N,)``.
    y_prob
        Predicted probabilities, shape ``(N, C)``. Rows need not sum to one;
        they are renormalised.
    class_names
        Names for the ``C`` columns, used for per-class keys.
    prefix
        Prepended to every key, e.g. ``"patient_"``.

    Notes
    -----
    A class absent from ``y_true`` has an undefined one-vs-rest AUC. Such
    classes are excluded from ``macro_auc_ovr`` (which is then computed over the
    present classes only) and the count is reported in
    ``macro_auc_ovr_n_classes`` so a reader can tell when it happened.
    """
    class_names = tuple(class_names) if class_names is not None else active_classes()

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:
        raise ValueError("y_prob must be 2-D (N, C)")
    if y_prob.shape[0] != y_true.shape[0]:
        raise ValueError(f"y_true has {y_true.shape[0]} rows, y_prob has {y_prob.shape[0]}")

    n_classes = y_prob.shape[1]
    labels = list(range(n_classes))
    y_prob = _normalise_rows(y_prob)
    y_pred = y_prob.argmax(axis=1)

    out: Dict[str, Any] = {
        f"{prefix}n_samples": int(y_true.size),
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}mcc": float(matthews_corrcoef(y_true, y_pred)) if y_true.size else float("nan"),
        f"{prefix}cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out[f"{prefix}macro_precision"] = float(np.mean(precision))
    out[f"{prefix}macro_recall"] = float(np.mean(recall))
    out[f"{prefix}macro_f1"] = float(np.mean(f1))

    for idx, name in enumerate(class_names[:n_classes]):
        out[f"{prefix}precision_{name}"] = float(precision[idx])
        out[f"{prefix}recall_{name}"] = float(recall[idx])
        out[f"{prefix}f1_{name}"] = float(f1[idx])
        out[f"{prefix}support_{name}"] = int(support[idx])

    # Per-class one-vs-rest AUC, skipping classes with no positive or no
    # negative example in this fold.
    aucs: List[float] = []
    for idx, name in enumerate(class_names[:n_classes]):
        binary = (y_true == idx).astype(int)
        if binary.min() == binary.max():
            out[f"{prefix}auc_{name}"] = float("nan")
            continue
        try:
            value = float(roc_auc_score(binary, y_prob[:, idx]))
        except ValueError:
            value = float("nan")
        out[f"{prefix}auc_{name}"] = value
        if not np.isnan(value):
            aucs.append(value)

    out[f"{prefix}macro_auc_ovr"] = float(np.mean(aucs)) if aucs else float("nan")
    out[f"{prefix}macro_auc_ovr_n_classes"] = len(aucs)

    out[f"{prefix}confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


def _normalise_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 0.0, None)
    totals = probabilities.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    normalised = probabilities / safe
    # Rows that were entirely zero become uniform rather than NaN.
    dead = (totals <= 0).reshape(-1)
    if dead.any():
        normalised[dead] = 1.0 / probabilities.shape[1]
    return normalised


def aggregate_to_patient(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    patient_uids: Sequence[str],
    method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Collapse slice predictions to one prediction per patient.

    ``method`` is ``"mean"`` (mean probability, the default and the more stable
    choice at 1-22 slices per patient) or ``"max"`` (max probability per class).
    Raises if a patient carries more than one distinct label, which would mean
    the manifest is corrupt.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = _normalise_rows(np.asarray(y_prob, dtype=np.float64))
    patient_uids = np.asarray(patient_uids, dtype=object).reshape(-1)

    order: List[str] = []
    seen = set()
    for uid in patient_uids:
        if uid not in seen:
            seen.add(uid)
            order.append(uid)

    labels = np.zeros(len(order), dtype=int)
    probabilities = np.zeros((len(order), y_prob.shape[1]), dtype=np.float64)

    for row, uid in enumerate(order):
        mask = patient_uids == uid
        unique_labels = np.unique(y_true[mask])
        if unique_labels.size != 1:
            raise ValueError(f"patient {uid!r} has multiple labels {unique_labels.tolist()}")
        labels[row] = int(unique_labels[0])
        block = y_prob[mask]
        probabilities[row] = block.max(axis=0) if method == "max" else block.mean(axis=0)

    return labels, _normalise_rows(probabilities), order


def compute_all_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    patient_uids: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    patient_method: str = "mean",
) -> Dict[str, Any]:
    """Slice-level metrics, plus patient-level metrics when ``patient_uids`` is given.

    On a binary label space, sensitivity / specificity / PPV / NPV are added as
    well: a clinical reader of a pseudohypoxic-vs-rest model wants operating
    characteristics, not only macro-F1.
    """
    from ppgl.labels import binary_metrics

    out = compute_metrics(y_true, y_prob, class_names=class_names, prefix="")
    for key, value in binary_metrics(y_true, y_prob).items():
        out[key] = value

    if patient_uids is not None and len(patient_uids):
        p_true, p_prob, uids = aggregate_to_patient(
            y_true, y_prob, patient_uids, method=patient_method
        )
        out.update(compute_metrics(p_true, p_prob, class_names=class_names, prefix="patient_"))
        out["patient_n_patients"] = len(uids)
        for key, value in binary_metrics(p_true, p_prob).items():
            out[f"patient_{key}"] = value
    return out


def bootstrap_ci(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    metric: str = PRIMARY_METRIC,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Percentile bootstrap CI for one metric on a single prediction set.

    Used for reporting a single model's uncertainty. For *comparing* models
    across folds use :mod:`ppgl.experiments.stats`, which accounts for the
    dependence between overlapping training sets.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)

    point = compute_metrics(y_true, y_prob, class_names=class_names).get(metric, float("nan"))
    samples: List[float] = []
    n = y_true.size
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if np.unique(y_true[idx]).size < 2:
            continue
        try:
            value = compute_metrics(y_true[idx], y_prob[idx], class_names=class_names)[metric]
        except Exception:
            continue
        if value is not None and not np.isnan(value):
            samples.append(float(value))

    if not samples:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return {
        "point": float(point),
        "lo": float(np.percentile(samples, 100 * alpha / 2)),
        "hi": float(np.percentile(samples, 100 * (1 - alpha / 2))),
        "n_boot": len(samples),
    }


def region_metrics(
    region_true: Sequence[int],
    region_prob: np.ndarray,
    region_names: Sequence[str],
) -> Dict[str, Any]:
    """Metrics for the anatomical-region estimator.

    Rows whose region label is ``-1`` (unlabelled) are dropped.
    """
    region_true = np.asarray(region_true, dtype=int).reshape(-1)
    region_prob = np.asarray(region_prob, dtype=np.float64)
    mask = region_true >= 0
    if not mask.any():
        return {"region_n_samples": 0}
    out = compute_metrics(
        region_true[mask], region_prob[mask], class_names=region_names, prefix="region_"
    )
    return out


def summarise_folds(records: Sequence[Dict[str, Any]], metrics: Sequence[str] = SUMMARY_METRICS) -> Dict[str, Any]:
    """Mean, std and n across folds for the given metrics."""
    out: Dict[str, Any] = {"n_folds": len(records)}
    for metric in metrics:
        for scope in ("", "patient_"):
            key = f"{scope}{metric}"
            values = [
                float(r[key])
                for r in records
                if key in r and r[key] is not None and not _isnan(r[key])
            ]
            if not values:
                continue
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out[f"{key}_n"] = len(values)
    return out


def _isnan(value: Any) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


__all__ = [
    "HIGHER_IS_BETTER",
    "PRIMARY_METRIC",
    "SUMMARY_METRICS",
    "aggregate_to_patient",
    "bootstrap_ci",
    "compute_all_metrics",
    "compute_metrics",
    "region_metrics",
    "summarise_folds",
]
