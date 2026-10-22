"""Selective prediction: risk-coverage analysis from stored probabilities.

Motivation. The 4-class task saturates around 0.41 patient macro-F1 for every
architecture tried, and the anatomy-routed comparison model
measured as null. A model that must answer for every patient is therefore not
clinically useful. A model that answers confidently for part of the cohort and
defers the rest to genetic testing can be, and that is a question about the
*ranking* of the existing probabilities rather than about a better model. It
needs no retraining: every fold already wrote its per-slice probabilities.

Everything here reads :mod:`ppgl.experiments.results` and writes CSV/JSON. No
GPU, no model, no data access.

Definitions used, following the failure-prediction literature (El-Yaniv &
Wiener 2010; Geifman & El-Yaniv 2017; Geifman et al., "Bias-Reduced Uncertainty
Estimation", ICLR 2019):

Sort the ``n`` test items by confidence, most confident first. For each prefix of
size ``k``:

    coverage_k = k / n
    risk_k     = errors in the prefix / k

``AURC`` is the mean of ``risk_k`` over all ``k`` - the area under the
risk-coverage curve. It is not comparable across models with different base
error rates, so ``E-AURC`` is also reported: AURC minus the AURC of an oracle
ranking that places every correct prediction ahead of every error. E-AURC is
zero for a perfect confidence ranking and isolates ranking quality from
accuracy.

**The trap this module is written to expose.** SDHx is 56% of slices and 49% of
patients. A confidence score can lower selective risk simply by retaining
majority-class cases and deferring the rare clusters, which is clinically
worthless - those are precisely the patients for whom the prediction would have
changed management. Plain error rate cannot see this. So every coverage point
also reports the *balanced* error rate over retained classes, how many of the
label classes survive at that coverage, and the retained share of each class.
Read those columns before believing a risk-coverage curve on this dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ppgl.experiments import results as results_store
from ppgl.metrics import aggregate_to_patient
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

# Confidence scores. All are "higher means more confident".
#   msp    - maximum softmax probability. The standard baseline and hard to beat
#            (Hendrycks & Gimpel, ICLR 2017).
#   margin - top-1 minus top-2 probability. Sensitive to *which* alternative is
#            competing, which matters here because the confusable pairs are
#            specific (SDHx vs VHL/EPAS1, both pseudohypoxic).
#   negent - negative Shannon entropy of the full distribution. Uses all classes
#            rather than the top one or two.
CONFIDENCE_SCORES = ("msp", "margin", "negent")

# Coverage grid for the averaged curve. A fold holds ~29 patients, so a finer
# grid than this would interpolate between points that do not exist.
COVERAGE_GRID = tuple(np.round(np.arange(0.10, 1.0001, 0.05), 4))

PROB_PREFIX = "prob_"


@dataclass
class SelectiveSummary:
    """Headline numbers for one (experiment, level, score) combination."""

    experiment: str
    level: str
    score: str
    n_folds: int
    full_coverage_error: float
    aurc: float
    eaurc: float
    failure_auroc: float
    error_at_50: float
    error_at_80: float
    coverage_at_error_10: float
    coverage_at_error_20: float
    balanced_error_at_50: float
    classes_retained_at_50: float

    def as_row(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment,
            "level": self.level,
            "score": self.score,
            "n_folds": self.n_folds,
            "full_coverage_error": self.full_coverage_error,
            "aurc": self.aurc,
            "eaurc": self.eaurc,
            "failure_auroc": self.failure_auroc,
            "error_at_coverage_50": self.error_at_50,
            "error_at_coverage_80": self.error_at_80,
            "coverage_at_error_10": self.coverage_at_error_10,
            "coverage_at_error_20": self.coverage_at_error_20,
            "balanced_error_at_coverage_50": self.balanced_error_at_50,
            "classes_retained_at_coverage_50": self.classes_retained_at_50,
        }


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def confidence_from_probabilities(probabilities: np.ndarray, score: str = "msp") -> np.ndarray:
    """Per-row confidence, higher meaning more confident."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2-D (N, C)")

    if score == "msp":
        return probabilities.max(axis=1)
    if score == "margin":
        if probabilities.shape[1] < 2:
            raise ValueError("margin needs at least 2 classes")
        ordered = np.sort(probabilities, axis=1)
        return ordered[:, -1] - ordered[:, -2]
    if score == "negent":
        safe = np.clip(probabilities, 1e-12, None)
        return (safe * np.log(safe)).sum(axis=1)
    raise ValueError(f"unknown confidence score {score!r}; expected one of {CONFIDENCE_SCORES}")


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def _prefix_risks(correct: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Risk at every prefix size, ordered by descending confidence.

    Ties are broken by putting errors first within a tie group, which is the
    pessimistic choice. Optimistic tie-breaking makes a score that outputs a
    constant look informative.
    """
    order = np.lexsort((correct.astype(np.int8), -confidence))
    ordered_correct = correct[order]
    k = np.arange(1, ordered_correct.size + 1)
    errors = np.cumsum(~ordered_correct)
    return k / ordered_correct.size, errors / k


def _oracle_aurc(correct: np.ndarray) -> float:
    """AURC of a ranking that puts every correct prediction ahead of every error."""
    n = correct.size
    n_errors = int((~correct).sum())
    n_correct = n - n_errors
    k = np.arange(1, n + 1)
    errors = np.clip(k - n_correct, 0, None)
    return float(np.mean(errors / k))


def _failure_auroc(correct: np.ndarray, confidence: np.ndarray) -> float:
    """AUROC for detecting a misclassification from low confidence.

    0.5 means confidence carries no information about correctness. This is the
    single cleanest summary of whether selective prediction can work at all,
    because it is independent of where the operating point is put.
    """
    if correct.all() or (~correct).all():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score((~correct).astype(int), -confidence))


def _balanced_error(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int]:
    """Macro-averaged error over the classes present, and how many are present."""
    present = np.unique(y_true)
    if present.size == 0:
        return float("nan"), 0
    per_class = [
        1.0 - float((y_pred[y_true == cls] == cls).mean()) for cls in present
    ]
    return float(np.mean(per_class)), int(present.size)


def fold_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    score: str = "msp",
    grid: Sequence[float] = COVERAGE_GRID,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Risk-coverage curve and summary statistics for one fold."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_pred = probabilities.argmax(axis=1)
    correct = y_pred == y_true
    confidence = confidence_from_probabilities(probabilities, score)

    coverage, risk = _prefix_risks(correct, confidence)
    aurc = float(np.mean(risk))
    oracle = _oracle_aurc(correct)

    statistics = {
        "n": int(y_true.size),
        "full_coverage_error": float(1.0 - correct.mean()),
        "aurc": aurc,
        "oracle_aurc": oracle,
        "eaurc": aurc - oracle,
        "failure_auroc": _failure_auroc(correct, confidence),
    }

    # Evaluate on the shared grid so folds can be averaged.
    order = np.lexsort((correct.astype(np.int8), -confidence))
    n = y_true.size
    n_classes = probabilities.shape[1]
    rows: List[Dict[str, Any]] = []
    for target in grid:
        k = max(1, int(np.ceil(float(target) * n)))
        keep = order[:k]
        kept_true, kept_pred = y_true[keep], y_pred[keep]
        balanced, n_present = _balanced_error(kept_true, kept_pred)
        row: Dict[str, Any] = {
            "coverage_target": float(target),
            "coverage_actual": k / n,
            "n_retained": k,
            "error": float(1.0 - (kept_pred == kept_true).mean()),
            "balanced_error": balanced,
            "classes_retained": n_present,
        }
        # Retained share per class: exposes selectivity that just drops the rare
        # clusters. A class whose share collapses is being deferred wholesale.
        for cls in range(n_classes):
            total = int((y_true == cls).sum())
            row[f"retained_frac_class{cls}"] = (
                float((kept_true == cls).sum() / total) if total else float("nan")
            )
        rows.append(row)

    return pd.DataFrame(rows), statistics


def _coverage_at_error(curve: pd.DataFrame, target: float) -> float:
    """Largest averaged coverage whose error is at or below ``target``."""
    ok = curve[curve["error"] <= target]
    return float(ok["coverage_target"].max()) if len(ok) else float("nan")


def _at_coverage(curve: pd.DataFrame, coverage: float, column: str) -> float:
    row = curve.iloc[(curve["coverage_target"] - coverage).abs().argmin()]
    return float(row[column])


# --------------------------------------------------------------------------- #
# Per-experiment analysis
# --------------------------------------------------------------------------- #
def _load_fold_frames(
    output_dir: str, experiment: str, mode: str
) -> List[Tuple[str, pd.DataFrame, List[str]]]:
    folds_root = os.path.join(results_store.run_dir(output_dir, experiment), "folds")
    if not os.path.isdir(folds_root):
        return []
    out = []
    for tag in sorted(os.listdir(folds_root)):
        path = os.path.join(folds_root, tag, f"predictions_{mode}.csv")
        if not os.path.isfile(path):
            continue
        frame = pd.read_csv(path)
        columns = [c for c in frame.columns if c.startswith(PROB_PREFIX)]
        if columns and "y_true" in frame.columns:
            out.append((tag, frame, columns))
    return out


def analyse_experiment(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
    scores: Sequence[str] = CONFIDENCE_SCORES,
    levels: Sequence[str] = ("slice", "patient"),
    grid: Sequence[float] = COVERAGE_GRID,
) -> Tuple[pd.DataFrame, List[SelectiveSummary]]:
    """Risk-coverage curves for one experiment, averaged over its folds.

    Folds are the unit of analysis: each fold's curve is computed on its own test
    split and the curves are then averaged. Pooling every fold's predictions into
    one ranking instead would compare confidences produced by different models,
    and repeated CV means a patient appears in several folds, so the pooled set
    would silently weight patients by how often they were held out.
    """
    frames = _load_fold_frames(output_dir, experiment, mode)
    if not frames:
        log.warning("no predictions for %s (mode=%s)", experiment, mode)
        return pd.DataFrame(), []

    curves: List[pd.DataFrame] = []
    summaries: List[SelectiveSummary] = []

    for level in levels:
        for score in scores:
            per_fold: List[pd.DataFrame] = []
            statistics: List[Dict[str, float]] = []
            for tag, frame, columns in frames:
                y_true = frame["y_true"].to_numpy(dtype=int)
                probabilities = frame[columns].to_numpy(dtype=np.float64)

                if level == "patient":
                    if "patient_uid" not in frame.columns:
                        continue
                    y_true, probabilities, _ = aggregate_to_patient(
                        y_true, probabilities, frame["patient_uid"].tolist()
                    )
                if y_true.size < 2:
                    continue

                curve, stats = fold_curve(y_true, probabilities, score=score, grid=grid)
                curve.insert(0, "fold_tag", tag)
                per_fold.append(curve)
                statistics.append(stats)

            if not per_fold:
                continue

            stacked = pd.concat(per_fold, ignore_index=True)
            numeric = [c for c in stacked.columns if c not in ("fold_tag",)]
            averaged = stacked[numeric].groupby("coverage_target", as_index=False).mean()
            averaged.insert(0, "experiment", experiment)
            averaged.insert(1, "level", level)
            averaged.insert(2, "score", score)
            averaged["n_folds"] = len(per_fold)
            curves.append(averaged)

            summaries.append(
                SelectiveSummary(
                    experiment=experiment,
                    level=level,
                    score=score,
                    n_folds=len(per_fold),
                    full_coverage_error=float(np.mean([s["full_coverage_error"] for s in statistics])),
                    aurc=float(np.mean([s["aurc"] for s in statistics])),
                    eaurc=float(np.mean([s["eaurc"] for s in statistics])),
                    failure_auroc=float(
                        np.nanmean([s["failure_auroc"] for s in statistics])
                    ),
                    error_at_50=_at_coverage(averaged, 0.50, "error"),
                    error_at_80=_at_coverage(averaged, 0.80, "error"),
                    coverage_at_error_10=_coverage_at_error(averaged, 0.10),
                    coverage_at_error_20=_coverage_at_error(averaged, 0.20),
                    balanced_error_at_50=_at_coverage(averaged, 0.50, "balanced_error"),
                    classes_retained_at_50=_at_coverage(averaged, 0.50, "classes_retained"),
                )
            )

    combined = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return combined, summaries


def run_selective_analysis(
    output_dir: str,
    experiments: Sequence[str],
    mode: str = "soft",
    scores: Sequence[str] = CONFIDENCE_SCORES,
    levels: Sequence[str] = ("slice", "patient"),
    out_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse several experiments and write curves plus a summary table."""
    out_dir = out_dir or os.path.join(output_dir, "selective")
    os.makedirs(out_dir, exist_ok=True)

    all_curves: List[pd.DataFrame] = []
    all_rows: List[Dict[str, Any]] = []

    for experiment in experiments:
        curves, summaries = analyse_experiment(
            output_dir, experiment, mode=mode, scores=scores, levels=levels
        )
        if curves.empty:
            continue
        all_curves.append(curves)
        all_rows.extend(summary.as_row() for summary in summaries)
        log.info("selective analysis: %s (%d curve rows)", experiment, len(curves))

    curve_table = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    summary_table = pd.DataFrame(all_rows)

    if not curve_table.empty:
        curve_table.to_csv(os.path.join(out_dir, "risk_coverage_curves.csv"), index=False)
    if not summary_table.empty:
        summary_table = summary_table.sort_values(
            ["level", "score", "aurc"]
        ).reset_index(drop=True)
        summary_table.to_csv(os.path.join(out_dir, "selective_summary.csv"), index=False)
        with open(os.path.join(out_dir, "selective_summary.json"), "w") as fh:
            json.dump(
                results_store.jsonable(summary_table.to_dict(orient="records")),
                fh,
                indent=2,
                sort_keys=True,
            )
    return curve_table, summary_table


def plot_risk_coverage(
    curve_table: pd.DataFrame,
    out_path: str,
    level: str = "patient",
    score: str = "msp",
) -> Optional[str]:
    """Risk-coverage curves for every experiment, plus the balanced-error curve.

    Two panels deliberately: the left is the number a reader expects, the right
    is the one that shows whether the left was achieved by deferring rare
    clusters.
    """
    subset = curve_table[(curve_table["level"] == level) & (curve_table["score"] == score)]
    if subset.empty:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        log.warning("matplotlib unavailable, skipping plot (%s)", exc)
        return None

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for experiment, block in subset.groupby("experiment"):
        block = block.sort_values("coverage_target")
        axes[0].plot(block["coverage_target"], block["error"], marker="o", ms=3, label=experiment)
        axes[1].plot(
            block["coverage_target"], block["balanced_error"], marker="o", ms=3, label=experiment
        )

    axes[0].set_title(f"Risk-coverage ({level} level, {score})")
    axes[0].set_ylabel("error rate on retained cases")
    axes[1].set_title("Balanced error on retained cases")
    axes[1].set_ylabel("macro error over retained classes")
    for axis in axes:
        axis.set_xlabel("coverage")
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")

    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


__all__ = [
    "CONFIDENCE_SCORES",
    "COVERAGE_GRID",
    "SelectiveSummary",
    "analyse_experiment",
    "confidence_from_probabilities",
    "fold_curve",
    "plot_risk_coverage",
    "run_selective_analysis",
]
