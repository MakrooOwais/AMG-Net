"""Statistical comparison of models evaluated on identical folds.

Two things drive the choices here.

**The naive paired t-test over CV folds is invalid.** Folds share training data,
so per-fold scores are dependent and the plain paired t-test has badly inflated
false-positive rates (Zeng et al., bioRxiv 2026). An earlier version of this work used
exactly that test, and only for its ablation.

**The corrected resampled t-test is the convention in this literature.** Nadeau &
Bengio's correction inflates the variance by the train/test size ratio and is
what comparable radiomics work uses - GIST molecular stratification from CT
(Starmans et al., J. Digital Imaging 2022) and MRI radiomics (Jansma et al.,
Cancers 2024). Adopting it makes the analysis legible to this reviewer pool.

Across a family of baseline comparisons, p-values are corrected with
Holm-Bonferroni.

Comparisons are refused unless both experiments carry the same fold fingerprint,
which prevents silently comparing models that saw different splits - the exact
problem created by quoting baseline numbers from other papers rather than
re-running them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ppgl.experiments import results as store
from ppgl.metrics import HIGHER_IS_BETTER, PRIMARY_METRIC
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ComparisonResult:
    """One model-vs-model comparison on one metric."""

    metric: str
    model_a: str
    model_b: str
    n_folds: int
    mean_a: float
    mean_b: float
    mean_difference: float
    std_difference: float
    t_statistic: float
    p_value: float
    df: int
    ci_low: float
    ci_high: float
    effect_size: float
    correction: str = "none"
    p_adjusted: Optional[float] = None
    significant: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.__dict__.copy()
        payload["notes"] = list(self.notes)
        return payload


def corrected_resampled_ttest(
    differences: Sequence[float],
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Nadeau & Bengio corrected resampled t-test.

    The variance of the mean difference is inflated from ``sigma^2 / n`` to
    ``sigma^2 * (1/n + n_test/n_train)`` to account for the dependence induced by
    overlapping training sets across folds.

    Parameters
    ----------
    differences
        Per-fold differences ``metric_a - metric_b``.
    n_train, n_test
        Mean training and test set sizes, used for the correction term.
    """
    from scipy import stats

    values = np.asarray([d for d in differences if d is not None and np.isfinite(d)], dtype=np.float64)
    n = values.size
    if n < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "df": max(n - 1, 0),
            "mean_difference": float(values.mean()) if n else float("nan"),
            "std_difference": 0.0,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "effect_size": float("nan"),
        }

    mean_difference = float(values.mean())
    variance = float(values.var(ddof=1))
    ratio = (float(n_test) / float(n_train)) if n_train > 0 else 0.0
    corrected_variance = variance * (1.0 / n + ratio)

    if corrected_variance <= 0:
        # Identical scores on every fold.
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "df": n - 1,
            "mean_difference": mean_difference,
            "std_difference": float(np.sqrt(variance)),
            "ci_low": mean_difference,
            "ci_high": mean_difference,
            "effect_size": 0.0,
        }

    standard_error = float(np.sqrt(corrected_variance))
    t_statistic = mean_difference / standard_error
    df = n - 1
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    return {
        "t_statistic": float(t_statistic),
        "p_value": p_value,
        "df": int(df),
        "mean_difference": mean_difference,
        "std_difference": float(np.sqrt(variance)),
        "ci_low": mean_difference - critical * standard_error,
        "ci_high": mean_difference + critical * standard_error,
        # Cohen's d on the paired differences, uncorrected.
        "effect_size": float(mean_difference / np.sqrt(variance)) if variance > 0 else 0.0,
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values and rejections."""
    try:
        from statsmodels.stats.multitest import multipletests

        reject, adjusted, _, _ = multipletests(list(p_values), alpha=alpha, method="holm")
        return [float(p) for p in adjusted], [bool(r) for r in reject]
    except Exception:
        pass

    values = np.asarray(p_values, dtype=np.float64)
    n = values.size
    if n == 0:
        return [], []
    order = np.argsort(values)
    adjusted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * values[index]
        running = max(running, min(candidate, 1.0))
        adjusted[index] = running
    return [float(p) for p in adjusted], [bool(p <= alpha) for p in adjusted]


def _fold_key(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["_key"] = out["repeat"].astype(str) + ":" + out["fold"].astype(str)
    return out


def _fingerprints(table: pd.DataFrame) -> set:
    if "fold_fingerprint" not in table.columns:
        return set()
    return {value for value in table["fold_fingerprint"].dropna().unique() if value}


def compare(
    output_dir: str,
    model_a: str,
    model_b: str,
    metric: str = PRIMARY_METRIC,
    mode: str = "soft",
    alpha: float = 0.05,
    require_matching_folds: bool = True,
) -> Optional[ComparisonResult]:
    """Compare two experiments on one metric across their shared folds."""
    table_a = store.load_fold_metrics(output_dir, model_a, mode=mode)
    table_b = store.load_fold_metrics(output_dir, model_b, mode=mode)

    if table_a.empty or table_b.empty:
        log.warning("cannot compare %s vs %s: missing results", model_a, model_b)
        return None
    if metric not in table_a.columns or metric not in table_b.columns:
        log.warning("metric %r absent for %s or %s", metric, model_a, model_b)
        return None

    notes: List[str] = []
    fingerprints_a, fingerprints_b = _fingerprints(table_a), _fingerprints(table_b)
    if fingerprints_a and fingerprints_b and fingerprints_a != fingerprints_b:
        message = (
            f"fold fingerprints differ ({sorted(fingerprints_a)} vs {sorted(fingerprints_b)}); "
            "these models were not evaluated on identical splits"
        )
        if require_matching_folds:
            raise ValueError(f"refusing to compare {model_a} with {model_b}: {message}")
        notes.append(message)

    keyed_a, keyed_b = _fold_key(table_a), _fold_key(table_b)
    merged = keyed_a[["_key", metric]].merge(
        keyed_b[["_key", metric]], on="_key", suffixes=("_a", "_b")
    )
    if merged.empty:
        log.warning("no shared folds between %s and %s", model_a, model_b)
        return None
    if len(merged) < len(keyed_a) or len(merged) < len(keyed_b):
        notes.append(
            f"compared on {len(merged)} shared folds "
            f"({len(keyed_a)} for {model_a}, {len(keyed_b)} for {model_b})"
        )

    differences = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy(dtype=np.float64)
    n_train = int(_mean_column(table_a, "n_train_slices") or 0)
    n_test = int(_mean_column(table_a, "n_test_slices") or 0)
    if n_train <= 0 or n_test <= 0:
        notes.append("train/test sizes unavailable; correction term omitted (n_test/n_train=0)")

    statistics = corrected_resampled_ttest(differences, n_train=n_train, n_test=n_test, alpha=alpha)

    if not HIGHER_IS_BETTER.get(metric.replace("patient_", ""), True):
        notes.append("lower is better for this metric; sign of the difference is inverted in meaning")

    return ComparisonResult(
        metric=metric,
        model_a=model_a,
        model_b=model_b,
        n_folds=int(len(merged)),
        mean_a=float(merged[f"{metric}_a"].mean()),
        mean_b=float(merged[f"{metric}_b"].mean()),
        mean_difference=statistics["mean_difference"],
        std_difference=statistics["std_difference"],
        t_statistic=statistics["t_statistic"],
        p_value=statistics["p_value"],
        df=int(statistics["df"]),
        ci_low=statistics["ci_low"],
        ci_high=statistics["ci_high"],
        effect_size=statistics["effect_size"],
        notes=notes,
    )


def _mean_column(table: pd.DataFrame, column: str) -> Optional[float]:
    if column not in table.columns:
        return None
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def compare_against(
    output_dir: str,
    reference: str,
    others: Sequence[str],
    metrics: Sequence[str] = (PRIMARY_METRIC, f"patient_{PRIMARY_METRIC}", "macro_auc_ovr", "mcc"),
    mode: str = "soft",
    alpha: float = 0.05,
    require_matching_folds: bool = True,
) -> pd.DataFrame:
    """Compare ``reference`` against each of ``others``, correcting within each metric family."""
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        family: List[ComparisonResult] = []
        for other in others:
            if other == reference:
                continue
            try:
                result = compare(
                    output_dir,
                    reference,
                    other,
                    metric=metric,
                    mode=mode,
                    alpha=alpha,
                    require_matching_folds=require_matching_folds,
                )
            except ValueError as exc:
                log.error("%s", exc)
                continue
            if result is not None:
                family.append(result)

        if not family:
            continue
        adjusted, rejected = holm_bonferroni([r.p_value for r in family], alpha=alpha)
        for result, p_adjusted, is_significant in zip(family, adjusted, rejected):
            result.correction = f"holm (family of {len(family)})"
            result.p_adjusted = float(p_adjusted)
            result.significant = bool(is_significant)
            rows.append(result.to_dict())

    return pd.DataFrame(rows)


def paired_mode_test(
    output_dir: str,
    experiment: str,
    metric: str = f"patient_{PRIMARY_METRIC}",
    mode_a: str = "oracle",
    mode_b: str = "soft",
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Test the routing gap: the same weights scored under two routing rules.

    This is a within-model comparison, so the training sets are literally
    identical across the two arms and the correction term is conservative here
    rather than anti-conservative.
    """
    table_a = store.load_fold_metrics(output_dir, experiment, mode=mode_a)
    table_b = store.load_fold_metrics(output_dir, experiment, mode=mode_b)
    if table_a.empty or table_b.empty or metric not in table_a.columns:
        log.warning("cannot test %s vs %s for %s", mode_a, mode_b, experiment)
        return None

    merged = _fold_key(table_a)[["_key", metric]].merge(
        _fold_key(table_b)[["_key", metric]], on="_key", suffixes=("_a", "_b")
    )
    if merged.empty:
        return None

    differences = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy(dtype=np.float64)
    statistics = corrected_resampled_ttest(
        differences,
        n_train=int(_mean_column(table_a, "n_train_slices") or 0),
        n_test=int(_mean_column(table_a, "n_test_slices") or 0),
        alpha=alpha,
    )
    return {
        "experiment": experiment,
        "metric": metric,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "n_folds": int(len(merged)),
        f"mean_{mode_a}": float(merged[f"{metric}_a"].mean()),
        f"mean_{mode_b}": float(merged[f"{metric}_b"].mean()),
        **statistics,
    }


def run_all(
    output_dir: str,
    reference: str,
    others: Sequence[str],
    metrics: Sequence[str] = (PRIMARY_METRIC, f"patient_{PRIMARY_METRIC}", "macro_auc_ovr", "mcc"),
    mode: str = "soft",
    alpha: float = 0.05,
    save_dir: Optional[str] = None,
    require_matching_folds: bool = True,
) -> pd.DataFrame:
    """Full comparison sweep, written to disk."""
    table = compare_against(
        output_dir,
        reference,
        others,
        metrics=metrics,
        mode=mode,
        alpha=alpha,
        require_matching_folds=require_matching_folds,
    )

    target = save_dir or os.path.join(output_dir, "analysis")
    os.makedirs(target, exist_ok=True)
    if not table.empty:
        path = os.path.join(target, f"comparisons_{reference}.csv")
        table.to_csv(path, index=False)
        log.info("wrote %d comparisons to %s", len(table), path)
        for row in table.itertuples(index=False):
            log.info(
                "%s: %s vs %s  d=%+.4f  p=%.4g  p_holm=%.4g%s",
                row.metric,
                row.model_a,
                row.model_b,
                row.mean_difference,
                row.p_value,
                row.p_adjusted if row.p_adjusted is not None else float("nan"),
                "  *" if row.significant else "",
            )

    gaps = []
    for mode_a, mode_b in (("oracle", "soft"), ("argmax", "soft"), ("oracle", "argmax")):
        result = paired_mode_test(output_dir, reference, mode_a=mode_a, mode_b=mode_b, alpha=alpha)
        if result:
            gaps.append(result)
    if gaps:
        with open(os.path.join(target, f"routing_gap_tests_{reference}.json"), "w") as fh:
            json.dump(store.jsonable(gaps), fh, indent=2, sort_keys=True)

    return table


__all__ = [
    "ComparisonResult",
    "compare",
    "compare_against",
    "corrected_resampled_ttest",
    "holm_bonferroni",
    "paired_mode_test",
    "run_all",
]
