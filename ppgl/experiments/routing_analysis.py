"""Routing analysis for the anatomy-routed comparison model.

Depends on anatomical region labels, which the archive does not contain, so nothing
here feeds a reported result. Retained because the analysis is what established that
the routing mechanism is not worth pursuing.

Four questions, all answered from the stored per-fold predictions so no model is
retrained:

1. **How good is the region estimator?** Accuracy, macro-F1 and the confusion
   matrix of the router against the region labels.
2. **What does imperfect routing cost?** The same trained weights are scored
   under ground-truth ``oracle`` routing, hard ``argmax`` routing, and ``soft``
   router probabilities. ``oracle - soft`` is the routing gap: an upper bound on
   what a perfect region estimator would buy.
3. **Where does the loss come from?** Cluster performance split by whether the
   router's top-1 region was right, and per-region breakdowns.
4. **Are the experts actually used?** Mean routing mass and top-1 share per
   expert, alongside each expert's effective training sample size. With head &
   neck at 64 slices and no kinase-signalling cases, expert collapse is the
   expected outcome, not a risk - so it is measured and reported.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ppgl import REGIONS
from ppgl.labels import active_classes
from ppgl.experiments import results as store
from ppgl.metrics import compute_metrics
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

GAP_METRICS = ("macro_f1", "patient_macro_f1", "balanced_accuracy", "macro_auc_ovr", "mcc")


def routing_gap(
    output_dir: str,
    experiment: str,
    metrics: Sequence[str] = GAP_METRICS,
) -> pd.DataFrame:
    """Per-fold metrics under each routing mode, plus oracle-minus-soft deltas."""
    available = store.available_modes(output_dir, experiment)
    if "soft" not in available:
        log.warning("experiment %r has no 'soft' predictions; is it a routed model?", experiment)
        return pd.DataFrame()

    frames = []
    for mode in available:
        table = store.load_fold_metrics(output_dir, experiment, mode=mode)
        if table.empty:
            continue
        keep = ["repeat", "fold"] + [m for m in metrics if m in table.columns]
        subset = table[keep].copy()
        subset["mode"] = mode
        frames.append(subset)

    if not frames:
        return pd.DataFrame()

    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["repeat", "fold"], columns="mode", values=list(metrics))
    wide.columns = [f"{metric}__{mode}" for metric, mode in wide.columns]
    wide = wide.reset_index()

    for metric in metrics:
        oracle_col, soft_col, argmax_col = (
            f"{metric}__oracle",
            f"{metric}__soft",
            f"{metric}__argmax",
        )
        if oracle_col in wide.columns and soft_col in wide.columns:
            wide[f"{metric}__gap_oracle_soft"] = wide[oracle_col] - wide[soft_col]
        if argmax_col in wide.columns and soft_col in wide.columns:
            wide[f"{metric}__gap_argmax_soft"] = wide[argmax_col] - wide[soft_col]
        if oracle_col in wide.columns and argmax_col in wide.columns:
            wide[f"{metric}__gap_oracle_argmax"] = wide[oracle_col] - wide[argmax_col]
    return wide


def region_estimator_report(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
) -> Dict[str, Any]:
    """Accuracy and confusion of the router against the region labels."""
    predictions = store.load_predictions(output_dir, experiment, mode=mode)
    if predictions.empty:
        return {"n_samples": 0}

    router_columns = [f"router_{region}" for region in REGIONS]
    region_columns = [f"region_prob_{region}" for region in REGIONS]

    source: Optional[List[str]] = None
    if all(column in predictions.columns for column in router_columns):
        source = router_columns
    elif all(column in predictions.columns for column in region_columns):
        source = region_columns

    if source is None or "region_true" not in predictions.columns:
        return {"n_samples": 0, "note": "no router or region_true columns stored"}

    valid = predictions["region_true"] >= 0
    if not valid.any():
        return {"n_samples": 0, "note": "no ground-truth region labels"}

    y_true = predictions.loc[valid, "region_true"].to_numpy(dtype=int)
    y_prob = predictions.loc[valid, source].to_numpy(dtype=np.float64)

    report = compute_metrics(y_true, y_prob, class_names=REGIONS, prefix="region_")
    report["region_source_columns"] = source

    # Per-region recall, for the paper's routing table.
    predicted = y_prob.argmax(axis=1)
    per_region = {}
    for idx, region in enumerate(REGIONS):
        mask = y_true == idx
        per_region[region] = {
            "n": int(mask.sum()),
            "recall": float((predicted[mask] == idx).mean()) if mask.any() else float("nan"),
        }
    report["per_region"] = per_region
    return report


def routing_error_breakdown(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
) -> Dict[str, Any]:
    """Cluster performance split by whether the router routed the slice correctly.

    This is the mechanism behind the gap: if correctly-routed slices are
    classified much better than mis-routed ones, the router is the bottleneck.
    """
    predictions = store.load_predictions(output_dir, experiment, mode=mode)
    if predictions.empty or "region_true" not in predictions.columns:
        return {"n_samples": 0}

    router_columns = [f"router_{region}" for region in REGIONS]
    if not all(column in predictions.columns for column in router_columns):
        return {"n_samples": 0, "note": "no router probabilities stored"}

    probability_columns = [f"prob_{cluster}" for cluster in active_classes()]
    valid = predictions["region_true"] >= 0
    subset = predictions.loc[valid].copy()
    if subset.empty:
        return {"n_samples": 0}

    subset["router_argmax"] = subset[router_columns].to_numpy().argmax(axis=1)
    subset["routed_correctly"] = subset["router_argmax"] == subset["region_true"]

    out: Dict[str, Any] = {
        "n_samples": int(len(subset)),
        "routing_accuracy": float(subset["routed_correctly"].mean()),
    }
    for label, group in (("correct", True), ("incorrect", False)):
        block = subset[subset["routed_correctly"] == group]
        if block.empty:
            out[label] = {"n": 0}
            continue
        metrics = compute_metrics(
            block["y_true"].to_numpy(dtype=int),
            block[probability_columns].to_numpy(dtype=np.float64),
            class_names=active_classes(),
        )
        out[label] = {
            "n": int(len(block)),
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "mcc": metrics["mcc"],
        }
    if out.get("correct", {}).get("n") and out.get("incorrect", {}).get("n"):
        out["macro_f1_penalty_when_misrouted"] = (
            out["correct"]["macro_f1"] - out["incorrect"]["macro_f1"]
        )
    return out


def per_region_performance(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
) -> pd.DataFrame:
    """Cluster metrics computed separately within each anatomical region."""
    predictions = store.load_predictions(output_dir, experiment, mode=mode)
    if predictions.empty or "region_true" not in predictions.columns:
        return pd.DataFrame()

    probability_columns = [f"prob_{cluster}" for cluster in active_classes()]
    rows: List[Dict[str, Any]] = []
    for idx, region in enumerate(REGIONS):
        block = predictions[predictions["region_true"] == idx]
        if block.empty:
            rows.append({"region": region, "n_slices": 0})
            continue
        metrics = compute_metrics(
            block["y_true"].to_numpy(dtype=int),
            block[probability_columns].to_numpy(dtype=np.float64),
            class_names=active_classes(),
        )
        row: Dict[str, Any] = {
            "region": region,
            "n_slices": int(len(block)),
            "n_patients": int(block["patient_uid"].nunique()),
        }
        for key in ("macro_f1", "balanced_accuracy", "macro_auc_ovr", "mcc", "accuracy"):
            row[key] = metrics.get(key)
        for cluster in active_classes():
            row[f"support_{cluster}"] = metrics.get(f"support_{cluster}", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def expert_utilisation(output_dir: str, experiment: str, mode: str = "soft") -> Dict[str, Any]:
    """Mean routing mass and top-1 share per expert, plus effective sample sizes."""
    predictions = store.load_predictions(output_dir, experiment, mode=mode)
    out: Dict[str, Any] = {}

    router_columns = [f"router_{region}" for region in REGIONS]
    if not predictions.empty and all(column in predictions.columns for column in router_columns):
        matrix = predictions[router_columns].to_numpy(dtype=np.float64)
        top1 = matrix.argmax(axis=1)
        for idx, region in enumerate(REGIONS):
            out[f"router_mass_{region}"] = float(matrix[:, idx].mean())
            out[f"router_top1_share_{region}"] = float((top1 == idx).mean())
        # A collapsed router puts nearly all top-1 mass on one expert.
        shares = np.array([out[f"router_top1_share_{r}"] for r in REGIONS])
        out["router_top1_max_share"] = float(shares.max())
        out["router_collapsed"] = bool(shares.max() > 0.95)

    # Effective training sample size per expert, recorded per fold by run_cv.
    pattern = os.path.join(
        store.run_dir(output_dir, experiment), "folds", "*", store.METRICS_FILENAME
    )
    import glob

    per_expert: Dict[str, List[int]] = {region: [] for region in REGIONS}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except Exception:
            continue
        effective = payload.get("expert_effective_n") or {}
        for region, info in effective.items():
            if region in per_expert and isinstance(info, dict):
                per_expert[region].append(int(info.get("n_slices", 0)))
    for region, values in per_expert.items():
        if values:
            out[f"expert_train_slices_mean_{region}"] = float(np.mean(values))
    return out


def analyse(
    output_dir: str,
    experiment: str,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the whole routing analysis and persist it."""
    report: Dict[str, Any] = {"experiment": experiment}

    gap = routing_gap(output_dir, experiment)
    if not gap.empty:
        summary: Dict[str, Any] = {}
        for column in gap.columns:
            if "__" not in column:
                continue
            values = gap[column].dropna().to_numpy(dtype=np.float64)
            if values.size:
                summary[column] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "n_folds": int(values.size),
                }
        report["routing_modes"] = summary

    report["region_estimator"] = region_estimator_report(output_dir, experiment)
    report["routing_error_breakdown"] = routing_error_breakdown(output_dir, experiment)
    report["expert_utilisation"] = expert_utilisation(output_dir, experiment)

    per_region = per_region_performance(output_dir, experiment)

    target_dir = save_dir or os.path.join(output_dir, "analysis")
    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, f"routing_{experiment}.json"), "w") as fh:
        json.dump(store.jsonable(report), fh, indent=2, sort_keys=True)
    if not gap.empty:
        gap.to_csv(os.path.join(target_dir, f"routing_gap_{experiment}.csv"), index=False)
    if not per_region.empty:
        per_region.to_csv(os.path.join(target_dir, f"per_region_{experiment}.csv"), index=False)

    _log_headline(report)
    return report


def _log_headline(report: Dict[str, Any]) -> None:
    modes = report.get("routing_modes") or {}
    gap_key = "patient_macro_f1__gap_oracle_soft"
    if gap_key in modes:
        entry = modes[gap_key]
        log.info(
            "routing gap (oracle - soft) on patient macro-F1: %+.4f +/- %.4f over %d folds",
            entry["mean"],
            entry["std"],
            entry["n_folds"],
        )
    estimator = report.get("region_estimator") or {}
    if estimator.get("region_accuracy") is not None:
        log.info(
            "region estimator: accuracy=%.4f macro-F1=%.4f",
            estimator.get("region_accuracy", float("nan")),
            estimator.get("region_macro_f1", float("nan")),
        )
    utilisation = report.get("expert_utilisation") or {}
    if utilisation.get("router_collapsed"):
        log.warning(
            "router has collapsed: %.1f%% of slices route to a single expert",
            100.0 * utilisation.get("router_top1_max_share", 0.0),
        )


__all__ = [
    "analyse",
    "expert_utilisation",
    "per_region_performance",
    "region_estimator_report",
    "routing_error_breakdown",
    "routing_gap",
]
