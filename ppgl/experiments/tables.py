"""Result tables, emitted as both CSV and LaTeX.

Sized for the ISBI four-page budget: a full paper allows four pages *including*
figures, tables and references, so the main results table is single-column and
carries five metric columns at most. Longer variants are written to CSV for the
arXiv companion.

Every table is generated from the stored per-fold results, so no number is typed
by hand and every value traces to a fold.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ppgl import REGIONS
from ppgl.labels import active_classes
from ppgl.experiments import results as store
from ppgl.metrics import PRIMARY_METRIC
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

# Display names for the paper. Keys are experiment names.
DISPLAY_NAMES = {
    "region_majority": "Region prior (floor)",
    "location_only": "Location only",
    "radiomics_gbm": "Radiomics + GBM",
    "radiomics_gbm_region": "Radiomics + GBM + location",
    "resnet101": "ResNet-101",
    "vit": "ViT-B",
    "vit_cl": "ViT-B + SupCon",
    "swin": "Swin-T",
    "ppgl_transformer": "PPGL-Transformer [Santra 2023]",
    "single_head_aux": "Swin-T + aux. region loss",
    "routed": "Proposed (anatomy-routed)",
    "routed_hard": "Proposed, hard routing",
    "routed_soft": "Proposed, soft routing",
    "routed_anneal": "Proposed, annealed routing",
    "routed_consistency": "Proposed, router consistency",
    "swin_dinov2": "DINOv2 ViT-B",
    "swin_radimagenet": "RadImageNet ResNet-50",
    "routed_lora": "Proposed, LoRA",
    "routed_linear": "Proposed, linear probe",
}

MAIN_METRICS = ("patient_macro_f1", "patient_mcc", "patient_balanced_accuracy", "patient_macro_auc_ovr")

# Publication spellings for the label vocabulary.
CLUSTER_DISPLAY = {
    "SDHx": "SDHx",
    "VHL_EPAS1": "VHL/EPAS1",
    "Kinase": "Kinase",
    "Sporadic": "Sporadic",
}
REGION_DISPLAY = {
    "head_neck": "Head \\& neck",
    "chest": "Chest",
    "abdomen_pelvis": "Abdomen/pelvis",
}


def display_name(experiment: str) -> str:
    return DISPLAY_NAMES.get(experiment, experiment.replace("_", " "))


def cluster_display(cluster: str) -> str:
    """Publication spelling for a class name in either vocabulary."""
    from ppgl.labels import active_label_space

    if cluster in CLUSTER_DISPLAY:
        return CLUSTER_DISPLAY[cluster]
    return active_label_space().display.get(cluster, cluster.replace("_", " "))


def _latex_escape(text: str) -> str:
    """Escape LaTeX specials in a plain-text cell.

    Cluster names contain underscores (``VHL_EPAS1``), which would otherwise fail
    to compile. Applied before ``+/-`` is turned into ``$\\pm$`` so the escape
    does not mangle the maths.
    """
    for char in ("\\",):
        text = text.replace(char, "\\textbackslash{}")
    for char in ("_", "&", "%", "#", "$", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def _mean_std(values: Sequence[float]) -> str:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return "--"
    if clean.size == 1:
        return f"{clean[0]:.3f}"
    return f"{clean.mean():.3f}$\\pm${clean.std(ddof=1):.3f}"


def _mean_std_plain(values: Sequence[float]) -> str:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return "--"
    if clean.size == 1:
        return f"{clean[0]:.3f}"
    return f"{clean.mean():.3f}+/-{clean.std(ddof=1):.3f}"


def main_results(
    output_dir: str,
    experiments: Sequence[str],
    metrics: Sequence[str] = MAIN_METRICS,
    mode: str = "soft",
) -> pd.DataFrame:
    """One row per model, mean +/- std across folds."""
    rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        table = store.load_fold_metrics(output_dir, experiment, mode=mode)
        if table.empty:
            log.warning("no results for %r; omitted from the table", experiment)
            continue
        row: Dict[str, Any] = {
            "experiment": experiment,
            "model": display_name(experiment),
            "n_folds": int(len(table)),
        }
        for metric in metrics:
            if metric in table.columns:
                values = pd.to_numeric(table[metric], errors="coerce").to_numpy()
                row[metric] = float(np.nanmean(values))
                row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_str"] = _mean_std_plain(values)
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_results(
    output_dir: str,
    experiments: Sequence[str],
    metric: str = "f1",
    mode: str = "soft",
    patient_level: bool = True,
) -> pd.DataFrame:
    """Per-cluster breakdown: where the macro average actually comes from.

    Unlike that table, every column here is the same quantity for every model.
    """
    prefix = "patient_" if patient_level else ""
    rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        table = store.load_fold_metrics(output_dir, experiment, mode=mode)
        if table.empty:
            continue
        row: Dict[str, Any] = {"experiment": experiment, "model": display_name(experiment)}
        for cluster in active_classes():
            column = f"{prefix}{metric}_{cluster}"
            if column in table.columns:
                values = pd.to_numeric(table[column], errors="coerce").to_numpy()
                row[cluster] = float(np.nanmean(values))
                row[f"{cluster}_str"] = _mean_std_plain(values)
        rows.append(row)
    return pd.DataFrame(rows)


def routing_table(output_dir: str, experiment: str) -> pd.DataFrame:
    """Routing modes side by side, with the oracle gap. The paper's key table."""
    from ppgl.experiments.routing_analysis import routing_gap

    gap = routing_gap(output_dir, experiment)
    if gap.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for metric in ("patient_macro_f1", "macro_f1", "patient_balanced_accuracy", "macro_auc_ovr"):
        row: Dict[str, Any] = {"metric": metric}
        for mode in ("soft", "argmax", "oracle"):
            column = f"{metric}__{mode}"
            if column in gap.columns:
                row[mode] = _mean_std_plain(gap[column].to_numpy())
        gap_column = f"{metric}__gap_oracle_soft"
        if gap_column in gap.columns:
            row["oracle_minus_soft"] = _mean_std_plain(gap[gap_column].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_table(manifest_path: str) -> pd.DataFrame:
    """Dataset composition, as analysed.

    Reports what the manifest actually contains rather than the counts quoted in
    earlier write-ups of this cohort, which do not match this archive.
    """
    from ppgl import CLUSTERS
    from ppgl.data.inventory import load_manifest

    frame = load_manifest(manifest_path)
    rows: List[Dict[str, Any]] = []
    # Deliberately the canonical 4-way clusters, not the active label space: this
    # table describes what the archive contains, independent of the target chosen
    # for a given experiment.
    for cluster in CLUSTERS:
        subset = frame[frame["cluster"] == cluster]
        slices_per_patient = subset.groupby("patient_uid").size()
        rows.append(
            {
                "cluster": cluster_display(cluster),
                "patients": int(subset["patient_uid"].nunique()),
                "slices": int(len(subset)),
                "studies": int(subset["study_uid"].nunique()),
                "slices_per_patient_mean": round(float(slices_per_patient.mean()), 2)
                if len(slices_per_patient)
                else 0.0,
                "slices_per_patient_max": int(slices_per_patient.max()) if len(slices_per_patient) else 0,
                "multifocal_slices": int((subset["n_lesions"] > 1).sum()),
            }
        )
    rows.append(
        {
            "cluster": "Total",
            "patients": int(frame["patient_uid"].nunique()),
            "slices": int(len(frame)),
            "studies": int(frame["study_uid"].nunique()),
            "slices_per_patient_mean": round(
                float(frame.groupby("patient_uid").size().mean()), 2
            ),
            "slices_per_patient_max": int(frame.groupby("patient_uid").size().max()),
            "multifocal_slices": int((frame["n_lesions"] > 1).sum()),
        }
    )
    return pd.DataFrame(rows)


def expert_capacity_table(output_dir: str, experiment: str) -> pd.DataFrame:
    """Training slices per anatomical expert, per cluster.

    Stated explicitly because head & neck contributes far too few slices to
    support an independent expert and contains no kinase-signalling cases at all.
    """
    import glob
    import json

    pattern = os.path.join(store.run_dir(output_dir, experiment), "folds", "*", store.METRICS_FILENAME)
    accumulated: Dict[str, Dict[str, List[int]]] = {
        region: {"n_slices": [], **{cluster: [] for cluster in active_classes()}} for region in REGIONS
    }
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except Exception:
            continue
        for region, info in (payload.get("expert_effective_n") or {}).items():
            if region not in accumulated or not isinstance(info, dict):
                continue
            accumulated[region]["n_slices"].append(int(info.get("n_slices", 0)))
            for cluster, count in (info.get("per_cluster") or {}).items():
                if cluster in accumulated[region]:
                    accumulated[region][cluster].append(int(count))

    rows: List[Dict[str, Any]] = []
    for region, buckets in accumulated.items():
        row: Dict[str, Any] = {"region": region}
        row["train_slices_mean"] = (
            round(float(np.mean(buckets["n_slices"])), 1) if buckets["n_slices"] else 0.0
        )
        for cluster in active_classes():
            row[cluster] = round(float(np.mean(buckets[cluster])), 1) if buckets[cluster] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def to_latex(
    frame: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    headers: Optional[Sequence[str]] = None,
    caption: str = "",
    label: str = "",
    bold_best: Optional[Sequence[str]] = None,
    column_format: Optional[str] = None,
) -> str:
    """Render a booktabs LaTeX table sized for a single ISBI column."""
    if frame.empty:
        return "% (no data)\n"

    columns = list(columns) if columns else list(frame.columns)
    headers = list(headers) if headers else [c.replace("_", " ") for c in columns]
    if len(headers) != len(columns):
        raise ValueError("headers and columns must have the same length")

    best_rows: Dict[str, int] = {}
    for column in bold_best or []:
        numeric_column = column[:-4] if column.endswith("_str") else column
        if numeric_column in frame.columns:
            values = pd.to_numeric(frame[numeric_column], errors="coerce")
            if values.notna().any():
                best_rows[column] = int(values.idxmax())

    fmt = column_format or ("l" + "c" * (len(columns) - 1))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{fmt}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for index, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "--")
            if isinstance(value, float):
                text = "--" if not np.isfinite(value) else f"{value:.3f}"
            elif isinstance(value, (int, np.integer)):
                text = str(int(value))
            else:
                text = _latex_escape(str(value))
            text = text.replace("+/-", "$\\pm$")
            if column in best_rows and best_rows[column] == index:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def build_all(
    output_dir: str,
    experiments: Sequence[str],
    reference: str = "routed",
    manifest_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    mode: str = "soft",
) -> Dict[str, str]:
    """Generate every table as CSV plus LaTeX. Returns a map of name to path."""
    target = save_dir or os.path.join(output_dir, "tables")
    os.makedirs(target, exist_ok=True)
    written: Dict[str, str] = {}

    def _write(name: str, frame: pd.DataFrame, latex: Optional[str] = None) -> None:
        if frame is None or frame.empty:
            log.warning("table %r is empty; skipped", name)
            return
        csv_path = os.path.join(target, f"{name}.csv")
        frame.to_csv(csv_path, index=False)
        written[name] = csv_path
        if latex:
            tex_path = os.path.join(target, f"{name}.tex")
            with open(tex_path, "w") as fh:
                fh.write(latex)
            written[f"{name}_tex"] = tex_path

    main = main_results(output_dir, experiments, mode=mode)
    _write(
        "main_results",
        main,
        to_latex(
            main,
            columns=["model", "patient_macro_f1_str", "patient_mcc_str", "patient_macro_auc_ovr_str"],
            headers=["Model", "macro-F1", "MCC", "AUC"],
            caption=(
                "Patient-level performance, mean$\\pm$s.d. over repeated patient-level "
                "cross-validation. All models were trained and evaluated by us on identical "
                "folds, and every metric uses one definition across all rows."
            ),
            label="tab:main",
            bold_best=["patient_macro_f1_str", "patient_mcc_str", "patient_macro_auc_ovr_str"],
        ),
    )

    per_class = per_class_results(output_dir, experiments, metric="f1", mode=mode)
    _write(
        "per_class_f1",
        per_class,
        to_latex(
            per_class,
            columns=["model"] + [f"{cluster}_str" for cluster in active_classes()],
            headers=["Model"] + [cluster_display(cluster) for cluster in active_classes()],
            caption="Patient-level per-cluster F1, mean$\\pm$s.d. over folds.",
            label="tab:perclass",
            bold_best=[f"{cluster}_str" for cluster in active_classes()],
        ),
    )

    routing = routing_table(output_dir, reference)
    _write(
        "routing_modes",
        routing,
        to_latex(
            routing,
            columns=["metric", "soft", "argmax", "oracle", "oracle_minus_soft"],
            headers=["Metric", "Soft", "Top-1", "Oracle", "Oracle $-$ Soft"],
            caption=(
                "The same trained weights scored under three routing rules. "
                "Oracle routing uses the ground-truth anatomical region and bounds "
                "what a perfect region estimator would provide."
            ),
            label="tab:routing",
        ),
    )

    capacity = expert_capacity_table(output_dir, reference)
    _write("expert_capacity", capacity)

    if manifest_path and os.path.isfile(manifest_path):
        dataset = dataset_table(manifest_path)
        _write(
            "dataset",
            dataset,
            to_latex(
                dataset,
                columns=["cluster", "patients", "slices", "slices_per_patient_mean", "multifocal_slices"],
                headers=["Cluster", "Patients", "Slices", "Slices/pt.", "Multifocal"],
                caption="Dataset composition, derived directly from the inventory manifest.",
                label="tab:dataset",
            ),
        )

    log.info("wrote %d table files to %s", len(written), target)
    return written


__all__ = [
    "MAIN_METRICS",
    "CLUSTER_DISPLAY",
    "REGION_DISPLAY",
    "build_all",
    "cluster_display",
    "dataset_table",
    "display_name",
    "expert_capacity_table",
    "main_results",
    "per_class_results",
    "routing_table",
    "to_latex",
]
