"""Publication figures.

Sized for IEEE two-column: single column is 3.39 in wide, double column 7.16 in.
Everything except :func:`attention_comparison_figure` is generated from the stored
per-fold predictions alone; that one needs two saved checkpoints because it renders
the models' actual attention.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ppgl import REGIONS
from ppgl.labels import active_classes
from ppgl.experiments import results as store
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

SINGLE_COLUMN_IN = 3.39
DOUBLE_COLUMN_IN = 7.16
DPI = 400


def _setup():
    import matplotlib

    matplotlib.use("Agg")  # headless server
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
        }
    )
    return plt


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    for alternative in (".pdf", ".png"):
        base, extension = os.path.splitext(path)
        if extension != alternative:
            fig.savefig(base + alternative)
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("wrote figure %s", path)
    return path


def confusion_matrix_figure(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
    patient_level: bool = True,
    normalise: bool = True,
    path: Optional[str] = None,
) -> Optional[str]:
    """Row-normalised confusion matrix, pooled across folds."""
    from ppgl.metrics import aggregate_to_patient
    from sklearn.metrics import confusion_matrix

    plt = _setup()
    predictions = store.load_predictions(output_dir, experiment, mode=mode)
    if predictions.empty:
        log.warning("no predictions for %r", experiment)
        return None

    probability_columns = [f"prob_{cluster}" for cluster in active_classes()]
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_prob = predictions[probability_columns].to_numpy(dtype=np.float64)

    if patient_level and "patient_uid" in predictions.columns:
        # Pool per (repeat, patient) so a patient appearing in several repeats
        # contributes once per repeat rather than once overall.
        keys = (
            predictions["repeat"].astype(str) + ":" + predictions["patient_uid"].astype(str)
            if "repeat" in predictions.columns
            else predictions["patient_uid"]
        )
        y_true, y_prob, _ = aggregate_to_patient(y_true, y_prob, keys.tolist())

    matrix = confusion_matrix(y_true, y_prob.argmax(axis=1), labels=list(range(len(active_classes()))))
    display = matrix.astype(np.float64)
    if normalise:
        totals = display.sum(axis=1, keepdims=True)
        display = np.divide(display, np.where(totals > 0, totals, 1.0))

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, SINGLE_COLUMN_IN * 0.95))
    ax.grid(False)
    image = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0 if normalise else display.max())

    ax.set_xticks(range(len(active_classes())))
    ax.set_yticks(range(len(active_classes())))
    ax.set_xticklabels(active_classes(), rotation=35, ha="right")
    ax.set_yticklabels(active_classes())
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = (display.max() + display.min()) / 2.0
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            text = f"{display[i, j]:.2f}" if normalise else f"{int(matrix[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if display[i, j] > threshold else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    target = path or os.path.join(output_dir, "figures", f"confusion_{experiment}_{mode}.pdf")
    return _save(fig, target)


def roc_figure(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
    path: Optional[str] = None,
) -> Optional[str]:
    """One-vs-rest ROC curves pooled across folds."""
    from sklearn.metrics import roc_auc_score, roc_curve

    plt = _setup()
    predictions = store.load_predictions(output_dir, experiment, mode=mode)
    if predictions.empty:
        return None

    y_true = predictions["y_true"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, SINGLE_COLUMN_IN * 0.9))

    for idx, cluster in enumerate(active_classes()):
        column = f"prob_{cluster}"
        if column not in predictions.columns:
            continue
        binary = (y_true == idx).astype(int)
        if binary.min() == binary.max():
            continue
        scores = predictions[column].to_numpy(dtype=np.float64)
        fpr, tpr, _ = roc_curve(binary, scores)
        auc = roc_auc_score(binary, scores)
        ax.plot(fpr, tpr, linewidth=1.0, label=f"{cluster} ({auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)

    target = path or os.path.join(output_dir, "figures", f"roc_{experiment}_{mode}.pdf")
    return _save(fig, target)


def routing_gap_figure(
    output_dir: str,
    experiment: str,
    metric: str = "patient_macro_f1",
    path: Optional[str] = None,
) -> Optional[str]:
    """Per-fold paired plot of soft vs oracle routing, plus the gap distribution."""
    from ppgl.experiments.routing_analysis import routing_gap

    plt = _setup()
    gap = routing_gap(output_dir, experiment)
    soft_column, oracle_column = f"{metric}__soft", f"{metric}__oracle"
    if gap.empty or soft_column not in gap.columns or oracle_column not in gap.columns:
        log.warning("no routing-mode results for %r", experiment)
        return None

    soft = gap[soft_column].to_numpy(dtype=np.float64)
    oracle = gap[oracle_column].to_numpy(dtype=np.float64)
    differences = oracle - soft

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN * 0.72, SINGLE_COLUMN_IN * 0.85))

    ax = axes[0]
    for index in range(len(soft)):
        ax.plot([0, 1], [soft[index], oracle[index]], color="0.7", linewidth=0.5, zorder=1)
    ax.scatter(np.zeros_like(soft), soft, s=8, label="predicted (soft)", zorder=2)
    ax.scatter(np.ones_like(oracle), oracle, s=8, label="oracle region", zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Soft", "Oracle"])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlim(-0.3, 1.3)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    ax.hist(differences, bins=min(12, max(4, len(differences) // 2)), color="0.6", edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.axvline(
        float(np.mean(differences)),
        color="crimson",
        linewidth=0.9,
        label=f"mean {np.mean(differences):+.3f}",
    )
    ax.set_xlabel("Oracle $-$ soft")
    ax.set_ylabel("Folds")
    ax.legend(frameon=False)

    target = path or os.path.join(output_dir, "figures", f"routing_gap_{experiment}.pdf")
    return _save(fig, target)


def router_confusion_figure(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
    path: Optional[str] = None,
) -> Optional[str]:
    """Confusion matrix of the anatomical-region estimator."""
    from sklearn.metrics import confusion_matrix

    plt = _setup()
    predictions = store.load_predictions(output_dir, experiment, mode=mode)
    router_columns = [f"router_{region}" for region in REGIONS]
    if predictions.empty or "region_true" not in predictions.columns:
        return None
    if not all(column in predictions.columns for column in router_columns):
        return None

    valid = predictions["region_true"] >= 0
    if not valid.any():
        return None

    y_true = predictions.loc[valid, "region_true"].to_numpy(dtype=int)
    y_pred = predictions.loc[valid, router_columns].to_numpy(dtype=np.float64).argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(REGIONS)))).astype(np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, np.where(totals > 0, totals, 1.0))

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN * 0.9, SINGLE_COLUMN_IN * 0.85))
    ax.grid(False)
    image = ax.imshow(normalised, cmap="Purples", vmin=0.0, vmax=1.0)
    labels = [region.replace("_", " ") for region in REGIONS]
    ax.set_xticks(range(len(REGIONS)))
    ax.set_yticks(range(len(REGIONS)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Routed to")
    ax.set_ylabel("True region")
    for i in range(len(REGIONS)):
        for j in range(len(REGIONS)):
            ax.text(
                j,
                i,
                f"{normalised[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if normalised[i, j] > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    target = path or os.path.join(output_dir, "figures", f"router_confusion_{experiment}.pdf")
    return _save(fig, target)


def model_comparison_figure(
    output_dir: str,
    experiments: Sequence[str],
    metric: str = "patient_macro_f1",
    path: Optional[str] = None,
) -> Optional[str]:
    """Horizontal bars with per-fold scatter, so spread is visible not just the mean."""
    from ppgl.experiments.tables import display_name

    plt = _setup()
    entries: List[Dict[str, Any]] = []
    for experiment in experiments:
        table = store.load_fold_metrics(output_dir, experiment, mode="soft")
        if table.empty or metric not in table.columns:
            continue
        values = pd.to_numeric(table[metric], errors="coerce").dropna().to_numpy()
        if values.size:
            entries.append(
                {"name": display_name(experiment), "values": values, "mean": float(values.mean())}
            )
    if not entries:
        return None

    entries.sort(key=lambda item: item["mean"])
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_IN, max(1.4, 0.22 * len(entries) + 0.55))
    )
    positions = np.arange(len(entries))
    ax.barh(
        positions,
        [entry["mean"] for entry in entries],
        color="0.78",
        edgecolor="black",
        linewidth=0.4,
        height=0.62,
    )
    for position, entry in zip(positions, entries):
        jitter = (np.random.default_rng(0).random(entry["values"].size) - 0.5) * 0.28
        ax.scatter(entry["values"], position + jitter, s=3, color="crimson", alpha=0.6, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([entry["name"] for entry in entries])
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_xlim(0, 1)

    target = path or os.path.join(output_dir, "figures", f"comparison_{metric}.pdf")
    return _save(fig, target)


def attention_maps_figure(
    output_dir: str,
    experiment: str,
    checkpoint: Optional[str] = None,
    n_examples: int = 4,
    path: Optional[str] = None,
) -> Optional[str]:
    """Grad-CAM overlays for a few example slices.

    Requires a saved checkpoint. Training does not persist weights by default
    (400 folds x ~110 MB is not worth the disk), so pass ``checkpoint`` from a
    dedicated single-fold run. Returns ``None`` with a warning when unavailable,
    so the figure pipeline does not fail the whole grid.
    """
    if not checkpoint or not os.path.isfile(checkpoint):
        log.warning(
            "attention_maps_figure needs a checkpoint; run a single fold with "
            "--save-checkpoint and pass it here"
        )
        return None

    import torch

    from ppgl.config import _from_dict
    from ppgl.data.dataset import PPGLSliceDataset
    from ppgl.data.inventory import load_manifest
    from ppgl.data.location import attach_regions
    from ppgl.models import build_model

    plt = _setup()

    payload = store.load_config_dict(output_dir, experiment)
    if not payload:
        log.warning("no stored config for %r", experiment)
        return None
    cfg = _from_dict(payload)

    frame = load_manifest(cfg.data.manifest)
    if cfg.data.location_source != "none":
        frame = attach_regions(frame, cfg.data)

    model = build_model(cfg.model)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    dataset = PPGLSliceDataset(
        frame.head(n_examples).reset_index(drop=True),
        cfg.data,
        train=False,
        mean=getattr(model, "mean", None),
        std=getattr(model, "std", None),
    )

    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}
    target_layer = _last_conv_or_norm(model.backbone)
    if target_layer is None:
        log.warning("could not locate a target layer for Grad-CAM in %s", cfg.model.backbone)
        return None

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handles = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook),
    ]

    fig, axes = plt.subplots(
        2, n_examples, figsize=(DOUBLE_COLUMN_IN, SINGLE_COLUMN_IN * 1.15), squeeze=False
    )
    try:
        for column in range(min(n_examples, len(dataset))):
            item = dataset[column]
            image = item["image"].unsqueeze(0)
            image.requires_grad_(True)

            outputs = model(image, region=item["region"].unsqueeze(0))
            logits = outputs["logits"]
            predicted = int(logits.argmax(dim=1).item())
            model.zero_grad(set_to_none=True)
            logits[0, predicted].backward()

            heatmap = _grad_cam(activations.get("value"), gradients.get("value"))
            display = _denormalise(item["image"])

            axes[0][column].imshow(display, cmap="gray")
            axes[0][column].set_title(
                f"{active_classes()[int(item['cluster'])]}\npred {active_classes()[predicted]}", fontsize=6
            )
            axes[1][column].imshow(display, cmap="gray")
            if heatmap is not None:
                axes[1][column].imshow(
                    _resize(heatmap, display.shape), cmap="jet", alpha=0.42
                )
            for row in (0, 1):
                axes[row][column].set_xticks([])
                axes[row][column].set_yticks([])
                axes[row][column].grid(False)
    finally:
        for handle in handles:
            handle.remove()

    target = path or os.path.join(output_dir, "figures", f"gradcam_{experiment}.pdf")
    return _save(fig, target)


def _last_conv_or_norm(module):
    import torch.nn as nn

    candidate = None
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d)):
            candidate = child
    return candidate


def _grad_cam(activation, gradient):
    if activation is None or gradient is None:
        return None
    import torch

    tensor, grad = activation, gradient
    if tensor.dim() == 4:  # (B, C, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * tensor).sum(dim=1).squeeze(0)
    elif tensor.dim() == 3:  # (B, N, C) transformer tokens
        weights = grad.mean(dim=1, keepdim=True)
        cam = (weights * tensor).sum(dim=-1).squeeze(0)
        length = cam.shape[0]
        side = int(round(length**0.5))
        if side * side != length:
            cam = cam[length - side * side :]
        cam = cam.reshape(side, side)
    else:
        return None
    cam = torch.relu(cam)
    cam = cam - cam.min()
    denominator = cam.max()
    if float(denominator) > 0:
        cam = cam / denominator
    return cam.cpu().numpy()


def _denormalise(tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = array.mean(axis=0) if array.ndim == 3 else array
    array = array - array.min()
    maximum = array.max()
    return array / maximum if maximum > 0 else array


def _resize(array: np.ndarray, shape) -> np.ndarray:
    rows = np.linspace(0, array.shape[0] - 1, shape[0]).astype(int)
    columns = np.linspace(0, array.shape[1] - 1, shape[1]).astype(int)
    return array[np.ix_(rows, columns)]


def build_all(
    output_dir: str,
    experiments: Sequence[str],
    reference: str = "routed",
    checkpoint: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> List[str]:
    """Generate every figure that the available results support."""
    target = save_dir or os.path.join(output_dir, "figures")
    os.makedirs(target, exist_ok=True)
    written: List[str] = []

    for produce in (
        lambda: model_comparison_figure(output_dir, experiments),
        lambda: confusion_matrix_figure(output_dir, reference),
        lambda: roc_figure(output_dir, reference),
        lambda: routing_gap_figure(output_dir, reference),
        lambda: router_confusion_figure(output_dir, reference),
        lambda: attention_maps_figure(output_dir, reference, checkpoint=checkpoint),
    ):
        try:
            path = produce()
            if path:
                written.append(path)
        except Exception as exc:  # noqa: BLE001 - a missing figure must not fail the pipeline
            log.warning("figure generation failed: %s: %s", type(exc).__name__, exc)

    log.info("wrote %d figures to %s", len(written), target)
    return written


__all__ = [
    "attention_comparison_figure",
    "attention_maps_figure",
    "build_all",
    "confusion_matrix_figure",
    "model_comparison_figure",
    "roc_figure",
    "router_confusion_figure",
    "routing_gap_figure",
]


# --------------------------------------------------------------------------- #
# Control-vs-proposed attention comparison
# --------------------------------------------------------------------------- #
# Rare clusters, in the order the comparison figure shows them. The gains from the
# attention term are concentrated here, so these are the classes worth illustrating;
# the majority class gains ~0.01 F1 and would show nothing.
_RARE_CLUSTERS = ("VHL_EPAS1", "Kinase", "Sporadic")


def _patch_mass(attention_map: "np.ndarray", mask: "np.ndarray") -> float:
    """Fraction of attention mass inside the lesion, on the patch grid.

    Reproduces :class:`ppgl.losses.AttentionLocalisationLoss` exactly - average-pool
    the mask to the attention grid, then take the inner product with the attention
    distribution. Computing it any other way here would print a number the training
    objective never saw.
    """
    import torch
    import torch.nn.functional as F

    side = attention_map.shape[-1]
    pooled = F.adaptive_avg_pool2d(
        torch.as_tensor(mask, dtype=torch.float32)[None, None], (side, side)
    )
    flat_mask = pooled.flatten().numpy()
    flat_attention = np.asarray(attention_map, dtype=np.float64).ravel()
    total = flat_attention.sum()
    if total <= 0:
        return float("nan")
    return float((flat_attention / total * flat_mask).sum())


def _load_model_for_figure(output_dir: str, experiment: str, checkpoint: str):
    """Rebuild a model from its stored config and load a saved fold checkpoint."""
    import torch

    from ppgl.config import _from_dict, _validate
    from ppgl.models import build_model

    payload = store.load_config_dict(output_dir, experiment)
    if not payload:
        raise FileNotFoundError(f"no stored config for {experiment!r} in {output_dir!r}")
    cfg = _from_dict(payload)
    _validate(cfg)
    model = build_model(cfg.model)
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing or unexpected:
        # Loud rather than silent: a figure drawn from a partially loaded model is
        # worse than no figure, because it looks plausible.
        log.warning(
            "%s checkpoint loaded with %d missing and %d unexpected keys",
            experiment,
            len(missing),
            len(unexpected),
        )
    model.eval()
    return cfg, model


def attention_comparison_figure(
    output_dir: str,
    proposed: str = "amg",
    control: str = "amg_control",
    proposed_checkpoint: Optional[str] = None,
    control_checkpoint: Optional[str] = None,
    repeat: int = 0,
    fold: int = 0,
    path: Optional[str] = None,
) -> Optional[str]:
    """Class-token attention of the control and the proposed model, side by side.

    Three rows, one per rare cluster; three columns: the crop with the annotated
    lesion box, the control's attention, the proposed model's attention. The measured
    attention mass inside the box is printed on each overlay.

    Two things make this figure honest rather than decorative.

    *The quantity shown is the quantity supervised.* It is the class token's attention
    over patches in the final block - read from the model's own ``attention_map``
    output, the same tensor the loss consumes - not Grad-CAM or a saliency proxy. The
    control is the same architecture with ``lambda_attention = 0``, so the two panels
    differ by one term.

    *The examples are chosen by a stated rule, not by eye.* Within each rare cluster,
    among the held-out slices of one fold, we take the slice whose mass **gain** is the
    median for that cluster. Medians, not maxima: the figure then shows a typical case
    per class rather than the three most flattering slices in the cohort. State the
    rule in the caption.

    Both models must come from the *same* fold, and the slices shown must be that
    fold's test slices - otherwise the figure displays attention on training data.
    """
    if not proposed_checkpoint or not os.path.isfile(str(proposed_checkpoint)):
        log.warning(
            "attention_comparison_figure needs a checkpoint for %r; run that fold with "
            "runtime.save_checkpoint=true",
            proposed,
        )
        return None
    if not control_checkpoint or not os.path.isfile(str(control_checkpoint)):
        log.warning(
            "attention_comparison_figure needs a checkpoint for %r; run that fold with "
            "runtime.save_checkpoint=true",
            control,
        )
        return None

    import torch

    from ppgl.data.dataset import PPGLSliceDataset
    from ppgl.experiments.data_prep import build_folds, prepare_frame

    plt = _setup()

    cfg_p, model_p = _load_model_for_figure(output_dir, proposed, proposed_checkpoint)
    cfg_c, model_c = _load_model_for_figure(output_dir, control, control_checkpoint)

    # The comparison is only paired if both arms saw the same split. The stored
    # checkpoints carry the fold fingerprint; refuse rather than draw a figure whose
    # two columns come from different partitions.
    fp_p = torch.load(proposed_checkpoint, map_location="cpu").get("fold_fingerprint")
    fp_c = torch.load(control_checkpoint, map_location="cpu").get("fold_fingerprint")
    if fp_p and fp_c and fp_p != fp_c:
        log.error(
            "checkpoints are from different folds (%s vs %s); the comparison would not "
            "be paired",
            fp_p,
            fp_c,
        )
        return None

    frame = prepare_frame(cfg_p)
    folds = build_folds(cfg_p, frame)
    selected = [f for f in folds if f.repeat == repeat and f.fold == fold]
    if not selected:
        log.error("fold r%sf%s not found", repeat, fold)
        return None
    test_frame = frame.iloc[selected[0].test_idx].reset_index(drop=True)

    dataset = PPGLSliceDataset(
        test_frame,
        cfg_p.data,
        train=False,
        mean=getattr(model_p, "mean", None),
        std=getattr(model_p, "std", None),
        return_mask=True,
    )

    # Score every held-out slice under both models, then select. Selecting before
    # scoring would make the rule unstatable.
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            mask = item.get("lesion_mask")
            if mask is None:
                continue
            mask_np = mask.detach().cpu().numpy()
            mask_np = mask_np[0] if mask_np.ndim == 3 else mask_np
            if mask_np.sum() <= 0:
                continue
            image = item["image"].unsqueeze(0)
            out_p = model_p(image)
            out_c = model_c(image)
            if "attention_map" not in out_p or "attention_map" not in out_c:
                log.error("a model did not return attention_map; cannot draw the figure")
                return None
            map_p = out_p["attention_map"][0].detach().cpu().numpy()
            map_c = out_c["attention_map"][0].detach().cpu().numpy()
            rows.append(
                {
                    "index": index,
                    "cluster": str(test_frame.loc[index, "cluster"]),
                    "mass_proposed": _patch_mass(map_p, mask_np),
                    "mass_control": _patch_mass(map_c, mask_np),
                    "map_proposed": map_p,
                    "map_control": map_c,
                    "mask": mask_np,
                    "image": _denormalise(item["image"]),
                }
            )
    if not rows:
        log.warning("no held-out slice in r%sf%s has a lesion mask", repeat, fold)
        return None

    # Keyed by dataset index, NOT by position in ``rows``: slices without a lesion mask
    # are skipped above, so the two are not the same and indexing the list by dataset
    # index would silently select the wrong slice.
    by_index = {int(r["index"]): r for r in rows}
    scored = pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows]
    )
    scored["gain"] = scored["mass_proposed"] - scored["mass_control"]

    chosen: List[Dict[str, Any]] = []
    for cluster in _RARE_CLUSTERS:
        subset = scored[scored["cluster"] == cluster].sort_values("gain")
        if subset.empty:
            log.warning("cluster %s has no held-out slice in r%sf%s", cluster, repeat, fold)
            continue
        median_row = subset.iloc[len(subset) // 2]
        record = dict(by_index[int(median_row["index"])])
        record["gain"] = float(median_row["gain"])
        chosen.append(record)
    if not chosen:
        log.warning("no rare-cluster examples available in r%sf%s", repeat, fold)
        return None

    n_rows = len(chosen)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(SINGLE_COLUMN_IN * 1.55, SINGLE_COLUMN_IN * 0.56 * n_rows),
        squeeze=False,
    )
    titles = ("slice + annotation", "control ($\\lambda$=0)", "proposed ($\\lambda$=3)")
    for row_index, record in enumerate(chosen):
        display = record["image"]
        mask = record["mask"]
        for column in range(3):
            ax = axes[row_index][column]
            ax.imshow(display, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if column == 0:
                ax.contour(mask, levels=[0.5], colors="#e8000b", linewidths=0.9)
            else:
                key = "map_control" if column == 1 else "map_proposed"
                mass_key = "mass_control" if column == 1 else "mass_proposed"
                heat = _resize(record[key], display.shape)
                ax.imshow(heat, cmap="inferno", alpha=0.55)
                ax.contour(mask, levels=[0.5], colors="#39ff14", linewidths=0.8)
                ax.text(
                    0.03,
                    0.03,
                    f"mass {record[mass_key]:.2f}",
                    transform=ax.transAxes,
                    fontsize=6,
                    color="white",
                    va="bottom",
                    ha="left",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.0, "edgecolor": "none"},
                )
            if row_index == 0:
                ax.set_title(titles[column], fontsize=7)
        axes[row_index][0].set_ylabel(
            str(record["cluster"]).replace("_", "/"), fontsize=6.5
        )

    fig.tight_layout(pad=0.25)
    return _save(fig, path or os.path.join(output_dir, "figures", "attention_comparison.png"))
