"""Hand-crafted 2D features for the radiomics comparator.

Radiomics + gradient boosting is the dominant methodology in the PPGL imaging
literature (Zhou et al. 2024/2025; Yi et al.), so reviewers in this niche expect
it as a baseline. This module supplies it.

**Important caveat that must appear in the paper.** These are *not* HU-calibrated
radiomic features. The archive stores 8-bit slices already windowed at 50/450 HU,
so every intensity statistic here is computed on quantised, window-clipped
grey levels. First-order intensity features are therefore not comparable to
published HU-based radiomics, and no feature can be interpreted as an
attenuation measurement. Texture and shape descriptors remain meaningful.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ppgl.utils.logging import get_logger

log = get_logger(__name__)

_GLCM_DISTANCES = (1, 2, 4)
_GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def extract_features(image: np.ndarray, boxes: Optional[List[List[float]]] = None) -> Dict[str, float]:
    """Feature vector for one lesion crop.

    Parameters
    ----------
    image
        2-D uint8 crop, already restricted to the lesion region.
    boxes
        Original lesion boxes in slice coordinates, used for shape and
        multifocality descriptors.
    """
    array = image.astype(np.float32)
    features: Dict[str, float] = {}

    # ---- first order (on windowed 8-bit grey levels, NOT HU) ---------------
    flat = array.reshape(-1)
    if flat.size == 0:
        return {"invalid": 1.0}
    features["fo_mean"] = float(flat.mean())
    features["fo_std"] = float(flat.std())
    features["fo_min"] = float(flat.min())
    features["fo_max"] = float(flat.max())
    features["fo_range"] = features["fo_max"] - features["fo_min"]
    for q in (10, 25, 50, 75, 90):
        features[f"fo_p{q}"] = float(np.percentile(flat, q))
    features["fo_iqr"] = features["fo_p75"] - features["fo_p25"]
    features["fo_skew"] = _skew(flat)
    features["fo_kurtosis"] = _kurtosis(flat)
    features["fo_entropy"] = _entropy(flat)
    features["fo_uniformity"] = _uniformity(flat)
    features["fo_mad"] = float(np.mean(np.abs(flat - flat.mean())))
    features["fo_rms"] = float(np.sqrt(np.mean(flat**2)))
    features["fo_cv"] = float(features["fo_std"] / (features["fo_mean"] + 1e-6))

    # ---- gradient / edge ---------------------------------------------------
    if array.ndim == 2 and min(array.shape) >= 3:
        gy, gx = np.gradient(array)
        magnitude = np.sqrt(gx**2 + gy**2)
        features["grad_mean"] = float(magnitude.mean())
        features["grad_std"] = float(magnitude.std())
        features["grad_p90"] = float(np.percentile(magnitude, 90))
    else:
        features["grad_mean"] = features["grad_std"] = features["grad_p90"] = 0.0

    # ---- texture ----------------------------------------------------------
    features.update(_glcm_features(image))
    features.update(_lbp_histogram(image))

    # ---- shape / multifocality -------------------------------------------
    features.update(_shape_features(boxes or []))
    return features


def _skew(values: np.ndarray) -> float:
    centered = values - values.mean()
    denom = (values.std() + 1e-6) ** 3
    return float(np.mean(centered**3) / denom)


def _kurtosis(values: np.ndarray) -> float:
    centered = values - values.mean()
    denom = (values.std() + 1e-6) ** 4
    return float(np.mean(centered**4) / denom - 3.0)


def _histogram(values: np.ndarray, bins: int = 32) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 255.0))
    total = counts.sum()
    return counts / total if total else counts.astype(np.float64)


def _entropy(values: np.ndarray) -> float:
    probs = _histogram(values)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero))) if nonzero.size else 0.0


def _uniformity(values: np.ndarray) -> float:
    probs = _histogram(values)
    return float(np.sum(probs**2))


def _glcm_features(image: np.ndarray) -> Dict[str, float]:
    """Grey-level co-occurrence descriptors, averaged over angles."""
    out: Dict[str, float] = {}
    try:
        from skimage.feature import graycomatrix, graycoprops
    except Exception:
        # scikit-image is optional; emit zeros so the feature vector stays fixed.
        for distance in _GLCM_DISTANCES:
            for prop in _GLCM_PROPS:
                out[f"glcm_d{distance}_{prop}"] = 0.0
        return out

    quantised = (np.asarray(image, dtype=np.uint8) // 4).astype(np.uint8)  # 64 levels
    for distance in _GLCM_DISTANCES:
        if min(quantised.shape) <= distance + 1:
            for prop in _GLCM_PROPS:
                out[f"glcm_d{distance}_{prop}"] = 0.0
            continue
        try:
            matrix = graycomatrix(
                quantised,
                distances=[distance],
                angles=list(_GLCM_ANGLES),
                levels=64,
                symmetric=True,
                normed=True,
            )
            for prop in _GLCM_PROPS:
                out[f"glcm_d{distance}_{prop}"] = float(np.mean(graycoprops(matrix, prop)))
        except Exception:
            for prop in _GLCM_PROPS:
                out[f"glcm_d{distance}_{prop}"] = 0.0
    return out


def _lbp_histogram(image: np.ndarray, points: int = 8, radius: int = 1) -> Dict[str, float]:
    """Uniform local binary pattern histogram."""
    n_bins = points + 2
    try:
        from skimage.feature import local_binary_pattern

        codes = local_binary_pattern(
            np.asarray(image, dtype=np.uint8), P=points, R=radius, method="uniform"
        )
        counts, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
        total = counts.sum()
        normalised = counts / total if total else counts.astype(np.float64)
    except Exception:
        normalised = np.zeros(n_bins, dtype=np.float64)
    return {f"lbp_{i}": float(v) for i, v in enumerate(normalised)}


def _shape_features(boxes: List[List[float]]) -> Dict[str, float]:
    if not boxes:
        return {
            "shape_n_lesions": 0.0,
            "shape_area": 0.0,
            "shape_width": 0.0,
            "shape_height": 0.0,
            "shape_aspect": 0.0,
            "shape_max_area": 0.0,
            "shape_total_area": 0.0,
            "shape_spread": 0.0,
        }
    widths = [float(b[2] - b[0]) for b in boxes]
    heights = [float(b[3] - b[1]) for b in boxes]
    areas = [w * h for w, h in zip(widths, heights)]
    centres = np.array(
        [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes], dtype=np.float32
    )
    spread = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max()) if len(centres) > 1 else 0.0

    largest = int(np.argmax(areas))
    return {
        "shape_n_lesions": float(len(boxes)),
        "shape_area": float(areas[largest]),
        "shape_width": widths[largest],
        "shape_height": heights[largest],
        "shape_aspect": float(widths[largest] / (heights[largest] + 1e-6)),
        "shape_max_area": float(max(areas)),
        "shape_total_area": float(sum(areas)),
        "shape_spread": spread,
    }


def build_feature_table(
    frame: pd.DataFrame,
    data_cfg,
    cache_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract features for every slice in ``frame``.

    Returns the feature table (indexed like ``frame``) and the feature column
    names. Extraction is deterministic, so the result is cached.
    """
    import os

    from ppgl.data.dataset import compute_crop_box, read_slice

    if cache_csv and os.path.isfile(cache_csv):
        cached = pd.read_csv(cache_csv)
        if len(cached) == len(frame) and set(cached["slice_uid"]) == set(frame["slice_uid"]):
            ordered = cached.set_index("slice_uid").loc[frame["slice_uid"]].reset_index()
            columns = [c for c in ordered.columns if c != "slice_uid"]
            log.info("reusing cached radiomics features from %s", cache_csv)
            return ordered, columns
        log.warning("radiomics cache at %s does not match the manifest; recomputing", cache_csv)

    records: List[Dict[str, float]] = []
    for row in frame.itertuples(index=False):
        boxes = _row_boxes(row)
        try:
            image = read_slice(row.image_path, as_gray=True)
            box = compute_crop_box(
                boxes,
                image.shape,
                crop_mode="roi" if data_cfg.crop_mode != "full" else "full",
                context_scale=data_cfg.roi_context_scale,
                min_crop_px=data_cfg.min_crop_px,
            )
            crop = image[box[1] : box[3], box[0] : box[2]] if box is not None else image
            features = extract_features(crop, boxes)
        except Exception as exc:
            log.warning("radiomics extraction failed for %s (%s)", row.slice_uid, exc)
            features = {}
        features["slice_uid"] = row.slice_uid
        records.append(features)

    table = pd.DataFrame.from_records(records).fillna(0.0)
    columns = [c for c in table.columns if c != "slice_uid"]
    table = table[["slice_uid"] + sorted(columns)]
    columns = sorted(columns)

    if cache_csv:
        _atomic_write_csv(table, cache_csv)
        log.info("wrote %d x %d radiomics feature table to %s", len(table), len(columns), cache_csv)

    ordered = table.set_index("slice_uid").loc[frame["slice_uid"]].reset_index()
    return ordered, columns


def _atomic_write_csv(table: pd.DataFrame, path: str) -> None:
    """Write via a unique temp file then rename.

    Several fold workers can reach this concurrently on the first run of a grid,
    and a plain ``to_csv`` left a window where another process could read a
    half-written file. Extraction is deterministic, so whichever writer lands
    last is equivalent; ``os.replace`` is atomic on POSIX, so a reader sees either
    the old file or a complete new one.
    """
    import os
    import tempfile

    resolved = os.path.abspath(path)
    directory = os.path.dirname(resolved)
    os.makedirs(directory, exist_ok=True)

    handle, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(resolved) + ".", suffix=".tmp", dir=directory
    )
    os.close(handle)
    try:
        table.to_csv(temp_path, index=False)
        os.replace(temp_path, resolved)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _row_boxes(row) -> List[List[float]]:
    raw = getattr(row, "bboxes_json", None)
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    boxes = getattr(row, "bboxes", None)
    return boxes if isinstance(boxes, list) else []


__all__ = ["build_feature_table", "extract_features"]
