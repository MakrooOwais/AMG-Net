"""Anatomical region labels (head & neck / chest / abdomen-pelvis).

**These labels do not exist in the archive.** The XMLs carry only a class name
and bounding boxes. An earlier draft states the region was "set by an
experienced radiologist", but those labels were not distributed with the data,
so this module provides two paths:

``location_source="csv"``
    Authoritative labels from a radiologist-provided CSV. **Use this for any
    number that goes in the paper.** Required columns: an identifier column
    (``slice_uid``, or ``patient_uid`` to label a whole patient) and ``region``
    with values in :data:`ppgl.REGIONS`.

``location_source="heuristic"``
    A provisional estimator so the pipeline is runnable end-to-end before the
    labels arrive. It thresholds the fraction of air-like pixels inside the body
    mask (lung parenchyma appears dark even at a 50/450 window) and the body
    cross-sectional area, then emits a per-slice label with a confidence score.
    It is **research scaffolding, not ground truth** - every row it produces is
    flagged ``provisional=True`` and it refuses to be used silently.

Run ``python -m ppgl.cli locations --review-csv <path>`` to dump the heuristic
labels sorted by ascending confidence for radiologist correction.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ppgl import REGION_TO_IDX, REGIONS
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_REGION_COLUMN = "region"
_ID_COLUMNS = ("slice_uid", "patient_uid")

# Heuristic thresholds for the 50 HU level / 450 HU width window of this archive.
#
# That window maps HU linearly from -175..275 onto 0..1, so:
#   air and aerated lung  (<= -175 HU)  -> 0.00  (clipped)
#   fat                   (~ -100 HU)   -> 0.17
#   water                 (    0 HU)    -> 0.39
#   soft tissue           (~   50 HU)   -> 0.50
#   cortical bone         (>=  275 HU)  -> 1.00
#
# So "air-like" must mean *very* dark. An earlier threshold of 0.22 counted fat
# as lung and mislabelled most abdominal slices as chest.
_AIR_LEVEL = 0.05               # below this is air or aerated lung
_BODY_LEVEL = 0.10              # above this is body rather than scanner background
_BONE_LEVEL = 0.90              # cortical bone saturates at the top of the window
_AIR_FRACTION_CHEST = 0.10      # aerated lung occupies a large share of the thorax
# Head and neck sections carry a thick cortical ring (skull, mandible, cervical
# spine) that the abdomen does not. Body cross-sectional *area* does not separate
# them, because head and neck acquisitions use a smaller field of view and so the
# anatomy fills a similar share of the frame.
#
# This threshold was calibrated so that the *marginal* region distribution over
# this archive approximates the one reported in the prior work (roughly 506
# abdomen / 80 chest / 64 head & neck). Matching a marginal is not the same as
# being correct per slice, and per-slice accuracy here is unknown and probably
# poor. It is a starting point for radiologist review, nothing more.
_BONE_FRACTION_HEAD_NECK = 0.098
_BODY_FRACTION_ABDOMEN = 0.30


# --------------------------------------------------------------------------- #
# Authoritative CSV
# --------------------------------------------------------------------------- #
def load_location_csv(path: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """Load radiologist region labels and align them to the manifest.

    Returns a frame with ``slice_uid``, ``region``, ``region_idx``,
    ``region_confidence`` and ``provisional`` columns.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"location CSV not found: {path!r}")

    table = pd.read_csv(path)
    if REQUIRED_REGION_COLUMN not in table.columns:
        raise ValueError(f"{path!r} must contain a {REQUIRED_REGION_COLUMN!r} column")

    id_column = next((c for c in _ID_COLUMNS if c in table.columns), None)
    if id_column is None:
        raise ValueError(f"{path!r} must contain one of {_ID_COLUMNS}")

    table = table.copy()
    table[REQUIRED_REGION_COLUMN] = (
        table[REQUIRED_REGION_COLUMN].astype(str).str.strip().str.lower().map(_normalise_region)
    )
    bad = table[table[REQUIRED_REGION_COLUMN].isna()]
    if len(bad):
        raise ValueError(
            f"{path!r} has {len(bad)} unrecognised region values; allowed: {list(REGIONS)}"
        )

    if id_column == "patient_uid":
        merged = manifest[["slice_uid", "patient_uid"]].merge(
            table[["patient_uid", REQUIRED_REGION_COLUMN]], on="patient_uid", how="left"
        )
    else:
        merged = manifest[["slice_uid"]].merge(
            table[["slice_uid", REQUIRED_REGION_COLUMN]], on="slice_uid", how="left"
        )

    missing = int(merged[REQUIRED_REGION_COLUMN].isna().sum())
    if missing:
        raise ValueError(
            f"{path!r} covers only {len(merged) - missing}/{len(merged)} slices. "
            "Region labels must be complete; fill the gaps or use "
            "data.location_source=heuristic for a provisional run."
        )

    merged["region_idx"] = merged[REQUIRED_REGION_COLUMN].map(REGION_TO_IDX).astype(int)
    merged["region_confidence"] = 1.0
    merged["provisional"] = False
    log.info("loaded authoritative region labels for %d slices from %s", len(merged), path)
    return merged[["slice_uid", "region", "region_idx", "region_confidence", "provisional"]]


def _normalise_region(value: str) -> Optional[str]:
    """Map common spellings onto the canonical region vocabulary."""
    if value is None:
        return None
    key = str(value).strip().lower().replace("&", "and")
    key = key.replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in key:
        key = key.replace("__", "_")
    aliases = {
        "head_neck": "head_neck",
        "head_and_neck": "head_neck",
        "headneck": "head_neck",
        "hn": "head_neck",
        "h_n": "head_neck",
        "neck": "head_neck",
        "chest": "chest",
        "thorax": "chest",
        "thoracic": "chest",
        "mediastinum": "chest",
        "abdomen_pelvis": "abdomen_pelvis",
        "abdomen": "abdomen_pelvis",
        "abdomen_and_pelvis": "abdomen_pelvis",
        "abdomen_to_pelvis": "abdomen_pelvis",
        "abdominal": "abdomen_pelvis",
        "pelvis": "abdomen_pelvis",
        "retroperitoneum": "abdomen_pelvis",
        "adrenal": "abdomen_pelvis",
    }
    return aliases.get(key)


# --------------------------------------------------------------------------- #
# Provisional heuristic
# --------------------------------------------------------------------------- #
def estimate_region_from_image(image: np.ndarray) -> Dict[str, object]:
    """Estimate the region for one axial slice.

    Parameters
    ----------
    image
        2-D uint8 grayscale array of the *full* slice (not a tumour crop).

    Returns
    -------
    dict
        ``region``, ``confidence`` in [0, 1], plus the raw features so the
        thresholds can be audited and retuned.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale slice, got shape {image.shape}")

    array = image.astype(np.float32)
    if array.max() > 1.0:
        array = array / 255.0

    # The body outline is everything brighter than background, but aerated lung
    # is *darker* than background-level and so appears as a hole in that mask.
    # The holes must be filled before looking for air inside the body, otherwise
    # intra-thoracic air is excluded by construction and no slice ever looks like
    # chest.
    body = _fill_holes(_largest_component(array > _BODY_LEVEL))
    body_fraction = float(body.mean())
    if body_fraction < 1e-4:
        return {
            "region": "abdomen_pelvis",
            "confidence": 0.0,
            "body_fraction": body_fraction,
            "air_fraction": 0.0,
            "largest_air_fraction": 0.0,
        }

    # Erode so the skin line and the external air boundary cannot contribute.
    interior = _erode(body, iterations=6)
    if interior.sum() < 256:
        interior = body
    air_mask = (array < _AIR_LEVEL) & interior
    air_fraction = float(air_mask.sum()) / float(max(interior.sum(), 1))

    # Aerated lung forms one or two large connected regions; bowel gas is
    # fragmented. The largest component separates the two far better than the
    # total air fraction.
    largest_air = _largest_component_fraction(air_mask, interior.sum())

    bone_fraction = float(((array > _BONE_LEVEL) & body).sum()) / float(max(body.sum(), 1))

    if largest_air >= _AIR_FRACTION_CHEST:
        region = "chest"
        confidence = _ramp(largest_air, _AIR_FRACTION_CHEST, 0.25)
    elif bone_fraction >= _BONE_FRACTION_HEAD_NECK:
        region = "head_neck"
        confidence = _ramp(bone_fraction, _BONE_FRACTION_HEAD_NECK, 0.11)
    else:
        region = "abdomen_pelvis"
        confidence = _ramp(body_fraction, 0.20, _BODY_FRACTION_ABDOMEN)
        # Large bowel gas collections look thoracic, and a dense lumbar spine or
        # pelvic ring can look like skull. Damp confidence near either boundary.
        if largest_air > _AIR_FRACTION_CHEST * 0.6:
            confidence *= 0.5
        if bone_fraction > _BONE_FRACTION_HEAD_NECK * 0.75:
            confidence *= 0.5

    return {
        "region": region,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "body_fraction": round(body_fraction, 4),
        "air_fraction": round(air_fraction, 4),
        "largest_air_fraction": round(float(largest_air), 4),
        "bone_fraction": round(bone_fraction, 4),
    }


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes so intrathoracic air counts as inside the body."""
    if not mask.any():
        return mask
    try:
        from scipy import ndimage

        return ndimage.binary_fill_holes(mask)
    except Exception:
        # Without scipy, approximate by filling each row and column between the
        # first and last set pixel and intersecting the two. Crude but adequate
        # for a convex-ish body cross-section.
        filled_rows = np.zeros_like(mask)
        for index in range(mask.shape[0]):
            hits = np.flatnonzero(mask[index])
            if hits.size:
                filled_rows[index, hits[0] : hits[-1] + 1] = True
        filled_columns = np.zeros_like(mask)
        for index in range(mask.shape[1]):
            hits = np.flatnonzero(mask[:, index])
            if hits.size:
                filled_columns[hits[0] : hits[-1] + 1, index] = True
        return filled_rows & filled_columns


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component, dropping the table and arms."""
    if not mask.any():
        return mask
    try:
        from scipy import ndimage

        labelled, count = ndimage.label(mask)
        if count <= 1:
            return mask
        sizes = ndimage.sum(mask, labelled, index=range(1, count + 1))
        return labelled == (int(np.argmax(sizes)) + 1)
    except Exception:
        return mask


def _largest_component_fraction(mask: np.ndarray, denominator: float) -> float:
    """Area of the largest connected component of ``mask``, as a fraction."""
    if not mask.any():
        return 0.0
    try:
        from scipy import ndimage

        labelled, count = ndimage.label(mask)
        if count == 0:
            return 0.0
        sizes = ndimage.sum(mask, labelled, index=range(1, count + 1))
        return float(np.max(sizes)) / float(max(denominator, 1))
    except Exception:
        # Without scipy, fall back to the total fraction. This over-estimates for
        # fragmented bowel gas, which is why the confidence damping above exists.
        return float(mask.sum()) / float(max(denominator, 1))


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary erosion with a 3x3 cross, via scipy when available."""
    try:
        from scipy import ndimage

        return ndimage.binary_erosion(mask, iterations=iterations)
    except Exception:
        eroded = mask
        for _ in range(iterations):
            shifted = eroded.copy()
            shifted[1:, :] &= eroded[:-1, :]
            shifted[:-1, :] &= eroded[1:, :]
            shifted[:, 1:] &= eroded[:, :-1]
            shifted[:, :-1] &= eroded[:, 1:]
            eroded = shifted
        return eroded


def build_heuristic_locations(
    manifest: pd.DataFrame,
    output_csv: Optional[str] = None,
    review_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Estimate a region for every slice in the manifest.

    Also writes a review file sorted by ascending confidence so a radiologist
    can correct the least certain slices first.
    """
    from ppgl.data.dataset import read_slice  # local import: avoids a cycle

    records: List[Dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        try:
            image = read_slice(row.image_path, as_gray=True)
            estimate = estimate_region_from_image(image)
        except Exception as exc:
            log.warning("region estimation failed for %s (%s)", row.image_path, exc)
            estimate = {
                "region": "abdomen_pelvis",
                "confidence": 0.0,
                "body_fraction": float("nan"),
                "air_fraction": float("nan"),
                "largest_air_fraction": float("nan"),
            }
        records.append(
            {
                "slice_uid": row.slice_uid,
                "patient_uid": row.patient_uid,
                "cluster": row.cluster,
                "region": estimate["region"],
                "region_idx": REGION_TO_IDX[str(estimate["region"])],
                "region_confidence": estimate["confidence"],
                "body_fraction": estimate["body_fraction"],
                "air_fraction": estimate["air_fraction"],
                "largest_air_fraction": estimate.get("largest_air_fraction", float("nan")),
                "provisional": True,
            }
        )

    frame = pd.DataFrame.from_records(records)
    counts = frame["region"].value_counts().to_dict()
    log.warning(
        "PROVISIONAL region labels generated for %d slices: %s. "
        "These are NOT radiologist labels and must not be used for reported results.",
        len(frame),
        counts,
    )

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        frame.to_csv(output_csv, index=False)
        log.info("wrote provisional region labels to %s", output_csv)
    if review_csv:
        os.makedirs(os.path.dirname(os.path.abspath(review_csv)), exist_ok=True)
        ordered = frame.sort_values("region_confidence")
        ordered.insert(len(ordered.columns), "radiologist_region", "")
        ordered.to_csv(review_csv, index=False)
        log.info(
            "wrote %d slices to %s for review, lowest confidence first; "
            "fill the radiologist_region column and pass it via data.location_csv",
            len(ordered),
            review_csv,
        )
    return frame


# --------------------------------------------------------------------------- #
# Entry point used by the dataset
# --------------------------------------------------------------------------- #
def attach_regions(manifest: pd.DataFrame, data_cfg) -> pd.DataFrame:
    """Return the manifest with region columns attached, per ``data_cfg``."""
    source = data_cfg.location_source

    if source == "none":
        out = manifest.copy()
        out["region"] = None
        out["region_idx"] = -1
        out["region_confidence"] = 0.0
        out["provisional"] = True
        return out

    if source == "csv":
        if not data_cfg.location_csv:
            raise ValueError("data.location_source='csv' requires data.location_csv to be set")
        regions = load_location_csv(data_cfg.location_csv, manifest)
    elif source == "heuristic":
        cache = data_cfg.location_cache
        if cache and os.path.isfile(cache):
            regions = pd.read_csv(cache)
            log.info("reusing cached provisional region labels from %s", cache)
            if "region_idx" not in regions.columns:
                regions["region_idx"] = regions["region"].map(REGION_TO_IDX)
        else:
            regions = build_heuristic_locations(manifest, output_csv=cache)
        regions = regions[["slice_uid", "region", "region_idx", "region_confidence", "provisional"]]
    else:  # pragma: no cover - validated in config
        raise ValueError(f"unknown location_source {source!r}")

    merged = manifest.merge(regions, on="slice_uid", how="left", validate="one_to_one")
    if merged["region_idx"].isna().any():
        missing = int(merged["region_idx"].isna().sum())
        raise ValueError(f"{missing} slices have no region label after merge")
    merged["region_idx"] = merged["region_idx"].astype(int)
    return merged


__all__ = [
    "attach_regions",
    "build_heuristic_locations",
    "estimate_region_from_image",
    "load_location_csv",
]
