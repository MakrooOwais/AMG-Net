"""Build the dataset manifest from ``detection_data_NIH/original``.

The manifest is the single source of truth for every count reported in the
paper. Nothing downstream re-derives labels from directory names.

On-disk layout::

    original/
      Slices/{RAW_CLASS}_{NUMERIC_ID}/{NUMERIC_ID}_{STUDY}_Slice_{N}.tiff
      XMLs/  {RAW_CLASS}_{NUMERIC_ID}/{NUMERIC_ID}_{STUDY}_Slice_{N}.xml

Three properties of this archive drive the implementation:

1. **Numeric IDs are reused across classes.** ``00004`` exists under Kinase,
   SDHB, SDHx *and* Sporadic - four different patients. The patient key must be
   ``{RAW_CLASS}_{NUMERIC_ID}``. Grouping on the numeric ID alone silently
   merges patients across classes and leaks between CV folds.
2. **XML ``<filename>`` and ``<path>`` fields are unreliable** - they disagree
   with the actual file name (series token differs). Only the directory name
   and the real file name are trusted.
3. **``SDHB`` and ``SDHx`` are separate directories but one cluster** (1a).

The sibling ``train/`` and ``test/`` directories are deliberately ignored: they
are flat, they strip the class prefix, and so two different patients sharing a
numeric ID become indistinguishable.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ppgl import CLUSTER_TO_IDX, RAW_CLASS_TO_CLUSTER
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

_PATIENT_DIR_RE = re.compile(r"^(?P<raw_class>.+)_(?P<numeric_id>\d+)$")
_SLICE_RE = re.compile(r"^(?P<study>.+)_Slice_(?P<index>-?\d+)$", re.IGNORECASE)

MANIFEST_COLUMNS = [
    "slice_uid",
    "patient_uid",
    "numeric_id",
    "raw_class",
    "cluster",
    "cluster_idx",
    "study_uid",
    "slice_index",
    "image_path",
    "xml_path",
    "width",
    "height",
    "n_lesions",
    "bbox_union_x0",
    "bbox_union_y0",
    "bbox_union_x1",
    "bbox_union_y1",
    "bboxes_json",
    "xml_class_names",
    "class_mismatch",
]


# --------------------------------------------------------------------------- #
# XML parsing
# --------------------------------------------------------------------------- #
def _parse_voc_xml(path: str) -> Dict[str, Any]:
    """Parse a PASCAL VOC annotation. Returns boxes, size and object names."""
    out: Dict[str, Any] = {"width": None, "height": None, "boxes": [], "names": []}
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        log.warning("unparseable XML %s (%s)", path, exc)
        return out

    size = root.find("size")
    if size is not None:
        for key, tag in (("width", "width"), ("height", "height")):
            node = size.find(tag)
            if node is not None and node.text:
                try:
                    out[key] = int(float(node.text))
                except ValueError:
                    pass

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is not None and name_node.text:
            out["names"].append(name_node.text.strip())
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = []
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            node = box.find(tag)
            if node is None or not node.text:
                coords = []
                break
            try:
                coords.append(int(round(float(node.text))))
            except ValueError:
                coords = []
                break
        if len(coords) == 4:
            x0, y0, x1, y1 = coords
            # Some annotations store the corners in the opposite order.
            out["boxes"].append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
    return out


def _split_slice_name(stem: str) -> Tuple[str, Optional[int]]:
    """``00007_2020..._N16_Slice_285`` -> (``00007_2020..._N16``, 285)."""
    match = _SLICE_RE.match(stem)
    if not match:
        return stem, None
    try:
        return match.group("study"), int(match.group("index"))
    except ValueError:
        return match.group("study"), None


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #
def build_manifest(
    root: str,
    output_csv: Optional[str] = None,
    qa_json: Optional[str] = None,
) -> pd.DataFrame:
    """Scan ``root`` and return the manifest as a DataFrame.

    Parameters
    ----------
    root
        Directory containing ``Slices/`` and ``XMLs/``.
    output_csv
        If given, the manifest is written here.
    qa_json
        If given, a data-quality report is written here.
    """
    slices_dir = os.path.join(root, "Slices")
    xmls_dir = os.path.join(root, "XMLs")
    if not os.path.isdir(slices_dir):
        raise FileNotFoundError(f"no Slices/ directory under {root!r}")
    if not os.path.isdir(xmls_dir):
        raise FileNotFoundError(f"no XMLs/ directory under {root!r}")

    qa: Dict[str, Any] = {
        "root": os.path.abspath(root),
        "empty_patient_dirs": [],
        "unrecognised_patient_dirs": [],
        "unknown_raw_classes": [],
        "missing_xml": [],
        "xml_without_boxes": [],
        "class_mismatches": [],
        "unparsed_slice_names": [],
        "numeric_id_collisions": {},
        "duplicate_slice_uids": [],
    }

    rows: List[Dict[str, Any]] = []
    numeric_to_classes: Dict[str, set] = defaultdict(set)

    patient_dirs = sorted(
        d for d in os.listdir(slices_dir) if os.path.isdir(os.path.join(slices_dir, d))
    )

    for patient_dir in patient_dirs:
        match = _PATIENT_DIR_RE.match(patient_dir)
        if not match:
            qa["unrecognised_patient_dirs"].append(patient_dir)
            log.warning("skipping unrecognised patient directory %r", patient_dir)
            continue

        raw_class = match.group("raw_class")
        numeric_id = match.group("numeric_id")
        numeric_to_classes[numeric_id].add(raw_class)

        cluster = RAW_CLASS_TO_CLUSTER.get(raw_class)
        if cluster is None:
            qa["unknown_raw_classes"].append(raw_class)
            log.error(
                "directory %r has raw class %r which is not in RAW_CLASS_TO_CLUSTER; skipping",
                patient_dir,
                raw_class,
            )
            continue

        patient_slice_dir = os.path.join(slices_dir, patient_dir)
        patient_xml_dir = os.path.join(xmls_dir, patient_dir)

        tiffs = sorted(
            f
            for f in os.listdir(patient_slice_dir)
            if f.lower().endswith((".tif", ".tiff")) and not f.startswith(".")
        )
        if not tiffs:
            qa["empty_patient_dirs"].append(patient_dir)
            continue

        for filename in tiffs:
            stem = os.path.splitext(filename)[0]
            study_uid, slice_index = _split_slice_name(stem)
            if slice_index is None:
                qa["unparsed_slice_names"].append(f"{patient_dir}/{filename}")

            image_path = os.path.join(patient_slice_dir, filename)
            xml_path = os.path.join(patient_xml_dir, stem + ".xml")

            parsed: Dict[str, Any] = {"width": None, "height": None, "boxes": [], "names": []}
            if os.path.isfile(xml_path):
                parsed = _parse_voc_xml(xml_path)
                if not parsed["boxes"]:
                    qa["xml_without_boxes"].append(f"{patient_dir}/{stem}")
            else:
                qa["missing_xml"].append(f"{patient_dir}/{stem}")
                xml_path = ""

            # Cross-check the XML object name against the directory class. The
            # directory wins; a mismatch is recorded, never silently resolved.
            xml_names = sorted(set(parsed["names"]))
            mismatch = False
            for name in xml_names:
                mapped = RAW_CLASS_TO_CLUSTER.get(name)
                if mapped is not None and mapped != cluster:
                    mismatch = True
            if mismatch:
                qa["class_mismatches"].append(
                    {"slice": f"{patient_dir}/{stem}", "dir_class": raw_class, "xml": xml_names}
                )

            boxes = parsed["boxes"]
            if boxes:
                union = [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ]
            else:
                union = [None, None, None, None]

            rows.append(
                {
                    "slice_uid": f"{patient_dir}/{stem}",
                    "patient_uid": patient_dir,
                    "numeric_id": numeric_id,
                    "raw_class": raw_class,
                    "cluster": cluster,
                    "cluster_idx": CLUSTER_TO_IDX[cluster],
                    "study_uid": study_uid,
                    "slice_index": slice_index,
                    "image_path": image_path,
                    "xml_path": xml_path,
                    "width": parsed["width"],
                    "height": parsed["height"],
                    "n_lesions": len(boxes),
                    "bbox_union_x0": union[0],
                    "bbox_union_y0": union[1],
                    "bbox_union_x1": union[2],
                    "bbox_union_y1": union[3],
                    "bboxes_json": json.dumps(boxes),
                    "xml_class_names": json.dumps(xml_names),
                    "class_mismatch": bool(mismatch),
                }
            )

    if not rows:
        raise RuntimeError(f"no slices found under {slices_dir!r}")

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    duplicates = frame["slice_uid"][frame["slice_uid"].duplicated()].tolist()
    if duplicates:
        qa["duplicate_slice_uids"] = duplicates
        raise RuntimeError(f"duplicate slice_uid values in manifest: {duplicates[:5]}")

    qa["numeric_id_collisions"] = {
        numeric: sorted(classes)
        for numeric, classes in sorted(numeric_to_classes.items())
        if len(classes) > 1
    }
    qa.update(summarise(frame))

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        frame.to_csv(output_csv, index=False)
        log.info("wrote manifest with %d slices to %s", len(frame), output_csv)
    if qa_json:
        os.makedirs(os.path.dirname(os.path.abspath(qa_json)), exist_ok=True)
        with open(qa_json, "w") as fh:
            json.dump(qa, fh, indent=2, sort_keys=True, default=str)
        log.info("wrote data-quality report to %s", qa_json)

    _log_qa(qa)
    return frame


def summarise(frame: pd.DataFrame) -> Dict[str, Any]:
    """Counts for the paper's dataset paragraph."""
    slices_per_patient = frame.groupby("patient_uid").size()
    studies_per_patient = frame.groupby("patient_uid")["study_uid"].nunique()

    return {
        "n_slices": int(len(frame)),
        "n_patients": int(frame["patient_uid"].nunique()),
        "n_studies": int(frame["study_uid"].nunique()),
        "slices_per_cluster": {k: int(v) for k, v in frame["cluster"].value_counts().items()},
        "patients_per_cluster": {
            k: int(v)
            for k, v in frame.drop_duplicates("patient_uid")["cluster"].value_counts().items()
        },
        "slices_per_raw_class": {k: int(v) for k, v in frame["raw_class"].value_counts().items()},
        "slices_per_patient": {
            "min": int(slices_per_patient.min()),
            "max": int(slices_per_patient.max()),
            "mean": round(float(slices_per_patient.mean()), 3),
            "median": float(slices_per_patient.median()),
            "histogram": {int(k): int(v) for k, v in Counter(slices_per_patient.values).items()},
        },
        "patients_with_multiple_studies": int((studies_per_patient > 1).sum()),
        "multifocal_slices": int((frame["n_lesions"] > 1).sum()),
        "max_lesions_on_a_slice": int(frame["n_lesions"].max()),
        "slices_without_boxes": int((frame["n_lesions"] == 0).sum()),
        "class_mismatch_slices": int(frame["class_mismatch"].sum()),
    }


def _log_qa(qa: Dict[str, Any]) -> None:
    log.info(
        "manifest: %d slices / %d patients / %d studies",
        qa.get("n_slices", 0),
        qa.get("n_patients", 0),
        qa.get("n_studies", 0),
    )
    log.info("slices per cluster: %s", qa.get("slices_per_cluster"))
    log.info("patients per cluster: %s", qa.get("patients_per_cluster"))

    collisions = qa.get("numeric_id_collisions") or {}
    if collisions:
        log.warning(
            "%d numeric patient IDs are reused across classes - patient_uid "
            "({CLASS}_{ID}) is therefore the only safe grouping key",
            len(collisions),
        )
    for key, message in (
        ("empty_patient_dirs", "patient directories contain no slices"),
        ("missing_xml", "slices have no XML annotation"),
        ("xml_without_boxes", "XMLs contain no bounding box"),
        ("class_mismatches", "slices disagree between directory class and XML object name"),
        ("unparsed_slice_names", "slice names did not match the expected pattern"),
    ):
        items = qa.get(key) or []
        if items:
            log.warning("%d %s (e.g. %s)", len(items), message, items[0])


def load_manifest(path: str) -> pd.DataFrame:
    """Read a manifest written by :func:`build_manifest`, validating structure."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"manifest not found at {path!r}. Run: python -m ppgl.cli inventory"
        )
    frame = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"manifest {path!r} is missing columns: {missing}")
    frame["bboxes"] = frame["bboxes_json"].apply(
        lambda value: json.loads(value) if isinstance(value, str) and value else []
    )
    return frame


__all__ = ["build_manifest", "load_manifest", "summarise", "MANIFEST_COLUMNS"]
