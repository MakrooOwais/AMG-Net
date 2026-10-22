"""Slice reading and the torch Dataset.

Image facts for this archive, verified by inspecting the TIFF IFDs directly:
512x512, 8-bit, ``SamplesPerPixel=4`` (RGBA), uncompressed, and **already
windowed at 50 HU level / 450 HU width**. Hounsfield units are therefore
unrecoverable: no re-windowing, no multi-window inputs, and no HU-calibrated
radiomics are possible on this data. The alpha channel is dropped; the three
colour channels are identical grayscale replicas.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ppgl.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_slice(path: str, as_gray: bool = True) -> np.ndarray:
    """Read one slice as a uint8 array.

    Returns ``(H, W)`` when ``as_gray``, else ``(H, W, 3)``. Handles RGBA, RGB,
    grayscale and 16-bit sources; 16-bit input is min-max scaled to 8-bit.
    """
    array = _read_raw(path)

    if array.ndim == 3:
        # Drop alpha, then collapse identical colour channels.
        if array.shape[2] >= 4:
            array = array[:, :, :3]
        if as_gray:
            if array.shape[2] == 3 and _channels_identical(array):
                array = array[:, :, 0]
            else:
                array = (
                    0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
                ).astype(array.dtype)
    elif array.ndim == 2 and not as_gray:
        array = np.repeat(array[:, :, None], 3, axis=2)

    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        array = np.clip((array.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def _read_raw(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"slice not found: {path!r}")
    try:
        from PIL import Image

        with Image.open(path) as handle:
            return np.array(handle)
    except Exception as pil_error:
        try:
            import tifffile

            return tifffile.imread(path)
        except Exception as tiff_error:
            raise RuntimeError(
                f"could not read {path!r}: PIL failed ({pil_error}); tifffile failed ({tiff_error})"
            )


def _channels_identical(array: np.ndarray, sample: int = 4096) -> bool:
    flat = array.reshape(-1, array.shape[2])
    if flat.shape[0] > sample:
        idx = np.linspace(0, flat.shape[0] - 1, sample).astype(int)
        flat = flat[idx]
    return bool(np.array_equal(flat[:, 0], flat[:, 1]) and np.array_equal(flat[:, 1], flat[:, 2]))


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #
def compute_crop_box(
    boxes: Sequence[Sequence[float]],
    image_shape: Tuple[int, int],
    crop_mode: str = "roi_context",
    context_scale: float = 2.0,
    min_crop_px: int = 48,
) -> Optional[Tuple[int, int, int, int]]:
    """Crop window as ``(x0, y0, x1, y1)``, or ``None`` for the whole slice.

    ``roi`` is the tight union of lesion boxes. ``roi_context`` scales that union
    about its centre by ``context_scale`` so the model still sees the surrounding
    anatomy, which is what the region estimator depends on. The window is
    squared to avoid anisotropic distortion on resize, and clamped to the image.
    """
    height, width = image_shape[:2]
    if crop_mode == "full" or not boxes:
        return None

    x0 = min(float(b[0]) for b in boxes)
    y0 = min(float(b[1]) for b in boxes)
    x1 = max(float(b[2]) for b in boxes)
    y1 = max(float(b[3]) for b in boxes)

    if crop_mode == "roi_context":
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = max(x1 - x0, y1 - y0) * context_scale / 2.0
        half = max(half, min_crop_px / 2.0)
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
    else:  # "roi": square it up to the longer side
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = max(max(x1 - x0, y1 - y0) / 2.0, min_crop_px / 2.0)
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half

    # Shift rather than shrink when the window runs off an edge, so the
    # requested field of view is preserved where possible.
    side = min(max(x1 - x0, y1 - y0), min(width, height))
    cx = float(np.clip((x0 + x1) / 2.0, side / 2.0, width - side / 2.0))
    cy = float(np.clip((y0 + y1) / 2.0, side / 2.0, height - side / 2.0))

    ix0 = int(round(cx - side / 2.0))
    iy0 = int(round(cy - side / 2.0))
    ix1 = int(round(ix0 + side))
    iy1 = int(round(iy0 + side))
    ix0, iy0 = max(0, ix0), max(0, iy0)
    ix1, iy1 = min(width, ix1), min(height, iy1)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    return ix0, iy0, ix1, iy1


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PPGLSliceDataset:
    """One item per annotated axial slice.

    Each item is a dict with:

    ``image``
        Float tensor ``(3, S, S)``, normalised.
    ``cluster``
        Long tensor scalar, index into :data:`ppgl.CLUSTERS`.
    ``region``
        Long tensor scalar, index into :data:`ppgl.REGIONS`, or ``-1`` if absent.
    ``region_confidence``
        Float scalar. Lets the loss down-weight provisional region labels.
    ``index``, ``slice_uid``, ``patient_uid``
        Identifiers, carried through so predictions can be joined back to the
        manifest for patient-level aggregation.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_cfg,
        train: bool,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        cache_images: bool = False,
        clinical: Optional[np.ndarray] = None,
        return_mask: bool = False,
    ) -> None:
        from ppgl.data.transforms import build_paired_transforms, build_transforms

        self.frame = frame.reset_index(drop=True)
        self.cfg = data_cfg
        self.train = train
        # ``return_mask`` switches to the paired pipeline so the lesion footprint
        # undergoes exactly the same geometry as the image. Only the
        # attention-supervised model needs it; every other model keeps the
        # original single-image pipeline untouched.
        self.return_mask = bool(return_mask)
        if self.return_mask:
            self.transform = build_paired_transforms(data_cfg, train=train, mean=mean, std=std)
        else:
            self.transform = build_transforms(data_cfg, train=train, mean=mean, std=std)
        self.cache_images = cache_images
        self._cache: Dict[int, np.ndarray] = {}

        # Preprocessed clinical matrix, row-aligned to ``frame``. Supplied by the
        # runner because the imputation and scaling must be fit on the training
        # fold only; the dataset never computes it.
        if clinical is not None and len(clinical) != len(self.frame):
            raise ValueError(
                f"clinical matrix has {len(clinical)} rows but the frame has {len(self.frame)}"
            )
        self.clinical = None if clinical is None else np.asarray(clinical, dtype=np.float32)

        self._boxes: List[List[List[float]]] = []
        for value in self.frame.get("bboxes_json", pd.Series([], dtype=object)):
            if isinstance(value, str) and value:
                try:
                    self._boxes.append(json.loads(value))
                except json.JSONDecodeError:
                    self._boxes.append([])
            elif isinstance(value, list):
                self._boxes.append(value)
            else:
                self._boxes.append([])

        self.has_regions = "region_idx" in self.frame.columns

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def cluster_labels(self) -> np.ndarray:
        return self.frame["cluster_idx"].to_numpy(dtype=np.int64)

    @property
    def region_labels(self) -> np.ndarray:
        if not self.has_regions:
            return np.full(len(self.frame), -1, dtype=np.int64)
        return self.frame["region_idx"].to_numpy(dtype=np.int64)

    @property
    def patient_uids(self) -> np.ndarray:
        return self.frame["patient_uid"].to_numpy()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.frame.iloc[index]

        if self.cache_images and index in self._cache:
            image = self._cache[index]
        else:
            image = read_slice(row["image_path"], as_gray=True)
            if self.cache_images:
                self._cache[index] = image

        box = compute_crop_box(
            self._boxes[index],
            image.shape,
            crop_mode=self.cfg.crop_mode,
            context_scale=self.cfg.roi_context_scale,
            min_crop_px=self.cfg.min_crop_px,
        )
        mask = self._lesion_mask(index, image.shape) if self.return_mask else None

        if box is not None:
            x0, y0, x1, y1 = box
            image = image[y0:y1, x0:x1]
            if mask is not None:
                mask = mask[y0:y1, x0:x1]

        if self.return_mask:
            tensor, mask_tensor = self.transform(image, mask)
        else:
            tensor, mask_tensor = self.transform(image), None

        region = int(row["region_idx"]) if self.has_regions else -1
        confidence = float(row.get("region_confidence", 1.0)) if self.has_regions else 0.0

        item: Dict[str, Any] = {
            "image": tensor,
            "cluster": torch.tensor(int(row["cluster_idx"]), dtype=torch.long),
            "region": torch.tensor(region, dtype=torch.long),
            "region_confidence": torch.tensor(confidence, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
            "slice_uid": str(row["slice_uid"]),
            "patient_uid": str(row["patient_uid"]),
        }
        if self.clinical is not None:
            item["clinical"] = torch.from_numpy(self.clinical[index])
        if mask_tensor is not None:
            item["lesion_mask"] = mask_tensor
        return item

    def _lesion_mask(self, index: int, shape) -> np.ndarray:
        """Binary lesion footprint in source-image coordinates.

        Built from the annotation boxes rather than a true segmentation, because
        the archive's XMLs contain bounding boxes only. That is a coarser target
        than a contour - a box over a round lesion includes roughly 21% background
        - so the supervision it provides is 'attend to this neighbourhood', not
        'attend to these voxels'. Worth stating plainly in the paper.
        """
        height, width = int(shape[0]), int(shape[1])
        mask = np.zeros((height, width), dtype=np.uint8)
        for box in self._boxes[index] or []:
            try:
                x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            except (TypeError, IndexError, ValueError):
                continue
            xa, xb = sorted((int(round(x0)), int(round(x1))))
            ya, yb = sorted((int(round(y0)), int(round(y1))))
            xa, ya = max(xa, 0), max(ya, 0)
            xb, yb = min(max(xb, xa + 1), width), min(max(yb, ya + 1), height)
            mask[ya:yb, xa:xb] = 1
        return mask


def collate_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors, keep string identifiers as lists."""
    import torch

    out: Dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], str):
            out[key] = values
        else:
            out[key] = torch.stack(values)
    return out


__all__ = [
    "PPGLSliceDataset",
    "collate_batch",
    "compute_crop_box",
    "read_slice",
]
