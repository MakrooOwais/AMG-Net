"""Mixup and CutMix.

Both are strong regularisers in the small-sample regime this dataset sits in
(630 slices, 287 patients), and they are cheap: no extra forward passes, no
architecture change.

How the loss is handled matters here. The training objective is not plain
cross-entropy - it also carries a Tversky term, an auxiliary region term and
optionally a router penalty, none of which accept soft targets. So rather than
building a soft label vector, the batch is evaluated twice against the two hard
label sets and the resulting losses are blended by ``lam``:

    L = lam * L(outputs, labels_a) + (1 - lam) * L(outputs, labels_b)

For cross-entropy this is exactly equivalent to the usual soft-target form. For
the Tversky term it is an approximation, but a consistent one, and it keeps every
component of the objective intact instead of silently disabling them whenever
mixing fires.

Two details that are easy to get wrong and are handled explicitly:

*Every* per-sample model input is mixed, not just the image. The fusion model also
consumes a clinical vector; mixing the image while leaving the covariates attached
to the original patient would ask that patient's covariates to predict the other
patient's label with weight ``1 - lam``, which is label noise rather than
regularisation. Both branches therefore carry the same convex mixture.

For CutMix, ``lam`` is recomputed from the box that was actually pasted. The box
is clipped at the image border, so the area realised differs substantially from
the area the Beta draw asked for - measured at ~0.18 mean absolute difference on
224x224 inputs. Trusting the requested value is a well-known CutMix bug that
mislabels every edge-adjacent paste.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# Per-sample targets that must follow the permutation when the loss is blended.
#
# ``lesion_mask`` belongs here rather than with the mixed inputs. Under mixup the
# image is lam*A + (1-lam)*B, and the blended objective evaluates the batch once
# against A's targets and once against B's; the attention-localisation term must
# therefore see A's lesion in the first evaluation and B's in the second. Blending
# the two masks into one target instead would ask the attention to cover both
# lesions simultaneously, which is not what the blended loss expresses.
_TARGET_KEYS = ("cluster", "region", "region_confidence", "lesion_mask")

# Per-sample model inputs that are mixed alongside the image, by the same lam.
_AUX_INPUT_KEYS = ("clinical",)


@dataclass
class MixResult:
    """A mixed batch plus what is needed to blend the two loss evaluations."""

    batch: Dict[str, Any]
    permutation: torch.Tensor
    lam: float
    mode: str

    @property
    def active(self) -> bool:
        return self.mode != "none"


def _sample_lam(alpha: float, rng: np.random.Generator) -> float:
    if alpha <= 0:
        return 1.0
    return float(rng.beta(alpha, alpha))


def _rand_bbox(height: int, width: int, lam: float, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """A random box covering ``1 - lam`` of the area, clipped to the image."""
    ratio = float(np.sqrt(max(1.0 - lam, 0.0)))
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    if cut_h < 1 or cut_w < 1:
        return 0, 0, 0, 0

    centre_y = int(rng.integers(0, height))
    centre_x = int(rng.integers(0, width))
    y0 = int(np.clip(centre_y - cut_h // 2, 0, height))
    y1 = int(np.clip(centre_y + cut_h // 2, 0, height))
    x0 = int(np.clip(centre_x - cut_w // 2, 0, width))
    x1 = int(np.clip(centre_x + cut_w // 2, 0, width))
    return y0, y1, x0, x1


def _blend_aux(batch: Dict[str, Any], permutation: torch.Tensor, lam: float) -> Dict[str, Any]:
    """Convex-combine the non-image per-sample inputs by the same ``lam``."""
    out = dict(batch)
    for key in _AUX_INPUT_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == permutation.shape[0]:
            out[key] = lam * value + (1.0 - lam) * value[permutation]
    return out


def mix_batch(
    batch: Dict[str, Any],
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    prob: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Optional[MixResult]:
    """Mix a training batch, or return ``None`` when mixing does not apply.

    Chooses mixup or cutmix at random when both alphas are set. Returns ``None``
    when neither alpha is positive, when the draw does not fire, when the batch
    has fewer than two samples, when the pasted box degenerates to nothing, or
    when the image tensor is bag-shaped (5-D) - blending across bags would mix
    instances belonging to different patients and destroy the attention
    semantics.
    """
    rng = rng or np.random.default_rng()

    images = batch.get("image")
    if not torch.is_tensor(images) or images.dim() != 4 or images.shape[0] < 2:
        return None

    candidates = []
    if mixup_alpha > 0:
        candidates.append("mixup")
    if cutmix_alpha > 0:
        candidates.append("cutmix")
    if not candidates or float(rng.random()) >= prob:
        return None

    mode = candidates[int(rng.integers(0, len(candidates)))]
    permutation = torch.randperm(images.shape[0], device=images.device)

    if mode == "mixup":
        lam = _sample_lam(mixup_alpha, rng)
        mixed = lam * images + (1.0 - lam) * images[permutation]
    else:
        lam = _sample_lam(cutmix_alpha, rng)
        height, width = images.shape[-2], images.shape[-1]
        y0, y1, x0, x1 = _rand_bbox(height, width, lam, rng)
        if y1 <= y0 or x1 <= x0:
            return None
        mixed = images.clone()
        mixed[..., y0:y1, x0:x1] = images[permutation][..., y0:y1, x0:x1]
        # Recompute from the area actually pasted: the box was clipped at the
        # border, so the requested lam is not the realised one.
        lam = 1.0 - ((y1 - y0) * (x1 - x0) / float(height * width))

    out = _blend_aux(batch, permutation, lam)
    out["image"] = mixed
    return MixResult(batch=out, permutation=permutation, lam=lam, mode=mode)


def permuted_batch(batch: Dict[str, Any], permutation: torch.Tensor) -> Dict[str, Any]:
    """The same batch with its per-sample targets permuted to match a mix."""
    out = dict(batch)
    for key in _TARGET_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[0] == permutation.shape[0]:
            out[key] = value[permutation]
    return out


__all__ = ["MixResult", "mix_batch", "permuted_batch"]
