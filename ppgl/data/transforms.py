"""Image transforms.

Augmentation is intentionally mild. At 630 slices the failure mode is
overfitting, but aggressive photometric jitter on already-windowed CT destroys
the soft-tissue contrast that distinguishes these tumours. Vertical flip is off
by default: axial CT has a fixed anterior/posterior orientation and flipping it
produces anatomically impossible images, which matters because the region
estimator reads gross anatomy.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

# ImageNet statistics. Overridden per-backbone from timm's pretrained_cfg where
# available (DINOv2 and RadImageNet use different values).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransformPipeline:
    """Callable mapping a uint8 ``HxW`` array to a normalised ``3xSxS`` tensor.

    A module-level class rather than a closure **because it must be picklable**.
    ``PPGLSliceDataset`` holds this object, and a DataLoader with
    ``num_workers > 0`` pickles the dataset to hand it to each worker process. An
    earlier version returned a nested function from ``build_transforms``, which
    raised ``Can't pickle local object 'build_transforms.<locals>.apply'`` and
    killed every job that used a DataLoader - i.e. every deep model, while the
    tabular controls passed and masked the cause.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def __call__(self, image: np.ndarray):
        import torch

        tensor = self.pipeline(image)
        # Grayscale sources arrive single-channel; ImageNet backbones want three.
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.to(torch.float32)


def build_transforms(
    data_cfg,
    train: bool,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> "TransformPipeline":
    """Return a picklable callable mapping a uint8 HxW array to a 3xSxS tensor."""
    import torch
    from torchvision import transforms as T

    mean = tuple(mean) if mean is not None else IMAGENET_MEAN
    std = tuple(std) if std is not None else IMAGENET_STD
    size = int(data_cfg.image_size)

    steps = [_ToPIL3Channel(hist_equalize=bool(data_cfg.hist_equalize))]

    if train:
        steps.append(T.Resize((size, size), antialias=True))
        affine_needed = (
            data_cfg.aug_rotate_deg > 0 or data_cfg.aug_scale > 0 or data_cfg.aug_translate > 0
        )
        if affine_needed:
            steps.append(
                T.RandomAffine(
                    degrees=float(data_cfg.aug_rotate_deg),
                    translate=(float(data_cfg.aug_translate),) * 2
                    if data_cfg.aug_translate > 0
                    else None,
                    scale=(1.0 - float(data_cfg.aug_scale), 1.0 + float(data_cfg.aug_scale))
                    if data_cfg.aug_scale > 0
                    else None,
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            )
        if data_cfg.aug_hflip:
            steps.append(T.RandomHorizontalFlip(p=0.5))
        if data_cfg.aug_vflip:
            steps.append(T.RandomVerticalFlip(p=0.5))
        if data_cfg.aug_brightness > 0 or data_cfg.aug_contrast > 0:
            steps.append(
                T.ColorJitter(
                    brightness=float(data_cfg.aug_brightness),
                    contrast=float(data_cfg.aug_contrast),
                )
            )
    else:
        steps.append(T.Resize((size, size), antialias=True))

    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=mean, std=std))
    return TransformPipeline(T.Compose(steps))


class PairedTransformPipeline:
    """Transform an image and its lesion mask under the *same* geometry.

    Needed by the attention-supervised model, which compares the ViT's
    class-token attention against the lesion footprint. The ordinary
    :class:`TransformPipeline` cannot be reused for that: ``RandomAffine`` and
    ``RandomHorizontalFlip`` draw their own parameters inside ``__call__``, so
    calling the pipeline twice would rotate the image one way and the mask
    another, and the supervision target would be silently wrong rather than
    obviously broken.

    So the geometric parameters are drawn once here and applied to both, with
    bilinear interpolation for the image and nearest for the mask (a mask must
    stay an indicator, not become a blur). Photometric jitter and normalisation
    touch the image only.

    Module-level class for the same reason as ``TransformPipeline``: a
    DataLoader with ``num_workers > 0`` pickles the dataset that holds it.
    """

    def __init__(
        self,
        size: int,
        mean: Sequence[float],
        std: Sequence[float],
        train: bool,
        hist_equalize: bool = False,
        rotate_deg: float = 0.0,
        scale: float = 0.0,
        translate: float = 0.0,
        hflip: bool = False,
        vflip: bool = False,
        brightness: float = 0.0,
        contrast: float = 0.0,
    ) -> None:
        self.size = int(size)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.train = bool(train)
        self.to_pil = _ToPIL3Channel(hist_equalize=hist_equalize)
        self.rotate_deg = float(rotate_deg)
        self.scale = float(scale)
        self.translate = float(translate)
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        import torch
        from PIL import Image
        from torchvision import transforms as T
        from torchvision.transforms import functional as F

        pil = self.to_pil(image)
        pil = F.resize(pil, [self.size, self.size], antialias=True)

        if mask is None:
            mask_pil = None
        else:
            array = (np.asarray(mask) > 0).astype(np.uint8) * 255
            mask_pil = Image.fromarray(array, mode="L")
            mask_pil = F.resize(
                mask_pil, [self.size, self.size],
                interpolation=T.InterpolationMode.NEAREST,
            )

        if self.train:
            if self.rotate_deg > 0 or self.scale > 0 or self.translate > 0:
                angle, translations, scale, shear = T.RandomAffine.get_params(
                    degrees=(-self.rotate_deg, self.rotate_deg),
                    translate=(self.translate, self.translate) if self.translate > 0 else None,
                    scale_ranges=(1.0 - self.scale, 1.0 + self.scale) if self.scale > 0 else None,
                    shears=None,
                    img_size=[self.size, self.size],
                )
                pil = F.affine(
                    pil, angle=angle, translate=list(translations), scale=scale, shear=list(shear),
                    interpolation=T.InterpolationMode.BILINEAR,
                )
                if mask_pil is not None:
                    mask_pil = F.affine(
                        mask_pil, angle=angle, translate=list(translations), scale=scale,
                        shear=list(shear), interpolation=T.InterpolationMode.NEAREST,
                    )
            if self.hflip and torch.rand(1).item() < 0.5:
                pil = F.hflip(pil)
                if mask_pil is not None:
                    mask_pil = F.hflip(mask_pil)
            if self.vflip and torch.rand(1).item() < 0.5:
                pil = F.vflip(pil)
                if mask_pil is not None:
                    mask_pil = F.vflip(mask_pil)
            if self.brightness > 0 or self.contrast > 0:
                pil = T.ColorJitter(brightness=self.brightness, contrast=self.contrast)(pil)

        tensor = F.to_tensor(pil)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        tensor = F.normalize(tensor.to(torch.float32), mean=list(self.mean), std=list(self.std))

        if mask_pil is None:
            return tensor, None
        mask_tensor = (F.to_tensor(mask_pil) > 0.5).to(torch.float32)
        return tensor, mask_tensor


def build_paired_transforms(
    data_cfg,
    train: bool,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> "PairedTransformPipeline":
    """Paired image/mask transform, matching ``build_transforms`` settings."""
    return PairedTransformPipeline(
        size=int(data_cfg.image_size),
        mean=tuple(mean) if mean is not None else IMAGENET_MEAN,
        std=tuple(std) if std is not None else IMAGENET_STD,
        train=train,
        hist_equalize=bool(data_cfg.hist_equalize),
        rotate_deg=float(data_cfg.aug_rotate_deg),
        scale=float(data_cfg.aug_scale),
        translate=float(data_cfg.aug_translate),
        hflip=bool(data_cfg.aug_hflip),
        vflip=bool(data_cfg.aug_vflip),
        brightness=float(data_cfg.aug_brightness),
        contrast=float(data_cfg.aug_contrast),
    )


class _ToPIL3Channel:
    """uint8 array -> 3-channel PIL image, with optional histogram equalisation.

    Equalisation is off by default. The archive is already windowed at 50/450,
    and per-slice equalisation removes the inter-scan intensity comparability
    that any density-based discrimination would rely on.
    """

    def __init__(self, hist_equalize: bool = False) -> None:
        self.hist_equalize = hist_equalize

    def __call__(self, image):
        from PIL import Image, ImageOps

        array = np.asarray(image)
        if array.dtype != np.uint8:
            finite = array[np.isfinite(array)] if array.size else array
            lo = float(finite.min()) if finite.size else 0.0
            hi = float(finite.max()) if finite.size else 1.0
            scale = 255.0 / (hi - lo) if hi > lo else 0.0
            array = np.clip((array.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            pil = Image.fromarray(array, mode="L")
        elif array.ndim == 3 and array.shape[2] >= 3:
            pil = Image.fromarray(array[:, :, :3], mode="RGB")
        else:
            pil = Image.fromarray(array.squeeze(), mode="L")

        if self.hist_equalize:
            pil = ImageOps.equalize(pil)
        return pil.convert("RGB")


def resolve_normalisation(model) -> "tuple":
    """Pull (mean, std) from a timm model's pretrained_cfg, falling back to ImageNet."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    mean = cfg.get("mean") or IMAGENET_MEAN
    std = cfg.get("std") or IMAGENET_STD
    return tuple(mean), tuple(std)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PairedTransformPipeline",
    "TransformPipeline",
    "build_paired_transforms",
    "build_transforms",
    "resolve_normalisation",
]
