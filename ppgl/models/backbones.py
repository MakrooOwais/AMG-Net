"""Backbone factory.

The encoder-comparison axis for the paper. The literature on whether
radiology-specific pretraining beats ImageNet at n<1000 is genuinely split -
RadImageNet reports wins (Mei et al. 2022) while RadiologyNET finds parity
(Napravnik et al. 2025) - so the comparison is run on our own folds rather than
asserted from citations.

Available and appropriate for this data::

    swin_tiny_patch4_window7_224      ImageNet Swin-T, the primary anchor
    vit_base_patch16_224              ImageNet ViT-B, for the ViT baselines
    resnet101 / resnet50              ImageNet CNN baselines
    vit_base_patch14_dinov2.lvd142m   DINOv2 self-supervised features
    resnet50 + backbone_weights=...   RadImageNet weights loaded locally

Deliberately excluded: CT-FM, CT-CLIP, Merlin and every other 3D CT foundation
model. They ingest raw Hounsfield volumes. This archive stores 8-bit slices
already windowed at 50/450 HU, so HU is unrecoverable and there is no z-axis;
replicating a slice into a fake volume discards exactly the inductive bias those
encoders provide. The exclusion is stated in the paper rather than worked around.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ppgl.models import weights
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

# Convenience aliases so configs stay readable.
BACKBONE_ALIASES = {
    "swin": "swin_tiny_patch4_window7_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "vit": "vit_base_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
}


def resolve_backbone_name(name: str) -> str:
    return BACKBONE_ALIASES.get(name, name)


def _is_vit_family(name: str) -> bool:
    return name.startswith("vit_") or "dinov2" in name or "deit" in name


_DOWNLOAD_ERROR_MARKERS = (
    "cannot find the requested files in the local cache",
    "check your connection",
    "couldn't connect",
    "could not connect",
    "connectionerror",
    "offline mode",
    "hf_hub_offline",
    "max retries exceeded",
    "name or service not known",
    "temporary failure in name resolution",
    "no such host",
    "timed out",
)


def _is_download_failure(error: Optional[BaseException]) -> bool:
    """Whether an exception is a weight-download failure rather than a bad name."""
    if error is None:
        return False
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _DOWNLOAD_ERROR_MARKERS)


def required_backbones(config_globs: Sequence[str] = ("configs/**/*.yaml",)) -> List[str]:
    """Distinct backbones referenced by the config files, for pre-fetching."""
    import glob

    import yaml

    names = set()
    for pattern in config_globs:
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                with open(path) as handle:
                    payload = yaml.safe_load(handle) or {}
            except Exception:
                continue
            model = payload.get("model") or {}
            backbone = model.get("backbone")
            # Skip configs that deliberately load local weights instead.
            if backbone and not model.get("backbone_weights"):
                names.add(resolve_backbone_name(str(backbone)))
    if not names:
        names.add(resolve_backbone_name("swin"))
    return sorted(names)


def check_pretrained_available(
    backbones: Optional[Sequence[str]] = None,
    weights_dir: Optional[str] = None,
) -> Dict[str, str]:
    """How each backbone's weights would be obtained, without downloading anything.

    Returns a name -> one of ``"bundled"``, ``"hub cache"``, ``"MISSING"``. Used by
    ``ppgl.cli doctor`` so an absent checkpoint is caught in seconds rather than
    after the grid has been dispatched across every GPU.
    """
    names = list(backbones) if backbones else required_backbones()
    out: Dict[str, str] = {}

    for name in names:
        if weights.local_path(name, weights_dir):
            out[name] = "bundled"
            continue
        out[name] = "MISSING"

    # Only pay for the timm/HF probe for backbones with no bundled checkpoint.
    unresolved = [name for name, state in out.items() if state == "MISSING"]
    if not unresolved:
        return out

    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import timm

        for name in unresolved:
            try:
                timm.create_model(name, pretrained=True, num_classes=0)
                out[name] = "hub cache"
            except Exception:
                out[name] = "MISSING"
    except Exception:
        pass
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
    return out


def create_backbone(model_cfg) -> Tuple[nn.Module, int, Tuple[float, ...], Tuple[float, ...]]:
    """Build a feature-extracting backbone.

    Returns ``(module, feature_dim, mean, std)``. The module maps a
    ``(B, 3, H, W)`` tensor to ``(B, feature_dim)``; classifier heads are removed
    and global pooling is applied by timm.
    """
    import timm

    name = resolve_backbone_name(model_cfg.backbone)
    pretrained = bool(model_cfg.pretrained) and not model_cfg.backbone_weights

    # ViT-family checkpoints are tied to the resolution they were trained at
    # (DINOv2 ViT-B/14 defaults to 518). dynamic_img_size lets timm interpolate
    # the positional embeddings so the same encoder accepts our 224 crops.
    attempts = [{"dynamic_img_size": True}, {}] if _is_vit_family(name) else [{}]

    # Prefer the checkpoint bundled with the project. Tried before any network
    # access, so an air-gapped host never attempts a download at all.
    #
    # The weights are handed to timm through pretrained_cfg_overlay(file=...)
    # rather than loaded by hand: timm treats 'file' as a first-class weight
    # source and still applies its checkpoint filter, which resamples positional
    # embeddings for a different input resolution and adapts the stem and
    # classifier. A manual load_state_dict would skip that and quietly break
    # every ViT used away from its native resolution.
    bundled: Optional[str] = None
    if pretrained:
        bundled = weights.local_path(name, getattr(model_cfg, "weights_dir", None))

    backbone = None
    last_error: Optional[Exception] = None

    if bundled:
        for extra in attempts:
            try:
                backbone = timm.create_model(
                    name,
                    pretrained=True,
                    num_classes=0,
                    pretrained_cfg_overlay=dict(file=bundled),
                    **extra,
                )
                log.info("loaded bundled weights for %s from %s", name, bundled)
                break
            except Exception as exc:  # noqa: BLE001 - fall through
                last_error = exc
        if backbone is None:
            log.warning(
                "bundled checkpoint %s could not be loaded into %s (%s); "
                "falling back to the hub",
                bundled,
                name,
                last_error,
            )

    if backbone is None:
        for extra in attempts:
            try:
                backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, **extra)
                break
            except Exception as exc:  # noqa: BLE001 - fall through to the next attempt
                last_error = exc

    if backbone is None and pretrained and _is_download_failure(last_error):
        # The weights could not be fetched. Training a 28M-parameter backbone from
        # scratch on 630 slices is not a substitute, so this fails loudly with the
        # fix rather than silently degrading - unless the caller has explicitly
        # opted in to random initialisation.
        if not bool(getattr(model_cfg, "allow_random_init", False)):
            expected = weights.WEIGHT_SOURCES.get(name)
            expected_file = expected.filename if expected else f"{name}.safetensors"
            raise RuntimeError(
                f"no pretrained weights available for {name!r}: no bundled checkpoint was "
                f"found and the Hugging Face hub is unreachable.\n"
                "\n"
                f"Expected a bundled checkpoint at:\n"
                f"  {os.path.join(weights.resolve_weights_dir(getattr(model_cfg, 'weights_dir', None)), expected_file)}\n"
                "\n"
                "Fix one of these:\n"
                "  1. Bundled weights are shipped with this project. If weights/ is missing, the\n"
                "     archive was unpacked without it, or model.weights_dir / $PPGL_WEIGHTS_DIR\n"
                "     points elsewhere. Check:  python -m ppgl.cli doctor\n"
                "  2. On a host WITH internet:  python -m ppgl.cli fetch-weights\n"
                "     then copy the weights/ directory to this machine.\n"
                "  3. To proceed WITHOUT pretrained weights (not publishable at n=630, but it\n"
                "     exercises the pipeline):  --set model.allow_random_init=true\n"
                f"\nUnderlying error: {last_error}"
            )
        log.error(
            "could not download pretrained weights for %s; falling back to RANDOM "
            "INITIALISATION because model.allow_random_init=true. Results are not "
            "comparable to any pretrained run and must not be reported.",
            name,
        )
        for extra in attempts:
            try:
                backbone = timm.create_model(name, pretrained=False, num_classes=0, **extra)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc

    if backbone is None:
        raise RuntimeError(
            f"could not create timm backbone {name!r}. Check the name against "
            f"`timm.list_models()`. Original error: {last_error}"
        )

    feature_dim = _infer_feature_dim(backbone)
    mean, std = _pretrained_stats(backbone)

    if model_cfg.backbone_weights:
        _load_local_weights(backbone, model_cfg.backbone_weights)

    _apply_finetune_strategy(backbone, model_cfg)

    log.info(
        "backbone %s: feature_dim=%d strategy=%s trainable=%d/%d",
        name,
        feature_dim,
        model_cfg.finetune,
        sum(p.numel() for p in backbone.parameters() if p.requires_grad),
        sum(p.numel() for p in backbone.parameters()),
    )
    return backbone, feature_dim, mean, std


def _infer_feature_dim(backbone: nn.Module) -> int:
    for attribute in ("num_features", "embed_dim", "feature_dim"):
        value = getattr(backbone, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    # Fall back to a forward pass with a dummy input.
    backbone.eval()
    size = 224
    cfg = getattr(backbone, "pretrained_cfg", None) or {}
    input_size = cfg.get("input_size")
    if isinstance(input_size, (tuple, list)) and len(input_size) == 3:
        size = int(input_size[-1])
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, size, size))
    return int(out.shape[-1])


def _pretrained_stats(backbone: nn.Module) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    from ppgl.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    cfg = getattr(backbone, "pretrained_cfg", None) or {}
    mean = tuple(cfg.get("mean") or IMAGENET_MEAN)
    std = tuple(cfg.get("std") or IMAGENET_STD)
    return mean, std


def _load_local_weights(backbone: nn.Module, path: str) -> None:
    """Load a local state_dict, e.g. RadImageNet ResNet50 converted to torch."""
    state = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict):
        raise ValueError(f"{path!r} does not contain a state_dict")

    cleaned = {}
    for key, value in state.items():
        stripped = key
        for prefix in ("module.", "backbone.", "encoder."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
        cleaned[stripped] = value

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    log.info(
        "loaded local backbone weights from %s (missing=%d unexpected=%d)",
        path,
        len(missing),
        len(unexpected),
    )
    if len(missing) > 0.5 * sum(1 for _ in backbone.state_dict()):
        log.warning(
            "more than half the backbone parameters were not found in %s; "
            "check that the checkpoint matches the architecture",
            path,
        )


def _apply_finetune_strategy(backbone: nn.Module, model_cfg) -> None:
    strategy = model_cfg.finetune

    if strategy == "llrd":
        for param in backbone.parameters():
            param.requires_grad = True
        return

    if strategy == "linear_probe":
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        return

    if strategy == "lora":
        from ppgl.models.lora import inject_lora, mark_only_lora_trainable

        for param in backbone.parameters():
            param.requires_grad = False
        count, names = inject_lora(
            backbone,
            targets=tuple(model_cfg.lora_targets),
            r=int(model_cfg.lora_r),
            alpha=float(model_cfg.lora_alpha),
        )
        mark_only_lora_trainable(backbone)
        if count == 0:
            raise RuntimeError(
                f"LoRA targets {tuple(model_cfg.lora_targets)} matched no nn.Linear layer in "
                f"{model_cfg.backbone!r}. Inspect the module names and set model.lora_targets."
            )
        log.info("injected LoRA into %d layers (e.g. %s)", count, names[:3])
        return

    raise ValueError(f"unknown finetune strategy {strategy!r}")


# --------------------------------------------------------------------------- #
# Layer-wise learning-rate decay
# --------------------------------------------------------------------------- #
def layer_wise_param_groups(
    backbone: nn.Module,
    head_modules: List[nn.Module],
    base_lr: float,
    decay: float = 0.7,
    weight_decay: float = 0.05,
) -> List[Dict[str, Any]]:
    """Parameter groups implementing layer-wise LR decay.

    The head trains at ``base_lr``; each block deeper into the backbone is scaled
    by ``decay``, so early generic features move least. This is the standard
    recipe in recent small-medical-data work, and replaces the fixed 1e-2 used in an
    earlier version of this work, which was large enough to damage the pretrained
    representation.

    Parameters with ``ndim <= 1`` (norms and biases) are excluded from weight
    decay.
    """
    depths = _assign_depths(backbone)
    max_depth = max(depths.values()) if depths else 0

    buckets: Dict[Tuple[int, bool], Dict[str, Any]] = {}
    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        depth = depths.get(name, max_depth)
        no_decay = param.ndim <= 1
        key = (depth, no_decay)
        if key not in buckets:
            scale = decay ** (max_depth - depth)
            buckets[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"backbone_depth{depth}{'_nodecay' if no_decay else ''}",
                "lr_scale": scale,
            }
        buckets[key]["params"].append(param)

    groups = [group for group in buckets.values() if group["params"]]

    head_decay, head_no_decay = [], []
    for module in head_modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            (head_no_decay if param.ndim <= 1 else head_decay).append(param)
    if head_decay:
        groups.append(
            {
                "params": head_decay,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "head",
                "lr_scale": 1.0,
            }
        )
    if head_no_decay:
        groups.append(
            {
                "params": head_no_decay,
                "lr": base_lr,
                "weight_decay": 0.0,
                "name": "head_nodecay",
                "lr_scale": 1.0,
            }
        )

    if not groups:
        raise RuntimeError("no trainable parameters found; check the finetune strategy")
    return groups


def _assign_depths(backbone: nn.Module) -> Dict[str, int]:
    """Map parameter name -> integer depth, 0 = closest to the input.

    Handles the two structural families we use: ViT-style ``blocks.N`` and
    Swin/ResNet-style ``layers.N``/``layerN``.
    """
    depths: Dict[str, int] = {}
    stage_ids: Dict[str, int] = {}
    next_id = 1

    for name, _ in backbone.named_parameters():
        parts = name.split(".")
        stage_key: Optional[str] = None
        for index, part in enumerate(parts):
            if part in ("blocks", "layers", "stages") and index + 1 < len(parts):
                stage_key = ".".join(parts[: index + 2])
                break
            if part.startswith("layer") and part[5:].isdigit():
                stage_key = part
                break
        if stage_key is None:
            # Patch embedding / cls token / positional embedding sit at depth 0;
            # final norms sit at the deepest level.
            depths[name] = 0 if _is_stem(name) else -1
            continue
        if stage_key not in stage_ids:
            stage_ids[stage_key] = next_id
            next_id += 1
        depths[name] = stage_ids[stage_key]

    deepest = max([v for v in depths.values() if v >= 0] or [0])
    for name, depth in depths.items():
        if depth < 0:
            depths[name] = deepest + 1
    return depths


def _is_stem(name: str) -> bool:
    return any(
        token in name
        for token in ("patch_embed", "cls_token", "pos_embed", "conv1", "bn1", "stem")
    )


__all__ = [
    "BACKBONE_ALIASES",
    "check_pretrained_available",
    "create_backbone",
    "layer_wise_param_groups",
    "required_backbones",
    "resolve_backbone_name",
]
