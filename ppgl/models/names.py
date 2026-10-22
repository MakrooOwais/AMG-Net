"""Model names and capability predicates. Deliberately free of torch imports.

Kept separate from :mod:`ppgl.models` so that the CLI, the job dispatcher and the
tabular controls can reason about models without importing torch. That is what
lets the location-only decision gate run on a CPU-only machine, and it keeps
``python -m ppgl.cli --help`` fast.
"""

from __future__ import annotations

# Deep models trained through ppgl.engine.
DEEP_MODELS = (
    "resnet101",
    "vit",
    "swin",
    "vit_cl",
    "ppgl_transformer",
    "routed",
    "single_head_aux",
    # Clinical + imaging fusion, and the torch clinical-only control.
    "fusion",
    "clinical_net",
    # Patient-level attention multiple-instance learning.
    "mil",
    # ViT with LoRA adapters and class-token attention supervised by the lesion box.
    "attn_lora",
)

# Controls trained through ppgl.experiments.run_tabular (numpy + sklearn only).
TABULAR_MODELS = (
    "location_only",
    "region_majority",
    "radiomics_gbm",
    "radiomics_gbm_region",
    "clinical_only",
    "clinical_radiomics",
)

ALL_MODELS = DEEP_MODELS + TABULAR_MODELS

# Backbone each model is defined with when the config leaves it at the default.
DEFAULT_BACKBONES = {
    "resnet101": "resnet101",
    "vit": "vit_base_patch16_224",
    "vit_cl": "vit_base_patch16_224",
    "swin": "swin_tiny_patch4_window7_224",
    "ppgl_transformer": "vit_base_patch16_224",
    "routed": "swin_tiny_patch4_window7_224",
    "single_head_aux": "swin_tiny_patch4_window7_224",
    # Attention supervision needs a ViT: it reads the class token's attention over
    # patches, which a hierarchical/windowed encoder like Swin does not expose in
    # that form. Defaults to the ViT-B/16 already bundled and already a baseline
    # row, so attn_lora minus vit isolates LoRA plus the supervision.
    "attn_lora": "vit_base_patch16_224",
}

# Models that cannot run without anatomical region labels.
_NEEDS_REGIONS = (
    "routed",
    "single_head_aux",
    "ppgl_transformer",
    "location_only",
    "region_majority",
    "radiomics_gbm_region",
    # These two carry the same auxiliary region head as single_head_aux. Declared
    # here so the dependency is visible: both were previously consuming region
    # labels opportunistically whenever the data layer happened to supply them,
    # which is how a reported model ends up depending on provisional labels
    # without anyone deciding that it should.
    "attn_lora",
    "mil",
)

# Models whose region dependence is ONLY an optional auxiliary head, and which
# therefore need no region labels at all once model.aux_region is False.
#
# The distinction matters because region labels do not exist in this archive. For
# `routed` the dependence is structural - routing *is* the model - so it can never
# be switched off. For these two it is one extra linear head, so a region-free
# variant is available and is what a reported method should use.
_OPTIONAL_REGION = ("attn_lora", "mil")

# Models supervised by the per-slice lesion footprint, which the dataset must
# therefore transform alongside the image.
_NEEDS_LESION_MASK = ("attn_lora",)

# Models that cannot run without the clinical covariate sheet.
_NEEDS_CLINICAL = ("fusion", "clinical_net", "clinical_only", "clinical_radiomics")

# Models that consume a whole patient as one training example.
_BAG_MODELS = ("mil",)


def is_tabular(name: str) -> bool:
    return name in TABULAR_MODELS


def is_deep(name: str) -> bool:
    return name in DEEP_MODELS


def needs_clinical(name: str) -> bool:
    """Whether this model requires the clinical covariate CSV."""
    return name in _NEEDS_CLINICAL


def is_bag_model(name: str) -> bool:
    """Whether this model is trained on patient-level bags of slices."""
    return name in _BAG_MODELS


def needs_lesion_mask(name: str) -> bool:
    """Whether this model's loss consumes the per-slice lesion footprint."""
    return name in _NEEDS_LESION_MASK


def requires_regions(name: str) -> bool:
    return name in _NEEDS_REGIONS


def region_dependence_is_optional(name: str) -> bool:
    """Whether this model's region use is just an auxiliary head that can be off."""
    return name in _OPTIONAL_REGION


def needs_region_labels(cfg) -> bool:
    """Whether this configuration genuinely requires anatomical region labels.

    Name alone is not enough: ``attn_lora`` and ``mil`` require them only while
    their auxiliary region head is enabled. Used by the data layer so that a
    region-free configuration never even asks for the provisional labels.
    """
    name = cfg.model.name
    if not requires_regions(name):
        return False
    if region_dependence_is_optional(name) and not bool(getattr(cfg.model, "aux_region", True)):
        return False
    return True


def uses_contrastive(name: str) -> bool:
    return name == "vit_cl"


def default_backbone(name: str) -> str:
    return DEFAULT_BACKBONES.get(name, "swin_tiny_patch4_window7_224")


__all__ = [
    "ALL_MODELS",
    "DEEP_MODELS",
    "DEFAULT_BACKBONES",
    "TABULAR_MODELS",
    "default_backbone",
    "is_bag_model",
    "is_deep",
    "is_tabular",
    "needs_clinical",
    "needs_lesion_mask",
    "needs_region_labels",
    "region_dependence_is_optional",
    "requires_regions",
    "uses_contrastive",
]
