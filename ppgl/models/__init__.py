"""Model registry.

:func:`build_model` returns a torch module for the deep models. The non-deep
controls (location-only, region-majority, radiomics + GBM) are sklearn-style
estimators constructed by :mod:`ppgl.experiments.run_tabular`, since they do not
use the torch training loop.

Torch-dependent symbols are resolved lazily so that importing
``ppgl.models.names`` - or anything that only needs the model *names* - does not
require torch. That is what keeps the location-only decision gate and all
post-hoc analysis runnable without a GPU stack.

Registered deep models
----------------------
``resnet101``, ``vit``, ``swin``
    Single-head baselines. Backbone comes from ``model.backbone``.
``vit_cl``
    ViT with a supervised-contrastive projection head (the ``ViT+CL`` baseline).
``ppgl_transformer``
    Reimplementation of the two-branch model of Santra et al. (AMAI 2023).
``routed``
    The proposed anatomy-routed mixture of experts.
``single_head_aux``
    The routed model's ablation: same auxiliary region supervision, one shared
    classifier instead of per-region experts. This is the correct
    "without region weight estimator" control, because it holds the extra label
    constant and removes only the routing.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

# Torch-free: safe to import eagerly.
from ppgl.models.names import (
    ALL_MODELS,
    DEEP_MODELS,
    DEFAULT_BACKBONES,
    TABULAR_MODELS,
    default_backbone,
    is_deep,
    is_tabular,
    requires_regions,
    uses_contrastive,
)

# numpy/sklearn only: also safe.
from ppgl.models.tabular import (
    LocationOnlyClassifier,
    RadiomicsGBMClassifier,
    RegionMajorityClassifier,
)

# Attribute name -> defining submodule. Imported on first access.
_LAZY = {
    "AnatomyRoutedClassifier": "ppgl.models.routed",
    "BACKBONE_ALIASES": "ppgl.models.backbones",
    "ContrastiveClassifier": "ppgl.models.baselines",
    "LoRALinear": "ppgl.models.lora",
    "PPGLTransformer": "ppgl.models.baselines",
    "PrototypeHead": "ppgl.models.baselines",
    "SingleHeadClassifier": "ppgl.models.baselines",
    "build_head": "ppgl.models.baselines",
    "create_backbone": "ppgl.models.backbones",
    "inject_lora": "ppgl.models.lora",
    "layer_wise_param_groups": "ppgl.models.backbones",
    "make_mlp_head": "ppgl.models.baselines",
    "resolve_backbone_name": "ppgl.models.backbones",
    "trainable_parameter_report": "ppgl.models.lora",
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(target), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


def build_model(model_cfg, n_classes: Optional[int] = None, n_regions: Optional[int] = None, **kwargs: Any):
    """Instantiate a deep model from its config. Requires torch."""
    from ppgl import REGIONS
    from ppgl.labels import n_active_classes
    from ppgl.models.baselines import (
        ContrastiveClassifier,
        PPGLTransformer,
        SingleHeadClassifier,
    )
    from ppgl.models.routed import AnatomyRoutedClassifier

    n_classes = n_classes or n_active_classes()
    n_regions = n_regions or len(REGIONS)
    n_regions = n_regions or len(REGIONS)

    name = model_cfg.name
    if name in TABULAR_MODELS:
        raise ValueError(
            f"{name!r} is a tabular control and is not built here; "
            "use ppgl.experiments.run_tabular"
        )

    common = {"n_classes": n_classes, "n_regions": n_regions}

    if name in ("resnet101", "vit", "swin"):
        return SingleHeadClassifier(model_cfg, predict_region=False, **common, **kwargs)
    if name == "single_head_aux":
        return SingleHeadClassifier(model_cfg, predict_region=True, **common, **kwargs)
    if name == "vit_cl":
        return ContrastiveClassifier(model_cfg, predict_region=False, **common, **kwargs)
    if name == "ppgl_transformer":
        return PPGLTransformer(model_cfg, **common, **kwargs)
    if name == "routed":
        return AnatomyRoutedClassifier(
            model_cfg,
            aggregate=getattr(model_cfg, "router_aggregate", "logit"),
            **common,
            **kwargs,
        )

    if name in ("fusion", "clinical_net"):
        from ppgl.models.fusion import ClinicalOnlyNet, ImageClinicalFusion

        n_clinical = kwargs.pop("n_clinical_features", None)
        if not n_clinical:
            raise ValueError(
                f"model {name!r} needs n_clinical_features; the runner must pass the width "
                "of the preprocessed clinical matrix"
            )
        builder = ImageClinicalFusion if name == "fusion" else ClinicalOnlyNet
        return builder(model_cfg, n_clinical_features=int(n_clinical), **common, **kwargs)

    if name == "mil":
        from ppgl.models.mil import AttentionMIL

        return AttentionMIL(model_cfg, **common, **kwargs)

    if name == "attn_lora":
        from ppgl.models.attn_lora import AttentionSupervisedViT

        # Region labels do not exist in this archive, so the auxiliary region head
        # is opt-out: model.aux_region=False removes it entirely.
        return AttentionSupervisedViT(
            model_cfg,
            predict_region=bool(getattr(model_cfg, "aux_region", True)),
            **common,
            **kwargs,
        )

    raise ValueError(f"unknown model {name!r}. Known: {ALL_MODELS}")


__all__ = [
    "ALL_MODELS",
    "AnatomyRoutedClassifier",
    "BACKBONE_ALIASES",
    "ContrastiveClassifier",
    "DEEP_MODELS",
    "DEFAULT_BACKBONES",
    "LocationOnlyClassifier",
    "PPGLTransformer",
    "PrototypeHead",
    "RadiomicsGBMClassifier",
    "RegionMajorityClassifier",
    "SingleHeadClassifier",
    "TABULAR_MODELS",
    "build_head",
    "build_model",
    "create_backbone",
    "default_backbone",
    "is_deep",
    "is_tabular",
    "layer_wise_param_groups",
    "make_mlp_head",
    "requires_regions",
    "resolve_backbone_name",
    "uses_contrastive",
]
