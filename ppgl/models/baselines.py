"""Baseline classifiers, re-implemented so they can be trained on our own folds.

An earlier version of this work quoted baseline numbers from other papers rather
than re-running them, which made paired significance testing impossible and left
no baseline AUCs to compare against. Every baseline here is trained and
evaluated by us on identical folds.

All models share one output contract, a dict with:

``logits``          ``(B, n_classes)`` - required
``region_logits``   ``(B, n_regions)`` - when the model predicts region
``router_probs``    ``(B, n_regions)`` - routed models only
``expert_logits``   ``(B, n_regions, n_classes)`` - routed models only
``embedding``       ``(B, D)`` - projection for the contrastive baseline
``features``        ``(B, D)`` - pooled backbone features, for Grad-CAM/analysis
"""

from __future__ import annotations

from typing import Dict, List, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppgl import REGIONS
from ppgl.labels import n_active_classes
from ppgl.models.backbones import create_backbone


def make_mlp_head(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.3,
) -> nn.Sequential:
    """The head used throughout: Linear -> BN -> LeakyReLU -> Dropout -> Linear.

    Matches the architecture described in the prior work so that the comparison
    isolates the routing mechanism rather than head capacity.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class PrototypeHead(nn.Module):
    """Cosine classifier against learnable class prototypes.

    The final linear layer is replaced by scaled cosine similarity to one
    prototype per class::

        z = normalise(project(features))
        p_c = normalise(prototype_c)
        logit_c = s * <z, p_c>

    Motivation for this dataset specifically: the rarest cluster has 74 slices,
    and a plain linear layer must fit ``hidden_dim`` free weights per class from
    that. Normalising both sides removes the weight-norm degree of freedom, which
    is what lets a linear classifier trade a rare class's margin away for a common
    class's - the standard failure mode behind long-tail bias in the classifier
    layer. Logits are also bounded by ``s``, so the model is far less able to be
    confidently wrong, which matters for the selective-prediction analysis.

    ``s`` is learned in log space and clamped, since a fixed scale is either too
    small to fit or large enough to saturate the softmax.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        init_scale: float = 10.0,
        max_scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.prototypes = nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(float(init_scale))))
        self.max_scale = float(max_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = F.normalize(self.project(features), dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        scale = self.log_scale.exp().clamp(max=self.max_scale)
        return scale * (embedded @ prototypes.t())


def build_head(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.3,
    head_type: str = "mlp",
) -> nn.Module:
    """Classifier-head factory, selected by ``model.head_type``.

    Every image model routes its head through here so that switching to the
    prototype classifier is a config change rather than an edit in four places.
    """
    if head_type == "mlp":
        return make_mlp_head(in_dim, hidden_dim, out_dim, dropout)
    if head_type == "prototype":
        return PrototypeHead(in_dim, hidden_dim, out_dim, dropout)
    raise ValueError(f"unknown head_type {head_type!r}; expected 'mlp' or 'prototype'")


def _head_type(model_cfg) -> str:
    return str(getattr(model_cfg, "head_type", "mlp"))


class SingleHeadClassifier(nn.Module):
    """Backbone + one MLP head.

    Covers the ResNet-101, ViT and Swin baselines, and doubles as the
    ``without region weight estimator`` ablation of the proposed model.
    """

    def __init__(
        self,
        model_cfg,
        n_classes: Optional[int] = None,
        n_regions: int = len(REGIONS),
        predict_region: bool = False,
    ) -> None:
        super().__init__()
        n_classes = n_classes or n_active_classes()
        self.backbone, self.feature_dim, self.mean, self.std = create_backbone(model_cfg)
        self.head = build_head(
            self.feature_dim,
            model_cfg.hidden_dim,
            n_classes,
            model_cfg.dropout,
            head_type=_head_type(model_cfg),
        )
        # Optional auxiliary region head: lets the single-head model be trained
        # with the same auxiliary supervision as the routed model, so the
        # comparison isolates routing and not the extra label.
        self.region_head = nn.Linear(self.feature_dim, n_regions) if predict_region else None

    def trainable_head_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [self.head]
        if self.region_head is not None:
            modules.append(self.region_head)
        return modules

    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del region, force_routing  # unused; keeps the call signature uniform
        features = self.backbone(images)
        out: Dict[str, torch.Tensor] = {"logits": self.head(features), "features": features}
        if self.region_head is not None:
            out["region_logits"] = self.region_head(features)
        return out


class ContrastiveClassifier(SingleHeadClassifier):
    """The ``ViT+CL`` baseline: adds a projection head for supervised contrastive loss.

    Only this baseline uses a contrastive term; the proposed model does not. An
    earlier draft claimed contrastive learning while the method contained none, so
    the claim was dropped rather than retrofitted.
    """

    def __init__(
        self,
        model_cfg,
        n_classes: Optional[int] = None,
        n_regions: int = len(REGIONS),
        projection_dim: int = 128,
        predict_region: bool = False,
    ) -> None:
        super().__init__(
            model_cfg,
            n_classes=n_classes or n_active_classes(),
            n_regions=n_regions,
            predict_region=predict_region,
        )
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, model_cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_cfg.hidden_dim, projection_dim),
        )

    def trainable_head_modules(self) -> List[nn.Module]:
        return super().trainable_head_modules() + [self.projection]

    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(images, region=region, force_routing=force_routing)
        out["embedding"] = self.projection(out["features"])
        return out


class PPGLTransformer(nn.Module):
    """Reimplementation of the two-branch model of Santra et al. (MICCAI AMAI 2023).

    That work predicts the anatomical region in one branch and conditions the
    genetic-cluster branch on it. Reference code was not available, so this is a
    reimplementation from the paper description and must be described as such:
    a shared backbone, a region branch, and a cluster branch that consumes the
    concatenation of pooled features with the region branch's representation.

    It differs from the routed model in a way worth stating precisely: here region
    information is *concatenated as a feature*, and one classifier is applied to all
    anatomy. In the routed
    model, region information *selects among separate expert classifiers*.
    """

    def __init__(
        self,
        model_cfg,
        n_classes: Optional[int] = None,
        n_regions: int = len(REGIONS),
        region_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        n_classes = n_classes or n_active_classes()
        self.backbone, self.feature_dim, self.mean, self.std = create_backbone(model_cfg)

        self.region_branch = nn.Sequential(
            nn.Linear(self.feature_dim, region_embed_dim),
            nn.BatchNorm1d(region_embed_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.region_classifier = nn.Linear(region_embed_dim, n_regions)
        self.cluster_head = build_head(
            self.feature_dim + region_embed_dim,
            model_cfg.hidden_dim,
            n_classes,
            model_cfg.dropout,
            head_type=_head_type(model_cfg),
        )

    def trainable_head_modules(self) -> List[nn.Module]:
        return [self.region_branch, self.region_classifier, self.cluster_head]

    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del region, force_routing
        features = self.backbone(images)
        region_embedding = self.region_branch(features)
        return {
            "logits": self.cluster_head(torch.cat([features, region_embedding], dim=1)),
            "region_logits": self.region_classifier(region_embedding),
            "features": features,
        }


__all__ = [
    "ContrastiveClassifier",
    "PPGLTransformer",
    "PrototypeHead",
    "SingleHeadClassifier",
    "build_head",
    "make_mlp_head",
]
