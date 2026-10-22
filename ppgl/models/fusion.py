"""Clinical + imaging fusion.

Rationale: on the 4-class task every architecture landed within 0.33-0.41 patient
macro-F1, which says the limit is information, not capacity. Clinical covariates
are a genuinely different information channel rather than a better model of the
same pixels, so this is the most plausible route to a difference large enough to
detect (roughly +0.10 needed for nominal significance at this fold count).

Design: late fusion. The image branch is any backbone from the registry, the
clinical branch is a small MLP, and the concatenated representation feeds one
head. Late fusion is chosen over early or attention-based fusion because with
~180 patients per class in the binary space the fusion layer must stay tiny, and
because it keeps the image branch identical to the ``swin`` baseline so the
comparison isolates the added covariates.

``clinical_dropout`` randomly zeroes the whole clinical vector during training.
That is modality dropout, not ordinary dropout: it forces the image branch to
stay predictive on its own, which matters because the chart review will be
incomplete for some patients and because it prevents the head from ignoring
imaging entirely when the covariates are strong.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ppgl.labels import n_active_classes
from ppgl.models.backbones import create_backbone
from ppgl.models.baselines import _head_type, build_head
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


class ClinicalEncoder(nn.Module):
    """Small MLP over the preprocessed clinical vector."""

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("clinical encoder needs at least one input feature")
        hidden = max(out_dim, min(128, in_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageClinicalFusion(nn.Module):
    """Backbone features concatenated with an encoded clinical vector."""

    def __init__(
        self,
        model_cfg,
        n_clinical_features: int,
        n_classes: Optional[int] = None,
        n_regions: int = 3,
        clinical_dim: int = 64,
        clinical_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        n_classes = n_classes or n_active_classes()
        self.backbone, self.feature_dim, self.mean, self.std = create_backbone(model_cfg)
        self.clinical = ClinicalEncoder(
            n_clinical_features, out_dim=clinical_dim, dropout=model_cfg.dropout
        )
        self.head = build_head(
            self.feature_dim + clinical_dim,
            model_cfg.hidden_dim,
            n_classes,
            model_cfg.dropout,
            head_type=_head_type(model_cfg),
        )
        self.clinical_dropout = float(clinical_dropout)
        self.n_clinical_features = int(n_clinical_features)
        log.info(
            "fusion model: %d image features + %d clinical features -> %d classes",
            self.feature_dim,
            n_clinical_features,
            n_classes,
        )

    def trainable_head_modules(self) -> List[nn.Module]:
        return [self.clinical, self.head]

    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
        clinical: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del region, force_routing

        features = self.backbone(images)

        if clinical is None or clinical.numel() == 0:
            raise ValueError(
                "the fusion model was called without clinical features; "
                "check data.clinical_csv and that the dataset supplies them"
            )
        clinical = clinical.to(features.dtype)
        encoded = self.clinical(clinical)

        # Modality dropout: drop the whole clinical vector for a random subset of
        # the batch so the image branch cannot be bypassed.
        if self.training and self.clinical_dropout > 0:
            keep = (
                torch.rand(encoded.shape[0], 1, device=encoded.device) >= self.clinical_dropout
            ).to(encoded.dtype)
            encoded = encoded * keep

        return {
            "logits": self.head(torch.cat([features, encoded], dim=1)),
            "features": features,
            "clinical_embedding": encoded,
        }


class ClinicalOnlyNet(nn.Module):
    """Clinical covariates alone, no imaging.

    A torch implementation of the clinical-only control, so it trains under the
    identical loop, schedule and loss as the fusion model. The sklearn variant in
    :mod:`ppgl.models.tabular` is the cheaper route when only the baseline number
    is wanted; this one exists so the fusion-minus-clinical difference is not
    confounded by the optimiser.
    """

    def __init__(
        self,
        model_cfg,
        n_clinical_features: int,
        n_classes: Optional[int] = None,
        n_regions: int = 3,
        clinical_dim: int = 64,
    ) -> None:
        super().__init__()
        n_classes = n_classes or n_active_classes()
        self.clinical = ClinicalEncoder(
            n_clinical_features, out_dim=clinical_dim, dropout=model_cfg.dropout
        )
        self.head = build_head(
            clinical_dim,
            model_cfg.hidden_dim,
            n_classes,
            model_cfg.dropout,
            head_type=_head_type(model_cfg),
        )
        # No backbone; exposed as None so the optimiser builder takes the flat path.
        self.backbone = None
        self.feature_dim = clinical_dim
        self.mean = None
        self.std = None

    def trainable_head_modules(self) -> List[nn.Module]:
        return [self.clinical, self.head]

    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
        clinical: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del images, region, force_routing
        if clinical is None or clinical.numel() == 0:
            raise ValueError("clinical_only model requires clinical features")
        encoded = self.clinical(clinical)
        return {"logits": self.head(encoded), "features": encoded}


__all__ = ["ClinicalEncoder", "ClinicalOnlyNet", "ImageClinicalFusion"]
