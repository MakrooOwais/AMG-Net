"""Attention-based multiple-instance learning over a patient's slices.

Gated attention pooling as in Ilse, Tomczak & Welling, "Attention-based Deep
Multiple Instance Learning" (ICML 2018):

    h_k = f(x_k)                                       instance embeddings
    a_k = w^T ( tanh(V h_k) * sigmoid(U h_k) )         gated attention score
    alpha = softmax_k(a_k)                             over valid instances only
    z = sum_k alpha_k h_k                              bag embedding
    y = g(z)                                           bag prediction

Why this is the architectural change worth trying, when routing was not: the
prediction unit becomes the patient, and the model sees up to 22 slices jointly
instead of one at a time with probabilities averaged afterwards. That changes the
information available per decision rather than the capacity used to fit it.

The attention weights are returned and stored, which gives a genuinely
interpretable output for the paper - which slices drove each patient's call -
without the machinery of a saliency method.

Padding: bags in a batch are padded to the longest, and the mask sets padded
attention logits to ``-inf`` before the softmax so they receive exactly zero
weight. Backbone activations on padded positions are computed and discarded; that
wastes a little compute in exchange for a single batched backbone call, which is
much faster than looping over bags.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppgl.labels import n_active_classes
from ppgl.models.backbones import create_backbone
from ppgl.models.baselines import _head_type, build_head
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


class GatedAttentionPool(nn.Module):
    """Gated attention pooling over the instance axis, respecting a validity mask."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        instances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """``instances`` is ``(B, K, D)``, ``mask`` is ``(B, K)`` boolean."""
        scores = self.score(self.value(instances) * self.gate(instances)).squeeze(-1)  # (B, K)

        if mask is not None:
            # Padded positions must receive exactly zero weight.
            scores = scores.masked_fill(~mask, float("-inf"))
            # A bag with no valid instance cannot occur, but guard against a
            # whole row of -inf producing NaN from the softmax.
            empty = ~mask.any(dim=1)
            if bool(empty.any()):
                scores = scores.masked_fill(empty.unsqueeze(1), 0.0)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bk,bkd->bd", weights, instances)
        return {"pooled": pooled, "attention": weights}


class AttentionMIL(nn.Module):
    """Backbone applied per instance, gated attention pooling, one bag-level head."""

    def __init__(
        self,
        model_cfg,
        n_classes: Optional[int] = None,
        n_regions: int = 3,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        n_classes = n_classes or n_active_classes()
        self.backbone, self.feature_dim, self.mean, self.std = create_backbone(model_cfg)
        self.attention = GatedAttentionPool(
            self.feature_dim, hidden_dim=attention_dim, dropout=model_cfg.dropout
        )
        self.head = build_head(
            self.feature_dim,
            model_cfg.hidden_dim,
            n_classes,
            model_cfg.dropout,
            head_type=_head_type(model_cfg),
        )
        # Auxiliary region head on the pooled bag embedding, so the MIL model can
        # be trained with the same auxiliary supervision as single_head_aux and
        # the comparison stays about pooling rather than about the extra label.
        #
        # Opt-out via model.aux_region: region labels are not present in this
        # archive, so a model that reports results should be able to run without
        # any dependence on the provisional estimates.
        self.region_head = (
            nn.Linear(self.feature_dim, n_regions)
            if bool(getattr(model_cfg, "aux_region", True))
            else None
        )

    def trainable_head_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [self.attention, self.head]
        if self.region_head is not None:
            modules.append(self.region_head)
        return modules

    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
        mask: Optional[torch.Tensor] = None,
        clinical: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del region, force_routing, clinical

        if images.dim() == 4:
            # A single slice per patient: treat it as a bag of one so the same
            # model can be evaluated on slice-shaped input.
            images = images.unsqueeze(1)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(1)

        batch, k = images.shape[0], images.shape[1]
        flat = images.reshape(batch * k, *images.shape[2:])
        features = self.backbone(flat).reshape(batch, k, -1)

        if mask is None:
            mask = torch.ones(batch, k, dtype=torch.bool, device=images.device)

        pooled = self.attention(features, mask=mask)

        out: Dict[str, torch.Tensor] = {
            "logits": self.head(pooled["pooled"]),
            "attention": pooled["attention"],
            "features": pooled["pooled"],
            "instance_features": features,
        }
        if self.region_head is not None:
            out["region_logits"] = self.region_head(pooled["pooled"])
        return out


__all__ = ["AttentionMIL", "GatedAttentionPool"]
