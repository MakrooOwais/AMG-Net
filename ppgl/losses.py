"""Loss functions.

An earlier version of this work carried a malformed loss equation: a stray leading
``1 -``, the cross-entropy term had the wrong sign structure, the Tversky index
was summed over samples in a segmentation formulation applied to classification,
and alpha/beta were never stated. The formulation here is written out explicitly.

Total objective::

    L = w_ce * CE(y_hat, y)
      + w_tv * Tversky(y_hat, y; alpha, beta)
      + lambda_region * CE(region_hat, region)
      + consistency_weight * RouterConsistency          (routing="consistency")

Multi-class Tversky, computed per class over the batch and macro-averaged::

    TP_c = sum_i p_ic * y_ic
    FN_c = sum_i y_ic * (1 - p_ic)
    FP_c = sum_i (1 - y_ic) * p_ic
    TI_c = (TP_c + s) / (TP_c + alpha * FN_c + beta * FP_c + s)
    L_tv = 1 - mean_c TI_c

``alpha > beta`` penalises false negatives more heavily, which is the intended
behaviour for the rare VHL/EPAS1 and kinase-signalling clusters. Defaults are
alpha=0.7, beta=0.3 and both are logged with every run.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Class weighting
# --------------------------------------------------------------------------- #
def inverse_frequency_weights(
    labels: Sequence[int],
    n_classes: int,
    normalise: bool = True,
) -> torch.Tensor:
    """Weights inversely proportional to class frequency.

    Classes absent from ``labels`` get weight 1.0 rather than infinity.
    """
    counts = np.bincount(np.asarray(labels, dtype=int).reshape(-1), minlength=n_classes).astype(
        np.float64
    )
    weights = np.ones(n_classes, dtype=np.float64)
    positive = counts > 0
    weights[positive] = counts[positive].sum() / (counts[positive] * positive.sum())
    if normalise:
        weights = weights * n_classes / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Tversky
# --------------------------------------------------------------------------- #
class LogitAdjustedCrossEntropy(nn.Module):
    """Cross-entropy with a class-prior offset added to the logits.

    From Menon et al., "Long-tail learning via logit adjustment" (ICLR 2021):

        loss = CE(logits + tau * log(prior), y)

    ``tau = 1`` recovers balanced softmax. This is better founded for a long-tail
    target than re-weighting the loss, because it corrects the decision rule
    rather than the gradient magnitude, and it leaves the model calibrated for the
    balanced distribution at test time.

    It **overlaps** with inverse-frequency class weighting - using both
    double-corrects the imbalance. The trainer warns when both are active.
    """

    def __init__(
        self,
        class_prior: torch.Tensor,
        tau: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        prior = class_prior.clamp_min(1e-12)
        prior = prior / prior.sum()
        self.register_buffer("log_prior", prior.log())
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_prior.to(logits.device, logits.dtype)
        return F.cross_entropy(adjusted, target, label_smoothing=self.label_smoothing)


class AttentionLocalisationLoss(nn.Module):
    """Penalty on class-token attention that falls outside the lesion.

        L = -log( sum_p  alpha_p * m_p )

    where ``alpha`` is the class token's attention over patches (a distribution,
    summing to one) and ``m_p`` in [0, 1] is the fraction of patch ``p`` covered by
    the lesion box. The loss is zero when all attention mass sits on fully covered
    patches and grows without bound as mass leaks outside.

    Why this form rather than Dice or MSE against the mask, which is what the
    original prototype used: ``alpha`` is softmax-normalised over ~196 patches, so
    its entries are around 0.005, while the mask is 0/1. Dice's numerator is then
    ~2*0.005*k against a denominator of ~1+k, so the index is bounded near zero and
    the loss sits at ~1 almost regardless of where the model looks - it cannot
    express "correct". MSE has the mirror problem: the minimum is at
    ``alpha_p = m_p``, which the simplex constraint forbids, so the gradient never
    vanishes even for perfect localisation. Working with attention *mass* keeps
    both sides on the same scale and makes the optimum reachable.

    ``target`` caps the demand. With ``target = tau < 1`` the loss becomes

        L = -log( min( mass / tau, 1 ) )

    which is zero as soon as ``tau`` of the mass is inside and non-zero below it.
    That matters here because the annotation is a bounding *box*: for a roughly
    round lesion about a fifth of the box is background, and peritumoral tissue
    outside it may carry real signal. Demanding all of the mass inside demands
    attention on tissue that is not tumour, and forbids the model from using
    context. Measured on this cohort, ``target = 1`` drives the mass to 98%.

    Attention may arrive per-head, ``(B, heads, P)``. Then the mass is computed
    per head and the loss uses the **best** head, so only one head has to localise
    and the others stay free to attend globally.

    Rows whose mask is empty contribute nothing, rather than contributing an
    arbitrary large penalty.
    """

    def __init__(self, target: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0.0 < float(target) <= 1.0:
            raise ValueError(f"target must be in (0, 1], got {target}")
        self.target = float(target)
        self.eps = float(eps)

    def forward(self, attention: torch.Tensor, lesion_mask: torch.Tensor) -> torch.Tensor:
        if attention.dim() not in (2, 3):
            raise ValueError(
                f"attention must be (B, P) or (B, heads, P), got {tuple(attention.shape)}"
            )

        n_patches = attention.shape[-1]
        side = int(round(float(n_patches) ** 0.5))
        if side * side != n_patches:
            raise ValueError(f"expected a square patch grid, got {n_patches} patches")

        mask = lesion_mask
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(f"lesion_mask must be (B, 1, H, W), got {tuple(mask.shape)}")

        # Average-pool to the patch grid: each value becomes the covered fraction
        # of that patch, which is the right target for a mass-based objective.
        pooled = F.adaptive_avg_pool2d(mask.to(attention.dtype), (side, side))
        target = pooled.flatten(1)

        if attention.dim() == 3:
            # (B, heads, P) against (B, P): mass per head, then the best head.
            inside = (attention * target.unsqueeze(1)).sum(dim=-1).max(dim=1).values
        else:
            inside = (attention * target).sum(dim=-1)

        has_lesion = target.sum(dim=1) > 0
        if not bool(has_lesion.any()):
            return attention.sum() * 0.0

        satisfied = inside[has_lesion] / self.target
        return -(satisfied.clamp(min=self.eps, max=1.0).log()).mean()


class MulticlassTverskyLoss(nn.Module):
    """Macro-averaged multi-class Tversky loss over the batch."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("Tversky alpha and beta must be non-negative")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else torch.empty(0)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), num_classes=n_classes).to(probs.dtype)

        tp = (probs * onehot).sum(dim=0)
        fn = (onehot * (1.0 - probs)).sum(dim=0)
        fp = ((1.0 - onehot) * probs).sum(dim=0)

        index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        if self.class_weights.numel() == n_classes:
            weights = self.class_weights.to(index.device, index.dtype)
            # Only average over classes present in this batch, so an absent
            # class does not contribute a spurious perfect score.
            present = (onehot.sum(dim=0) > 0).to(index.dtype)
            denom = (weights * present).sum().clamp_min(1e-8)
            return 1.0 - (index * weights * present).sum() / denom

        present = (onehot.sum(dim=0) > 0).to(index.dtype)
        denom = present.sum().clamp_min(1.0)
        return 1.0 - (index * present).sum() / denom


# --------------------------------------------------------------------------- #
# Supervised contrastive (for the ViT+CL baseline only)
# --------------------------------------------------------------------------- #
class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., NeurIPS 2020).

    Present solely to reproduce the ``vit_cl`` baseline from the prior work. The
    proposed model does **not** use a contrastive term - an earlier draft
    claimed one in its abstract, highlights and contributions while the method
    contained none, and that claim has been removed rather than retrofitted.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        batch = features.shape[0]
        if batch < 2:
            return features.sum() * 0.0

        similarity = features @ features.t() / self.temperature
        # Numerical stability: subtract the row max before exponentiating.
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).to(features.dtype)
        self_mask = torch.eye(batch, device=features.device, dtype=features.dtype)
        positive_mask = positive_mask - positive_mask * self_mask

        exp_sim = torch.exp(similarity) * (1.0 - self_mask)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not valid.any():
            return features.sum() * 0.0

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_counts[valid]
        return -mean_log_prob_pos.mean()


# --------------------------------------------------------------------------- #
# Router regularisation
# --------------------------------------------------------------------------- #
class RouterConsistencyLoss(nn.Module):
    """Penalty encouraging a confident, well-utilised router.

    Two terms:

    ``entropy``
        Mean per-sample entropy of the routing distribution. Lowering it sharpens
        routing, which reduces the gap between the soft weights used at inference
        and the near-one-hot weights an expert was effectively trained on.
    ``balance``
        Squared deviation of mean expert usage from the empirical region prior.
        Head & neck is 64 slices and chest 80, so left unconstrained the router
        collapses onto the abdomen expert. Standard MoE load balancing targets a
        uniform prior; here the target is the *observed* region distribution,
        because the anatomical imbalance is real and should not be equalised away.
    """

    def __init__(
        self,
        entropy_weight: float = 1.0,
        balance_weight: float = 1.0,
        target_prior: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.entropy_weight = float(entropy_weight)
        self.balance_weight = float(balance_weight)
        self.register_buffer(
            "target_prior", target_prior if target_prior is not None else torch.empty(0)
        )

    def forward(self, router_probs: torch.Tensor) -> torch.Tensor:
        probs = router_probs.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=1).mean()
        total = self.entropy_weight * entropy

        usage = probs.mean(dim=0)
        if self.target_prior.numel() == usage.numel():
            target = self.target_prior.to(usage.device, usage.dtype)
        else:
            target = torch.full_like(usage, 1.0 / usage.numel())
        total = total + self.balance_weight * ((usage - target) ** 2).sum()
        return total


# --------------------------------------------------------------------------- #
# Composite objective
# --------------------------------------------------------------------------- #
class PPGLLoss(nn.Module):
    """The full training objective, returning each component for logging."""

    def __init__(
        self,
        loss_cfg,
        n_classes: int,
        n_regions: int,
        class_weights: Optional[torch.Tensor] = None,
        region_prior: Optional[torch.Tensor] = None,
        class_prior: Optional[torch.Tensor] = None,
        consistency_weight: float = 0.0,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.cfg = loss_cfg
        self.n_classes = int(n_classes)
        self.n_regions = int(n_regions)
        self.consistency_weight = float(consistency_weight)
        self.contrastive_weight = float(contrastive_weight)

        label_smoothing = float(getattr(loss_cfg, "label_smoothing", 0.0))
        tau = float(getattr(loss_cfg, "logit_adjust_tau", 0.0))
        self.logit_adjust_tau = tau
        if tau > 0 and class_prior is not None:
            # Deliberately unweighted: logit adjustment already corrects the
            # class prior, and applying inverse-frequency weights on top would
            # double-correct it. The trainer warns when both were requested.
            self.ce = LogitAdjustedCrossEntropy(
                class_prior, tau=tau, label_smoothing=label_smoothing
            )
        else:
            if tau > 0:
                raise ValueError(
                    "loss.logit_adjust_tau > 0 but no class prior was supplied; "
                    "the trainer must pass train labels for the prior to be estimated"
                )
            self.ce = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        # The auxiliary region head is unweighted: region imbalance is a real
        # property of the anatomy, not a nuisance to be corrected.
        self.region_ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.tversky = MulticlassTverskyLoss(
            alpha=float(loss_cfg.tversky_alpha),
            beta=float(loss_cfg.tversky_beta),
            smooth=float(loss_cfg.tversky_smooth),
            class_weights=class_weights,
        )
        self.router_consistency = RouterConsistencyLoss(target_prior=region_prior)
        self.supcon = SupConLoss(temperature=contrastive_temperature)
        self.attention_localisation = AttentionLocalisationLoss(
            target=float(getattr(loss_cfg, "attention_target", 1.0))
        )
        # Position in training, in [0, 1]. Only the attention warm-down reads it;
        # the trainer sets it once per epoch, mirroring the router's set_progress.
        self._progress = 0.0

    def set_progress(self, progress: float) -> None:
        """Record training position so a scheduled term can decay."""
        self._progress = float(min(max(progress, 0.0), 1.0))

    def _attention_weight(self) -> float:
        """Current lambda for the attention term, after any warm-down.

        With ``attention_warmup_frac = f > 0`` the weight falls linearly to zero
        over the first ``f`` of training and is zero afterwards, so the run becomes
        attention-supervised pretraining followed by plain cross-entropy - the
        two-stage schedule of Di Stefano et al. (2026). With ``f = 0`` the weight
        is constant, which is the joint-training default.
        """
        base = float(getattr(self.cfg, "lambda_attention", 0.0))
        fraction = float(getattr(self.cfg, "attention_warmup_frac", 0.0))
        if base <= 0 or fraction <= 0:
            return base
        if self._progress >= fraction:
            return 0.0
        return base * (1.0 - self._progress / fraction)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        target = batch["cluster"]

        components: Dict[str, torch.Tensor] = {}
        total = logits.sum() * 0.0  # dtype/device-matched zero

        if self.cfg.ce_weight:
            ce = self.ce(logits, target)
            components["ce"] = ce.detach()
            total = total + float(self.cfg.ce_weight) * ce

        if self.cfg.tversky_weight:
            tversky = self.tversky(logits, target)
            components["tversky"] = tversky.detach()
            total = total + float(self.cfg.tversky_weight) * tversky

        region_logits = outputs.get("region_logits")
        if region_logits is not None and float(self.cfg.lambda_region) > 0:
            region_target = batch.get("region")
            if region_target is not None and (region_target >= 0).any():
                region_loss = self.region_ce(region_logits, region_target)
                components["region_ce"] = region_loss.detach()
                total = total + float(self.cfg.lambda_region) * region_loss

        router_probs = outputs.get("router_probs")
        if router_probs is not None and self.consistency_weight > 0:
            consistency = self.router_consistency(router_probs)
            components["router_consistency"] = consistency.detach()
            total = total + self.consistency_weight * consistency

        embedding = outputs.get("embedding")
        if embedding is not None and self.contrastive_weight > 0:
            contrastive = self.supcon(embedding, target)
            components["supcon"] = contrastive.detach()
            total = total + self.contrastive_weight * contrastive

        attention = outputs.get("cls_attention")
        lesion_mask = batch.get("lesion_mask")
        lambda_attention = self._attention_weight()
        if attention is not None and lesion_mask is not None and lambda_attention > 0:
            localisation = self.attention_localisation(attention, lesion_mask)
            components["attention_loc"] = localisation.detach()
            total = total + lambda_attention * localisation

        components["total"] = total
        return components


def class_prior_from_labels(labels: Sequence[int], n_classes: int) -> torch.Tensor:
    """Empirical class distribution, for logit adjustment.

    Absent classes get a floor of one pseudo-count rather than zero, because
    ``log(0)`` would drive their adjusted logit to ``-inf`` and make the class
    unpredictable for the whole fold.
    """
    counts = np.bincount(np.asarray(labels, dtype=int).reshape(-1), minlength=n_classes).astype(
        np.float64
    )
    counts = np.maximum(counts, 1.0)
    return torch.tensor(counts / counts.sum(), dtype=torch.float32)


__all__ = [
    "AttentionLocalisationLoss",
    "LogitAdjustedCrossEntropy",
    "MulticlassTverskyLoss",
    "PPGLLoss",
    "RouterConsistencyLoss",
    "SupConLoss",
    "class_prior_from_labels",
    "inverse_frequency_weights",
]
