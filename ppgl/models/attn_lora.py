"""ViT with LoRA adapters and class-token attention supervised by the lesion box.

The idea being tested: at 630 slices the model has enough capacity to fit the
data several times over, so the useful thing to add is not capacity but a
constraint on *where it looks*. The archive's XML annotations give a lesion
bounding box for every slice, and the existing pipeline spends that information
only on cropping. Here it is also used as a supervision target on the encoder's
attention, which is a genuinely different signal from anything else in the grid:
every other row is free to key off whatever background texture happens to
correlate with class in this cohort.

Mechanically:

* the encoder is a timm ViT, frozen, with LoRA adapters in the attention and MLP
  projections (``model.finetune: lora``), so only ~1% of parameters train;
* the class token's attention over patches in the final block is pulled out and
  compared against the lesion footprint pooled to the patch grid;
* classification is a head on the pooled features, exactly as in the other rows.

Extracting the attention needs one non-obvious step. timm's ``Attention`` uses
``F.scaled_dot_product_attention`` whenever ``fused_attn`` is set, and that kernel
never materialises the attention matrix, so no hook can observe it. Setting
``fused_attn = False`` on the final block restores the explicit
``softmax -> attn_drop -> @v`` path, and a forward hook on ``attn_drop`` then sees
the post-softmax weights as its input. Only the last block is switched, so the
rest of the encoder keeps the fast kernel.

Provenance: adapted from a prototype (``CT_LoRA``) that reported patient-level
macro-F1 around 0.64 and AUC around 0.89 on this cohort. Those numbers are not
comparable to anything here - that prototype split on slice index rather than
patient, which on this cohort leaves 73.5% of every held-out fold sharing a
patient with training, and it selected the best of 100 epochs on the same fold it
then reported. The mechanism is worth testing; the numbers were not evidence.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ppgl import REGIONS
from ppgl.labels import n_active_classes
from ppgl.models.backbones import create_backbone
from ppgl.models.baselines import _head_type, build_head
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


class AttentionSupervisedViT(nn.Module):
    """ViT encoder whose class-token attention is supervised toward the lesion."""

    def __init__(
        self,
        model_cfg,
        n_classes: Optional[int] = None,
        n_regions: int = len(REGIONS),
        predict_region: bool = True,
    ) -> None:
        super().__init__()
        n_classes = n_classes or n_active_classes()
        self.backbone, self.feature_dim, self.mean, self.std = create_backbone(model_cfg)

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError(
                f"model.backbone={model_cfg.backbone!r} has no transformer blocks, so its "
                "attention cannot be supervised. Use a ViT-family backbone "
                "(vit_small_patch16_224, vit_base_patch16_224, ...)."
            )

        self.attention_source = str(getattr(model_cfg, "attention_source", "last"))
        self.attention_heads = str(getattr(model_cfg, "attention_heads", "mean"))
        n_rollout = int(getattr(model_cfg, "attention_rollout_layers", 0))

        # Which blocks to observe. "last" needs one; rollout needs the whole path
        # it claims to trace, or the trailing N blocks if that is capped.
        if self.attention_source == "rollout":
            watched = list(blocks) if n_rollout <= 0 else list(blocks)[-n_rollout:]
        else:
            watched = [blocks[-1]]

        # Ordered store, one slot per watched block, so rollout can multiply them
        # in depth order regardless of hook firing order.
        self._captured: List[Optional[torch.Tensor]] = [None] * len(watched)
        self._watched_count = len(watched)

        for position, block in enumerate(watched):
            attention = getattr(block, "attn", None)
            if attention is None or not hasattr(attention, "attn_drop"):
                raise ValueError(
                    f"a block of {model_cfg.backbone!r} does not expose an attn_drop module; "
                    "attention supervision needs timm's standard Attention layer."
                )
            # timm routes through F.scaled_dot_product_attention when fused_attn is
            # set, and that kernel never materialises the attention matrix, so no
            # hook can observe it. Only the watched blocks are switched to the
            # explicit path; the rest keep the fast kernel.
            if hasattr(attention, "fused_attn"):
                attention.fused_attn = False
            attention.attn_drop.register_forward_hook(self._make_hook(position))

        # Number of leading non-patch tokens (class token, and register tokens on
        # some checkpoints). Read from the model rather than assumed to be 1.
        self.n_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        self.head = build_head(
            self.feature_dim,
            model_cfg.hidden_dim,
            n_classes,
            model_cfg.dropout,
            head_type=_head_type(model_cfg),
        )
        # Auxiliary region head, so this row carries the same extra supervision as
        # single_head_aux and the comparison stays about attention supervision.
        self.region_head = nn.Linear(self.feature_dim, n_regions) if predict_region else None

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(
            "attention-supervised ViT: source=%s heads=%s watching %d block(s), "
            "%d prefix token(s), %d/%d trainable params (%.1f%%)",
            self.attention_source,
            self.attention_heads,
            self._watched_count,
            self.n_prefix_tokens,
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )

    # ----------------------------------------------------------------- hooks
    def _make_hook(self, position: int):
        """Hook storing this block's post-softmax attention at a fixed slot."""

        def hook(module, inputs, output) -> None:
            self._captured[position] = inputs[0] if inputs else None

        return hook

    def _fuse_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """Element-wise head fusion, ``(B, heads, N, N)`` -> ``(B, N, N)``.

        Used inside rollout, where a single matrix per block is needed before the
        matrices can be multiplied. Mean and max fusion are both standard here
        (Abnar & Zuidema, ACL 2020).
        """
        if self.attention_heads == "max":
            return attention.max(dim=1).values
        return attention.mean(dim=1)

    def trainable_head_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [self.head]
        if self.region_head is not None:
            modules.append(self.region_head)
        return modules

    # --------------------------------------------------------------- helpers
    def _patch_grid(self, n_patches: int) -> Tuple[int, int]:
        grid = getattr(getattr(self.backbone, "patch_embed", None), "grid_size", None)
        if grid is not None and int(grid[0]) * int(grid[1]) == n_patches:
            return int(grid[0]), int(grid[1])
        side = int(round(math.sqrt(n_patches)))
        if side * side != n_patches:
            raise ValueError(
                f"cannot infer a square patch grid from {n_patches} patches; "
                "attention supervision assumes a square input"
            )
        return side, side

    def _take_captured(self) -> List[torch.Tensor]:
        """Consume the stored attentions, in block order, and reset the slots."""
        captured = [c for c in self._captured if c is not None and c.dim() == 4]
        self._captured = [None] * self._watched_count
        return captured

    def _class_attention(self) -> Optional[torch.Tensor]:
        """Class-token attention over patches, renormalised to sum to one.

        Returns ``(B, P)``, or ``(B, heads, P)`` when supervising the final layer
        with ``attention_heads="max"`` - in that case the loss reduces over heads
        by requiring only the best-localising head to satisfy the target, which
        leaves the remaining heads free to attend globally.
        """
        captured = self._take_captured()
        if not captured:
            return None

        if self.attention_source == "rollout":
            attention = self._rollout(captured)
        else:
            # Final block only; the hook order guarantees it is the last slot.
            attention = captured[-1][:, :, 0, self.n_prefix_tokens:]
            if self.attention_heads == "mean":
                attention = attention.mean(dim=1)

        if attention is None or attention.shape[-1] == 0:
            return None
        # Slicing away the prefix tokens (and, for rollout, the residual mass on
        # the class token itself) breaks the simplex, so renormalise: the loss is
        # defined on a *fraction* of attention mass.
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _rollout(self, captured: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Attention rollout over the watched blocks (Abnar & Zuidema, ACL 2020).

        Raw single-layer attention is a weak proxy for how much a patch actually
        influences the class token, because the residual stream carries information
        around each attention operation. Rollout accounts for that by adding the
        identity before multiplying:

            A_hat_l = rownorm(0.5 * A_l + 0.5 * I)
            R       = A_hat_L @ ... @ A_hat_1

        The class-token row of ``R`` then estimates dependence on each input patch.
        """
        fused = [self._fuse_heads(a) for a in captured]
        n_tokens = fused[0].shape[-1]
        eye = torch.eye(n_tokens, device=fused[0].device, dtype=fused[0].dtype)

        rolled: Optional[torch.Tensor] = None
        for matrix in fused:  # shallow -> deep
            residual = 0.5 * matrix + 0.5 * eye
            residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rolled = residual if rolled is None else residual @ rolled
        if rolled is None:
            return None
        return rolled[:, 0, self.n_prefix_tokens:]

    # --------------------------------------------------------------- forward
    def forward(
        self,
        images: torch.Tensor,
        region: Optional[torch.Tensor] = None,
        force_routing: Optional[str] = None,
        lesion_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del region, force_routing, lesion_mask  # the loss consumes the mask, not the model

        features = self.backbone(images)
        out: Dict[str, torch.Tensor] = {
            "logits": self.head(features),
            "features": features,
        }
        if self.region_head is not None:
            out["region_logits"] = self.region_head(features)

        attention = self._class_attention()
        if attention is not None:
            out["cls_attention"] = attention
            # A (B, H, W) map for figures and for the stored telemetry. With
            # per-head attention, average the heads purely for visualisation - the
            # loss still sees every head.
            flat = attention.mean(dim=1) if attention.dim() == 3 else attention
            height, width = self._patch_grid(flat.shape[-1])
            out["attention_map"] = flat.reshape(flat.shape[0], height, width)
        return out


__all__ = ["AttentionSupervisedViT"]
