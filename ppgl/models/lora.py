"""Minimal LoRA for ``nn.Linear``, used as the parameter-efficient fine-tuning arm.

Motivation: at n=630 slices, full fine-tuning of a Swin backbone is close to the
data-starved regime where LoRA has been reported to match or beat full
fine-tuning, with the advantage growing as labelled data shrinks (Lian et al.,
"Less could be better", 2024). This is implemented as an ablation axis rather
than a default so the comparison can be made on our own folds.

Only ``nn.Linear`` layers whose qualified name contains one of the target
substrings are adapted; in timm attention blocks that is ``qkv`` for ViT/Swin and
``q_proj``/``v_proj`` for models that split the projections.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """``y = base(x) + (alpha / r) * B(A(x))`` with ``base`` frozen."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_a = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        # A ~ Kaiming, B = 0, so the adapter starts as an exact no-op.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = torch.nn.functional.linear(x, self.lora_a)
        update = torch.nn.functional.linear(update, self.lora_b)
        return out + self.scaling * update

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def inject_lora(
    module: nn.Module,
    targets: Sequence[str] = ("qkv", "q_proj", "v_proj"),
    r: int = 8,
    alpha: float = 16.0,
) -> Tuple[int, List[str]]:
    """Replace matching ``nn.Linear`` layers in-place with :class:`LoRALinear`.

    Returns the number of layers adapted and their names. Call *after* freezing
    the backbone: the adapters are created with ``requires_grad=True`` and the
    wrapped base weights with ``requires_grad=False``.
    """
    replaced: List[str] = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(token in name for token in targets):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            replaced.append(name)
        else:
            count, names = inject_lora(child, targets=targets, r=r, alpha=alpha)
            replaced.extend(f"{name}.{sub}" for sub in names)
    return len(replaced), replaced


def mark_only_lora_trainable(module: nn.Module) -> None:
    """Freeze everything except LoRA parameters."""
    for name, param in module.named_parameters():
        param.requires_grad = "lora_a" in name or "lora_b" in name


def trainable_parameter_report(module: nn.Module) -> dict:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": round(trainable / total, 6) if total else 0.0,
    }


def iter_lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in module.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            yield param


__all__ = [
    "LoRALinear",
    "inject_lora",
    "iter_lora_parameters",
    "mark_only_lora_trainable",
    "trainable_parameter_report",
]
