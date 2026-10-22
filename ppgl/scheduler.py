"""Learning-rate schedules.

Linear warmup followed by cosine decay, stepped per optimiser step rather than
per epoch, which matters here because a fold has only ~36 steps per epoch.

Each parameter group keeps its own base LR, so this composes with the layer-wise
LR decay applied in :func:`ppgl.models.backbones.layer_wise_param_groups`.
"""

from __future__ import annotations

import math
from typing import List


class WarmupCosineSchedule:
    """Warmup then cosine decay to ``base_lr * min_lr_factor``."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_factor: float = 0.01,
        last_step: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), self.warmup_steps + 1)
        self.min_lr_factor = float(min_lr_factor)
        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]
        self._step = int(last_step)
        if self._step < 0:
            self.step(0)

    def _factor(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            # Start at 1/warmup_steps rather than 0 so the first step still moves.
            return float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_factor + (1.0 - self.min_lr_factor) * cosine

    def step(self, step: int = None) -> None:
        if step is None:
            self._step += 1
        else:
            self._step = int(step)
        factor = self._factor(self._step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"step": self._step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict) -> None:
        self._step = int(state.get("step", 0))
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))


class ConstantSchedule:
    """No-op schedule, for reproducing a fixed-LR run."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, step: int = None) -> None:  # noqa: D401 - deliberately inert
        return

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return


def build_scheduler(optimizer, optim_cfg, steps_per_epoch: int):
    """Construct the schedule named by ``optim_cfg.scheduler``."""
    steps_per_epoch = max(int(steps_per_epoch), 1)
    if optim_cfg.scheduler in ("warmup_cosine", "cosine"):
        return WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(optim_cfg.warmup_epochs) * steps_per_epoch,
            total_steps=int(optim_cfg.epochs) * steps_per_epoch,
            min_lr_factor=float(optim_cfg.min_lr_factor),
        )
    if optim_cfg.scheduler in ("none", "constant"):
        return ConstantSchedule(optimizer)
    raise ValueError(f"unknown scheduler {optim_cfg.scheduler!r}")


__all__ = ["ConstantSchedule", "WarmupCosineSchedule", "build_scheduler"]
