"""Deterministic seeding.

Fold results feed paired significance tests, so run-to-run variation must come
from the data split and not from uncontrolled RNG state. Each
*(model, repeat, fold)* job derives its seed from the base seed plus the repeat
and fold indices, so a rerun of one fold reproduces exactly.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN autotuning picks different algorithms per run; disable it.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some timm ops have no deterministic CUDA kernel. warn_only keeps the
        # run alive instead of aborting mid-grid.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def job_seed(base_seed: int, repeat: int, fold: int, extra: int = 0) -> int:
    """Stable per-job seed. Distinct (repeat, fold) pairs never collide."""
    return int((base_seed * 1_000_003 + repeat * 10_007 + fold * 101 + extra) % (2**31 - 1))


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding, so augmentation differs per worker but is reproducible."""
    try:
        import torch

        base = torch.initial_seed() % (2**31 - 1)
    except Exception:
        base = worker_id
    seed = (base + worker_id) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))


def make_generator(seed: int) -> Optional[object]:
    """A torch.Generator for DataLoader shuffling, or None if torch is absent."""
    try:
        import torch
    except Exception:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


__all__ = ["seed_everything", "job_seed", "worker_init_fn", "make_generator"]
