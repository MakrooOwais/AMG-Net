"""Optional DistributedDataParallel support.

Fold-level parallelism (see :mod:`ppgl.experiments.scheduler`) is the default
and is the better fit for this workload. DDP is provided for the case where a
single configuration must be trained as fast as possible across all GPUs, e.g.
a final model on the full training set.

Launch with::

    torchrun --nproc_per_node=<N> -m ppgl.cli train --config ... --set runtime.use_ddp=true
"""

from __future__ import annotations

import os
from typing import Any, List, Optional


def is_distributed() -> bool:
    """True when launched under torchrun/torch.distributed."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return global_rank() == 0


def init_process_group() -> None:
    """Initialise the default process group if running distributed."""
    if not is_distributed():
        return
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())


def destroy_process_group() -> None:
    if not is_distributed():
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if not is_distributed():
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def wrap_model(model: Any, device: Any) -> Any:
    """Wrap in DDP when distributed, otherwise return unchanged."""
    if not is_distributed():
        return model
    import torch
    from torch.nn.parallel import DistributedDataParallel

    device_ids = [local_rank()] if torch.cuda.is_available() else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=local_rank() if torch.cuda.is_available() else None,
        # The routed model leaves unused expert parameters when a batch
        # contains no slices from a given region.
        find_unused_parameters=True,
    )


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def gather_object(obj: Any) -> Optional[List[Any]]:
    """All-gather arbitrary picklable objects. Returns None if not distributed."""
    if not is_distributed():
        return None
    import torch.distributed as dist

    gathered: List[Any] = [None for _ in range(world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


__all__ = [
    "barrier",
    "destroy_process_group",
    "gather_object",
    "global_rank",
    "init_process_group",
    "is_distributed",
    "is_main_process",
    "local_rank",
    "unwrap_model",
    "world_size",
    "wrap_model",
]
