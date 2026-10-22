"""Device and parallelism discovery.

The remote server's GPU count is unknown at authoring time, so nothing here is
hard-coded. Resolution order for a single training process:

    explicit spec > CUDA (respecting CUDA_VISIBLE_DEVICES) > MPS > CPU

For the full experiment grid the natural unit of parallelism is the
*(model, repeat, fold)* job, not the minibatch: each job trains on ~570 slices
for ~60 epochs, so DDP inside a fold spends more time synchronising than
computing. :func:`resolve_parallel_jobs` therefore defaults to one worker
process per visible GPU and the scheduler pins each worker to its own device.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional


def _torch():
    import torch  # imported lazily so that --help works without torch

    return torch


def gpu_count() -> int:
    """Number of *visible* CUDA devices (0 if CUDA is unavailable)."""
    try:
        torch = _torch()
    except Exception:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def has_mps() -> bool:
    try:
        torch = _torch()
    except Exception:
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(spec: str = "auto", device_index: Optional[int] = None):
    """Return a ``torch.device``.

    Parameters
    ----------
    spec
        ``"auto"``, ``"cuda"``, ``"cuda:2"``, ``"mps"`` or ``"cpu"``.
    device_index
        When given and CUDA is in use, overrides the ordinal. The job scheduler
        uses this to pin worker *k* to GPU *k*.
    """
    torch = _torch()

    if spec and spec != "auto":
        if spec.startswith("cuda") and gpu_count() == 0:
            raise RuntimeError("runtime.device='cuda' requested but no CUDA device is visible")
        if spec == "cuda" and device_index is not None:
            return torch.device("cuda", device_index % max(gpu_count(), 1))
        return torch.device(spec)

    n_gpu = gpu_count()
    if n_gpu > 0:
        return torch.device("cuda", (device_index or 0) % n_gpu)
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_parallel_jobs(requested: int = 0) -> int:
    """How many fold-level worker processes to run concurrently.

    ``requested <= 0`` means auto: one worker per visible GPU, else 1.
    Capped at the number of GPUs when CUDA is present so that two processes
    never contend for the same device.
    """
    n_gpu = gpu_count()
    if requested and requested > 0:
        if n_gpu > 0:
            return min(requested, n_gpu)
        return max(1, requested)
    return n_gpu if n_gpu > 0 else 1


def supports_amp(device) -> bool:
    """Automatic mixed precision is only enabled on CUDA.

    MPS autocast is still unreliable for some timm attention kernels and gives
    no benefit at this batch size, so it is deliberately disabled.
    """
    return getattr(device, "type", str(device)) == "cuda"


def describe_hardware() -> Dict[str, Any]:
    """Snapshot of the execution environment, saved with every run."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": gpu_count(),
        "mps_available": has_mps(),
        "gpus": [],
    }
    try:
        torch = _torch()
        info["torch"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        for idx in range(gpu_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 1),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    except Exception as exc:  # pragma: no cover - diagnostics only
        info["torch_error"] = repr(exc)
    return info


__all__ = [
    "describe_hardware",
    "gpu_count",
    "has_mps",
    "resolve_device",
    "resolve_parallel_jobs",
    "supports_amp",
]
