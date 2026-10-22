"""Shared utilities: logging, seeding, device discovery."""

from ppgl.utils.device import (
    describe_hardware,
    gpu_count,
    resolve_device,
    resolve_parallel_jobs,
)
from ppgl.utils.logging import get_logger, setup_logging
from ppgl.utils.seed import seed_everything, worker_init_fn

__all__ = [
    "describe_hardware",
    "gpu_count",
    "resolve_device",
    "resolve_parallel_jobs",
    "get_logger",
    "setup_logging",
    "seed_everything",
    "worker_init_fn",
]
