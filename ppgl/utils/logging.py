"""Logging setup.

Worker processes in the fold scheduler prefix their records with a job tag so
that interleaved output from N GPUs stays readable, and each job additionally
writes its own log file next to its results.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False
_DEFAULT_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    job_tag: Optional[str] = None,
    force: bool = False,
) -> None:
    """Configure the root logger once per process."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        if log_file:
            _attach_file_handler(root, log_file)
        return

    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = _DEFAULT_FMT
    if job_tag:
        fmt = f"%(asctime)s %(levelname)-7s [{job_tag}] %(name)-22s %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if log_file:
        _attach_file_handler(root, log_file, formatter)

    # timm and PIL are chatty at DEBUG.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True


def _attach_file_handler(
    root: logging.Logger,
    log_file: str,
    formatter: Optional[logging.Formatter] = None,
) -> None:
    resolved = os.path.abspath(log_file)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(getattr(handler, "baseFilename", "")) == resolved:
                return
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    file_handler = logging.FileHandler(resolved)
    file_handler.setFormatter(formatter or logging.Formatter(_DEFAULT_FMT, datefmt="%H:%M:%S"))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
