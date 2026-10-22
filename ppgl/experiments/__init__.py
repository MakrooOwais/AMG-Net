"""Experiment orchestration: runners, dispatcher, analysis, tables, figures.

Submodules are imported lazily. ``run_cv``, ``figures`` and the model registry
pull in torch and matplotlib, whereas ``data_prep``, ``results``, ``stats`` and
``run_tabular`` need only numpy, pandas, scipy and scikit-learn. Deferring the
imports keeps the location-only decision gate and all post-hoc analysis runnable
on a machine without a GPU stack.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = (
    "data_prep",
    "dispatch",
    "figures",
    "results",
    "routing_analysis",
    "run_cv",
    "run_tabular",
    "stats",
    "tables",
)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(list(globals().keys()) + list(_SUBMODULES))


__all__ = list(_SUBMODULES)
