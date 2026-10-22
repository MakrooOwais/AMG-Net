"""On-disk results store.

Every ``(experiment, repeat, fold)`` writes its own directory containing the
per-slice predictions, the metrics, and the training history. Nothing is
recomputed for the tables, the statistics or the figures: they all read from
here. That is what makes it possible to trace any number in the paper back to a
specific fold, and it is what lets the paired significance tests run after the
fact rather than requiring all models to be trained in one process.

Layout::

    results/
      manifest.csv                  inventory, source of truth for counts
      data_qa.json                  data-quality report
      fold_summary.csv              per-fold patient and class counts
      runs/<experiment>/
        config.json                 fully-resolved config
        hardware.json               device snapshot
        folds/r0f0/
          metrics.json              metrics per routing mode
          predictions_<mode>.csv    per-slice probabilities and routing telemetry
          history.csv               per-epoch training log
        summary.json                aggregated across folds
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ppgl.metrics import SUMMARY_METRICS, summarise_folds
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

METRICS_FILENAME = "metrics.json"
HISTORY_FILENAME = "history.csv"
DONE_FILENAME = ".done"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def run_dir(output_dir: str, experiment: str) -> str:
    return os.path.join(output_dir, "runs", experiment)


def fold_dir(output_dir: str, experiment: str, repeat: int, fold: int) -> str:
    return os.path.join(run_dir(output_dir, experiment), "folds", f"r{repeat}f{fold}")


def config_fingerprint(cfg) -> str:
    """Hash of the settings that affect results.

    Runtime and bookkeeping fields are excluded so that changing the GPU count or
    re-running with ``resume`` does not invalidate a fold, but anything that
    changes what is computed does.

    This exists to close a specific trap: without it, switching
    ``data.location_source`` from the provisional heuristic to radiologist labels
    would leave every fold looking "already done" and the stale provisional
    results would be silently reused.
    """
    import hashlib

    payload = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    payload = {k: v for k, v in payload.items() if k not in ("output_dir", "experiment")}
    payload.pop("runtime", None)
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def stored_config_fingerprint(output_dir: str, experiment: str, repeat: int, fold: int) -> Optional[str]:
    path = os.path.join(fold_dir(output_dir, experiment, repeat, fold), METRICS_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh).get("config_fingerprint")
    except Exception:
        return None


def is_fold_complete(
    output_dir: str,
    experiment: str,
    repeat: int,
    fold: int,
    expected_fingerprint: Optional[str] = None,
) -> bool:
    """True when the fold has finished *under the same configuration*.

    When ``expected_fingerprint`` is given and the stored fingerprint differs, the
    fold is reported incomplete so it will be recomputed rather than silently
    reused with settings that no longer apply.
    """
    directory = fold_dir(output_dir, experiment, repeat, fold)
    complete = os.path.isfile(os.path.join(directory, DONE_FILENAME)) and os.path.isfile(
        os.path.join(directory, METRICS_FILENAME)
    )
    if not complete:
        return False
    if expected_fingerprint is None:
        return True

    stored = stored_config_fingerprint(output_dir, experiment, repeat, fold)
    if stored is None:
        log.warning(
            "%s/r%df%d has no stored config fingerprint (produced by an older run); recomputing",
            experiment,
            repeat,
            fold,
        )
        return False
    if stored != expected_fingerprint:
        log.info(
            "%s/r%df%d was computed under a different configuration (%s != %s); recomputing",
            experiment,
            repeat,
            fold,
            stored,
            expected_fingerprint,
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def save_fold_result(
    output_dir: str,
    experiment: str,
    repeat: int,
    fold: int,
    metrics_by_mode: Dict[str, Dict[str, Any]],
    predictions_by_mode: Optional[Dict[str, pd.DataFrame]] = None,
    history: Optional[List[Dict[str, float]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist one fold's results and mark it complete."""
    directory = fold_dir(output_dir, experiment, repeat, fold)
    os.makedirs(directory, exist_ok=True)

    payload: Dict[str, Any] = {
        "experiment": experiment,
        "repeat": int(repeat),
        "fold": int(fold),
        "modes": {mode: _jsonable(metrics) for mode, metrics in metrics_by_mode.items()},
    }
    if extra:
        payload.update(_jsonable(extra))

    with open(os.path.join(directory, METRICS_FILENAME), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    for mode, frame in (predictions_by_mode or {}).items():
        frame.to_csv(os.path.join(directory, f"predictions_{mode}.csv"), index=False)

    if history:
        pd.DataFrame(history).to_csv(os.path.join(directory, HISTORY_FILENAME), index=False)

    with open(os.path.join(directory, DONE_FILENAME), "w") as fh:
        fh.write("ok\n")

    return directory


def save_run_metadata(output_dir: str, experiment: str, cfg, hardware: Dict[str, Any]) -> None:
    directory = run_dir(output_dir, experiment)
    os.makedirs(directory, exist_ok=True)
    cfg.save(os.path.join(directory, "config.json"))
    with open(os.path.join(directory, "hardware.json"), "w") as fh:
        json.dump(hardware, fh, indent=2, sort_keys=True, default=str)


def jsonable(value: Any) -> Any:
    """Public alias of :func:`_jsonable`."""
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    """Make numpy types and arrays JSON-serialisable."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def load_fold_metrics(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
) -> pd.DataFrame:
    """One row per fold for the given routing mode.

    Returns an empty frame when the experiment has no completed folds, so that
    aggregation scripts can report a gap rather than crash.
    """
    pattern = os.path.join(run_dir(output_dir, experiment), "folds", "*", METRICS_FILENAME)
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except Exception as exc:
            log.warning("could not read %s (%s)", path, exc)
            continue
        modes = payload.get("modes", {})
        if mode not in modes:
            continue
        row: Dict[str, Any] = {
            "experiment": payload.get("experiment", experiment),
            "repeat": payload.get("repeat"),
            "fold": payload.get("fold"),
            "mode": mode,
        }
        for key, value in modes[mode].items():
            # Confusion matrices are nested lists; keep them out of the table.
            if isinstance(value, (int, float, str)) or value is None:
                row[key] = value
        for key in ("fold_fingerprint", "n_train_slices", "n_test_slices", "n_train_patients"):
            if key in payload:
                row[key] = payload[key]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["repeat", "fold"]).reset_index(drop=True)


def available_modes(output_dir: str, experiment: str) -> List[str]:
    pattern = os.path.join(run_dir(output_dir, experiment), "folds", "*", METRICS_FILENAME)
    modes: set = set()
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fh:
                modes.update(json.load(fh).get("modes", {}).keys())
        except Exception:
            continue
    return sorted(modes)


def list_experiments(output_dir: str) -> List[str]:
    root = os.path.join(output_dir, "runs")
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name, "folds"))
    )


def load_predictions(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
) -> pd.DataFrame:
    """Concatenate per-slice predictions across folds, tagged with repeat/fold."""
    pattern = os.path.join(
        run_dir(output_dir, experiment), "folds", "*", f"predictions_{mode}.csv"
    )
    frames: List[pd.DataFrame] = []
    for path in sorted(glob.glob(pattern)):
        tag = os.path.basename(os.path.dirname(path))
        frame = pd.read_csv(path)
        frame["fold_tag"] = tag
        repeat, _, fold = tag.partition("f")
        frame["repeat"] = int(repeat.lstrip("r") or 0)
        frame["fold"] = int(fold or 0)
        frame["experiment"] = experiment
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise_experiment(
    output_dir: str,
    experiment: str,
    mode: str = "soft",
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> Dict[str, Any]:
    """Mean/std across folds, written to ``summary.json``."""
    table = load_fold_metrics(output_dir, experiment, mode=mode)
    if table.empty:
        return {"experiment": experiment, "mode": mode, "n_folds": 0}

    records = table.to_dict(orient="records")
    summary = summarise_folds(records, metrics=metrics)
    summary["experiment"] = experiment
    summary["mode"] = mode

    path = os.path.join(run_dir(output_dir, experiment), "summary.json")
    existing: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}
    existing[mode] = summary
    with open(path, "w") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    return summary


def load_config_dict(output_dir: str, experiment: str) -> Dict[str, Any]:
    path = os.path.join(run_dir(output_dir, experiment), "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


__all__ = [
    "available_modes",
    "config_fingerprint",
    "fold_dir",
    "is_fold_complete",
    "stored_config_fingerprint",
    "jsonable",
    "list_experiments",
    "load_config_dict",
    "load_fold_metrics",
    "load_predictions",
    "run_dir",
    "save_fold_result",
    "save_run_metadata",
    "summarise_experiment",
]
