"""Post-hoc ensembling of experiments that have already been run.

This is the cheapest remaining source of accuracy in the grid: every experiment
already stored per-slice probabilities for all 50 folds, so combining them costs
no GPU time at all. The combination is written back into the results store as a
new pseudo-experiment, which means the existing tables, figures and paired
significance tests pick it up with no changes - an ensemble is just another row.

Correctness gates, in the order they are checked, because an ensemble that
silently averages misaligned rows would look like a free win:

1. **Same splits.** Every member must carry the same ``fold_fingerprint``. That
   value hashes the entire fold structure, so a single distinct value across
   members proves they were cross-validated on identical partitions.
2. **Same class vocabulary.** The ``prob_*`` columns must match in name *and*
   order, otherwise column 2 of one member would be averaged with a different
   class of another.
3. **Same rows.** Per fold, the set of ``slice_uid`` values must be identical
   across members, and each member's ``y_true`` for a given slice must agree.
   Rows are then reindexed to a common order rather than assumed aligned.

A note on how members should be chosen. Selecting members *after* looking at
their test scores is selection on the test set, and it inflates the result in
exactly the way this paper is trying to avoid. The API therefore takes an
explicit member list and offers no "top-k by test metric" mode. Where a
data-driven weighting is wanted, ``method="val_weighted"`` uses each member's
inner-validation monitor, which never saw the test fold.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ppgl.experiments import results as results_store
from ppgl.labels import active_classes
from ppgl.metrics import compute_all_metrics
from ppgl.utils.logging import get_logger

log = get_logger(__name__)

# How member probabilities are combined.
#   mean         - arithmetic mean of probabilities. The robust default: a member
#                  that is confidently wrong is diluted, not amplified.
#   geometric    - normalised geometric mean, i.e. mean in log space. Sharper
#                  than the arithmetic mean and closer to a product-of-experts;
#                  a single near-zero probability can veto a class.
#   val_weighted - arithmetic mean weighted per fold by each member's inner
#                  validation monitor. Leakage-free, since that monitor is
#                  computed on the inner split, never on the test fold.
COMBINE_METHODS = ("mean", "geometric", "val_weighted")

PROB_PREFIX = "prob_"
_ID_COLUMNS = ("slice_uid", "patient_uid", "y_true")


@dataclass
class EnsembleSpec:
    """What to combine and how."""

    members: List[str]
    name: str
    mode: str = "soft"
    method: str = "mean"
    weights: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError(
                f"an ensemble needs at least 2 members, got {self.members!r}"
            )
        if len(set(self.members)) != len(self.members):
            duplicates = sorted({m for m in self.members if self.members.count(m) > 1})
            raise ValueError(f"duplicate ensemble members would be double-counted: {duplicates}")
        if self.method not in COMBINE_METHODS:
            raise ValueError(f"method must be one of {COMBINE_METHODS}, got {self.method!r}")
        if self.weights is not None:
            if len(self.weights) != len(self.members):
                raise ValueError(
                    f"{len(self.weights)} weights for {len(self.members)} members"
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("ensemble weights must be non-negative")
            if float(sum(self.weights)) <= 0:
                raise ValueError("ensemble weights must not sum to zero")
            if self.method == "val_weighted":
                raise ValueError(
                    "method='val_weighted' derives its own weights; do not also pass weights"
                )

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "members": self.members,
                "mode": self.mode,
                "method": self.method,
                "weights": self.weights,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class MemberFold:
    """One member's stored predictions for one fold."""

    experiment: str
    frame: pd.DataFrame
    prob_columns: List[str]
    val_monitor: Optional[float] = None


@dataclass
class EnsembleReport:
    """What the run did, for logging and for the CLI to print."""

    name: str
    members: List[str]
    method: str
    mode: str
    n_folds: int = 0
    skipped: List[str] = field(default_factory=list)
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _fold_tags(output_dir: str, experiment: str, mode: str) -> List[str]:
    folds_root = os.path.join(results_store.run_dir(output_dir, experiment), "folds")
    if not os.path.isdir(folds_root):
        return []
    tags = []
    for tag in sorted(os.listdir(folds_root)):
        if os.path.isfile(os.path.join(folds_root, tag, f"predictions_{mode}.csv")):
            tags.append(tag)
    return tags


def _parse_tag(tag: str) -> Tuple[int, int]:
    repeat, _, fold = tag.partition("f")
    return int(repeat.lstrip("r") or 0), int(fold or 0)


def _split_fingerprints(output_dir: str, members: Sequence[str]) -> Dict[str, set]:
    """Collect the split fingerprint(s) each member was run under."""
    out: Dict[str, set] = {}
    for member in members:
        folds_root = os.path.join(results_store.run_dir(output_dir, member), "folds")
        seen: set = set()
        if os.path.isdir(folds_root):
            for tag in sorted(os.listdir(folds_root)):
                path = os.path.join(folds_root, tag, results_store.METRICS_FILENAME)
                if not os.path.isfile(path):
                    continue
                try:
                    with open(path) as fh:
                        value = json.load(fh).get("fold_fingerprint")
                except Exception:
                    continue
                if value:
                    seen.add(value)
        out[member] = seen
    return out


def _class_names_from_header(
    output_dir: str, experiment: str, tag: str, mode: str
) -> List[str]:
    """Class names in stored column order, read from the header alone."""
    path = os.path.join(
        results_store.run_dir(output_dir, experiment), "folds", tag, f"predictions_{mode}.csv"
    )
    header = pd.read_csv(path, nrows=0)
    return [c[len(PROB_PREFIX):] for c in header.columns if c.startswith(PROB_PREFIX)]


def _load_member_fold(
    output_dir: str, experiment: str, tag: str, mode: str
) -> Optional[MemberFold]:
    path = os.path.join(
        results_store.run_dir(output_dir, experiment), "folds", tag, f"predictions_{mode}.csv"
    )
    if not os.path.isfile(path):
        return None
    frame = pd.read_csv(path)
    prob_columns = [c for c in frame.columns if c.startswith(PROB_PREFIX)]
    if not prob_columns:
        raise ValueError(f"{path} has no {PROB_PREFIX}* columns")
    for column in _ID_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"{path} is missing required column {column!r}")

    monitor: Optional[float] = None
    metrics_path = os.path.join(
        results_store.run_dir(output_dir, experiment), "folds", tag, results_store.METRICS_FILENAME
    )
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as fh:
                raw = json.load(fh).get("best_val_monitor")
            if raw is not None and np.isfinite(float(raw)):
                monitor = float(raw)
        except Exception:
            monitor = None

    return MemberFold(
        experiment=experiment, frame=frame, prob_columns=prob_columns, val_monitor=monitor
    )


# --------------------------------------------------------------------------- #
# Combining
# --------------------------------------------------------------------------- #
def _normalise_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 0.0, None)
    totals = probabilities.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    out = probabilities / safe
    dead = (totals <= 0).reshape(-1)
    if dead.any():
        out[dead] = 1.0 / probabilities.shape[1]
    return out


def _resolve_weights(spec: EnsembleSpec, folds: Sequence[MemberFold]) -> np.ndarray:
    """Per-member weights for one fold, always normalised to sum to 1."""
    n = len(folds)
    if spec.weights is not None:
        weights = np.asarray(spec.weights, dtype=np.float64)
    elif spec.method == "val_weighted":
        monitors = [f.val_monitor for f in folds]
        if any(m is None for m in monitors):
            missing = [f.experiment for f in folds if f.val_monitor is None]
            log.warning(
                "method='val_weighted' but %s stored no best_val_monitor for this fold; "
                "falling back to equal weights",
                ", ".join(sorted(missing)),
            )
            weights = np.ones(n, dtype=np.float64)
        else:
            raw = np.asarray(monitors, dtype=np.float64)
            # The monitor is macro-F1 in [0, 1], but shift to be safe against a
            # negative-valued monitor: weights must stay non-negative.
            shifted = raw - raw.min()
            weights = shifted if shifted.sum() > 0 else np.ones(n, dtype=np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)
    return weights / weights.sum()


def combine_probabilities(
    blocks: Sequence[np.ndarray],
    method: str = "mean",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Combine aligned ``(N, C)`` probability blocks into one."""
    if not blocks:
        raise ValueError("no probability blocks to combine")
    stacked = np.stack([_normalise_rows(np.asarray(b, dtype=np.float64)) for b in blocks])
    if weights is None:
        weights = np.ones(stacked.shape[0], dtype=np.float64) / stacked.shape[0]
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)

    if method == "geometric":
        # Mean in log space. The floor keeps a zero probability from sending the
        # whole term to -inf and wiping out the class for every member.
        logs = np.log(np.clip(stacked, 1e-12, None))
        return _normalise_rows(np.exp((logs * weights).sum(axis=0)))
    # "mean" and "val_weighted" are both weighted arithmetic means; they differ
    # only in where the weights came from.
    return _normalise_rows((stacked * weights).sum(axis=0))


def _align(
    folds: Sequence[MemberFold], tag: str
) -> Tuple[pd.DataFrame, List[str], List[np.ndarray]]:
    """Check the members describe the same rows, and return them in one order.

    Returns the identifier frame, the probability column names, and one
    row-aligned ``(N, C)`` block per member.
    """
    reference = folds[0]
    prob_columns = list(reference.prob_columns)

    for other in folds[1:]:
        if list(other.prob_columns) != prob_columns:
            raise ValueError(
                f"{tag}: class columns differ between {reference.experiment} "
                f"({prob_columns}) and {other.experiment} ({other.prob_columns}). "
                "Averaging these would mix different classes."
            )

    base_ids = reference.frame["slice_uid"]
    if base_ids.duplicated().any():
        duplicated = base_ids[base_ids.duplicated()].unique()[:3].tolist()
        raise ValueError(
            f"{tag}: {reference.experiment} has duplicate slice_uid values "
            f"(e.g. {duplicated}); cannot align members unambiguously"
        )
    base_set = set(base_ids)

    ordered: List[pd.DataFrame] = []
    for member in folds:
        frame = member.frame
        if frame["slice_uid"].duplicated().any():
            raise ValueError(f"{tag}: {member.experiment} has duplicate slice_uid values")
        if set(frame["slice_uid"]) != base_set:
            only_ref = len(base_set - set(frame["slice_uid"]))
            only_other = len(set(frame["slice_uid"]) - base_set)
            raise ValueError(
                f"{tag}: test rows differ between {reference.experiment} and "
                f"{member.experiment} ({only_ref} missing, {only_other} extra). "
                "The members were not evaluated on the same fold."
            )
        ordered.append(frame.set_index("slice_uid").reindex(base_ids).reset_index())

    truth = ordered[0]["y_true"].to_numpy()
    for frame, member in zip(ordered[1:], folds[1:]):
        if not np.array_equal(frame["y_true"].to_numpy(), truth):
            n_bad = int((frame["y_true"].to_numpy() != truth).sum())
            raise ValueError(
                f"{tag}: y_true disagrees on {n_bad} rows between "
                f"{reference.experiment} and {member.experiment}. One of them was "
                "run under a different label space."
            )

    combined = ordered[0][list(_ID_COLUMNS)].copy()
    if "region_true" in ordered[0].columns:
        combined["region_true"] = ordered[0]["region_true"].to_numpy()
    blocks = [frame[prob_columns].to_numpy(dtype=np.float64) for frame in ordered]
    return combined, prob_columns, blocks


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_ensemble(
    output_dir: str,
    spec: EnsembleSpec,
    overwrite: bool = False,
) -> EnsembleReport:
    """Combine stored predictions and write the result as a new experiment."""
    report = EnsembleReport(
        name=spec.name, members=list(spec.members), method=spec.method, mode=spec.mode
    )

    if spec.name in spec.members:
        raise ValueError(f"ensemble {spec.name!r} cannot be one of its own members")

    # Gate 1: identical splits.
    fingerprints = _split_fingerprints(output_dir, spec.members)
    missing = [m for m, v in fingerprints.items() if not v]
    if missing:
        raise ValueError(
            f"no stored fold fingerprints for {missing}; run those experiments first"
        )
    multi = {m: sorted(v) for m, v in fingerprints.items() if len(v) > 1}
    if multi:
        raise ValueError(
            f"these members contain folds from more than one split scheme: {multi}. "
            "Re-run them so each experiment is internally consistent."
        )
    distinct = {next(iter(v)) for v in fingerprints.values()}
    if len(distinct) > 1:
        raise ValueError(
            "members were cross-validated on different splits, so their "
            f"predictions are not comparable per fold: "
            f"{ {m: next(iter(v)) for m, v in fingerprints.items()} }"
        )
    split_fingerprint = next(iter(distinct))

    # Folds present for every member. A fold missing from one member is skipped
    # rather than partially ensembled, so every reported fold has the same
    # member count and the paired tests stay balanced.
    per_member = {m: set(_fold_tags(output_dir, m, spec.mode)) for m in spec.members}
    empty = [m for m, tags in per_member.items() if not tags]
    if empty:
        raise ValueError(
            f"no predictions_{spec.mode}.csv found for {empty}; "
            f"check the mode (available: {[results_store.available_modes(output_dir, m) for m in empty]})"
        )
    common = set.intersection(*per_member.values())
    for member, tags in per_member.items():
        for tag in sorted(tags - common):
            report.skipped.append(f"{member}:{tag}")
    if not common:
        raise ValueError("members share no completed fold with predictions for this mode")

    log.info(
        "ensembling %d members over %d shared folds (method=%s, mode=%s)",
        len(spec.members),
        len(common),
        spec.method,
        spec.mode,
    )
    if report.skipped:
        log.warning(
            "%d member-folds skipped because they are not present in every member: %s",
            len(report.skipped),
            ", ".join(sorted(report.skipped)[:8]) + (" ..." if len(report.skipped) > 8 else ""),
        )

    # Class vocabulary comes from the stored column names, not from the active
    # label space: this command may be run in a process configured for a
    # different space than the one the members were trained under.
    class_names = _class_names_from_header(
        output_dir, spec.members[0], sorted(common, key=_parse_tag)[0], spec.mode
    )

    for tag in sorted(common, key=_parse_tag):
        repeat, fold = _parse_tag(tag)
        if not overwrite and results_store.is_fold_complete(
            output_dir, spec.name, repeat, fold, expected_fingerprint=spec.fingerprint()
        ):
            log.debug("%s/%s already built; skipping", spec.name, tag)
            report.n_folds += 1
            continue

        folds = []
        for member in spec.members:
            loaded = _load_member_fold(output_dir, member, tag, spec.mode)
            if loaded is None:
                raise RuntimeError(f"{member}:{tag} vanished while building the ensemble")
            folds.append(loaded)

        combined, prob_columns, blocks = _align(folds, tag)
        weights = _resolve_weights(spec, folds)
        probabilities = combine_probabilities(blocks, method=spec.method, weights=weights)

        for idx, column in enumerate(prob_columns):
            combined[column] = probabilities[:, idx]

        metrics = compute_all_metrics(
            combined["y_true"].to_numpy(),
            probabilities,
            patient_uids=combined["patient_uid"].tolist(),
            class_names=class_names,
        )

        results_store.save_fold_result(
            output_dir,
            spec.name,
            repeat,
            fold,
            metrics_by_mode={"soft": metrics},
            predictions_by_mode={"soft": combined},
            extra={
                "config_fingerprint": spec.fingerprint(),
                "fold_fingerprint": split_fingerprint,
                "ensemble_members": list(spec.members),
                "ensemble_method": spec.method,
                "ensemble_source_mode": spec.mode,
                "ensemble_weights": [round(float(w), 6) for w in weights],
                "n_test_slices": int(len(combined)),
                "n_test_patients": int(combined["patient_uid"].nunique()),
                "derived": True,
            },
        )
        report.n_folds += 1
        report.fold_metrics.append(
            {"tag": tag, "patient_macro_f1": metrics.get("patient_macro_f1"),
             "macro_f1": metrics.get("macro_f1")}
        )

    _write_metadata(output_dir, spec, split_fingerprint, class_names)
    results_store.summarise_experiment(output_dir, spec.name, mode="soft")
    return report


def _write_metadata(
    output_dir: str,
    spec: EnsembleSpec,
    split_fingerprint: str,
    class_names: Sequence[str],
) -> None:
    """Write a config.json for the pseudo-experiment.

    The label space is copied from the first member so that the analysis
    commands, which read the run's own config.json to decide how to name
    probability columns, resolve the same vocabulary the members used.
    """
    directory = results_store.run_dir(output_dir, spec.name)
    os.makedirs(directory, exist_ok=True)

    member_config = results_store.load_config_dict(output_dir, spec.members[0])
    label_space = (member_config.get("data") or {}).get("label_space", "cluster4")

    payload = {
        "experiment": spec.name,
        "output_dir": output_dir,
        "derived": True,
        "kind": "post_hoc_ensemble",
        "data": {"label_space": label_space},
        "ensemble": {
            "members": list(spec.members),
            "method": spec.method,
            "source_mode": spec.mode,
            "weights": spec.weights,
            "fingerprint": spec.fingerprint(),
            "split_fingerprint": split_fingerprint,
            "class_names": list(class_names),
        },
    }
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def parse_spec(
    text: str,
    name: Optional[str] = None,
    mode: str = "soft",
    method: str = "mean",
) -> EnsembleSpec:
    """Parse ``"a+b+c"`` or ``"a:2+b:1"`` into an :class:`EnsembleSpec`.

    The optional ``:w`` suffix gives a member a relative weight.
    """
    members: List[str] = []
    weights: List[float] = []
    weighted = False
    for part in text.split("+"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            member, _, raw = part.rpartition(":")
            members.append(member.strip())
            weights.append(float(raw))
            weighted = True
        else:
            members.append(part)
            weights.append(1.0)
    resolved = name or ("ensemble_" + "_".join(members))
    return EnsembleSpec(
        members=members,
        name=resolved,
        mode=mode,
        method=method,
        weights=weights if weighted else None,
    )


__all__ = [
    "COMBINE_METHODS",
    "EnsembleReport",
    "EnsembleSpec",
    "build_ensemble",
    "combine_probabilities",
    "parse_spec",
]
