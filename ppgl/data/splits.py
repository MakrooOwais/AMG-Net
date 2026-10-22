"""Patient-level cross-validation splits.

Two properties of this dataset make the split logic worth stating explicitly.

**The group key is ``patient_uid``, not the numeric ID.** 84 numeric IDs in this
archive are reused across class directories, so ``00004`` denotes four different
patients. Grouping on the number merges them and leaks across folds.

**Cluster is constant within a patient** (it is encoded in the directory name),
so stratifying *patients* by cluster is exactly equivalent to
``StratifiedGroupKFold`` over slices with ``groups=patient`` - and it is
deterministic and easier to audit. Patients are the unit of splitting; slices
follow their patient.

Repeated CV (default 5 x 10) is used because a single 10-fold estimate at 287
patients is too noisy to support the model comparisons this paper makes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from ppgl.labels import active_classes
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


class Fold(NamedTuple):
    """One evaluation fold. Indices reference rows of the slice-level frame."""

    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def tag(self) -> str:
        return f"r{self.repeat}f{self.fold}"


def patient_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with its cluster, asserting the label is consistent."""
    grouped = frame.groupby("patient_uid")["cluster"].nunique()
    inconsistent = grouped[grouped > 1]
    if len(inconsistent):
        raise ValueError(
            "patients with more than one cluster label: "
            f"{inconsistent.index.tolist()[:5]} - the manifest is corrupt"
        )
    table = (
        frame.drop_duplicates("patient_uid")[["patient_uid", "cluster", "cluster_idx"]]
        .sort_values("patient_uid")
        .reset_index(drop=True)
    )
    return table


def make_folds(
    frame: pd.DataFrame,
    n_splits: int = 10,
    n_repeats: int = 5,
    seed: int = 1337,
    inner_val_fraction: float = 0.15,
    require_all_classes: bool = True,
) -> List[Fold]:
    """Build all ``n_repeats * n_splits`` folds.

    The inner validation split (used for early stopping) is carved out of the
    training patients, never out of the test fold.
    """
    from sklearn.model_selection import StratifiedKFold

    patients = patient_table(frame)
    patient_to_rows: Dict[str, np.ndarray] = {
        uid: group.to_numpy()
        for uid, group in frame.reset_index().groupby("patient_uid")["index"]
    }

    folds: List[Fold] = []
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold_idx, (train_pat_idx, test_pat_idx) in enumerate(
            splitter.split(patients["patient_uid"], patients["cluster_idx"])
        ):
            train_patients = patients.iloc[train_pat_idx]
            test_patients = patients.iloc[test_pat_idx]

            inner_train, inner_val = _inner_split(
                train_patients,
                fraction=inner_val_fraction,
                seed=seed + repeat * 1000 + fold_idx,
            )

            fold = Fold(
                repeat=repeat,
                fold=fold_idx,
                train_idx=_rows_for(inner_train, patient_to_rows),
                val_idx=_rows_for(inner_val, patient_to_rows),
                test_idx=_rows_for(test_patients["patient_uid"].tolist(), patient_to_rows),
            )
            _validate_fold(frame, fold, require_all_classes)
            folds.append(fold)

    log.info(
        "built %d folds (%d repeats x %d splits) over %d patients / %d slices",
        len(folds),
        n_repeats,
        n_splits,
        len(patients),
        len(frame),
    )
    return folds


def _inner_split(
    train_patients: pd.DataFrame,
    fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Stratified patient-level holdout from the training patients."""
    from sklearn.model_selection import StratifiedKFold

    if fraction <= 0:
        uids = train_patients["patient_uid"].tolist()
        return uids, uids  # degenerate: early stopping on the training split

    n_inner = max(2, int(round(1.0 / max(fraction, 1e-6))))
    n_inner = min(n_inner, _max_splits(train_patients["cluster_idx"]))
    splitter = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    train_pos, val_pos = next(
        iter(splitter.split(train_patients["patient_uid"], train_patients["cluster_idx"]))
    )
    return (
        train_patients.iloc[train_pos]["patient_uid"].tolist(),
        train_patients.iloc[val_pos]["patient_uid"].tolist(),
    )


def _max_splits(labels: pd.Series) -> int:
    """StratifiedKFold cannot use more splits than the rarest class has members."""
    counts = labels.value_counts()
    return max(2, int(counts.min()))


def _rows_for(patient_uids, patient_to_rows: Dict[str, np.ndarray]) -> np.ndarray:
    chunks = [patient_to_rows[uid] for uid in patient_uids if uid in patient_to_rows]
    if not chunks:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(chunks)).astype(np.int64)


def _validate_fold(frame: pd.DataFrame, fold: Fold, require_all_classes: bool) -> None:
    """No patient may appear in two parts of a fold; every class must be present."""
    parts = {
        "train": set(frame.iloc[fold.train_idx]["patient_uid"]),
        "val": set(frame.iloc[fold.val_idx]["patient_uid"]),
        "test": set(frame.iloc[fold.test_idx]["patient_uid"]),
    }
    for left, right in (("train", "test"), ("val", "test"), ("train", "val")):
        overlap = parts[left] & parts[right]
        if overlap:
            raise AssertionError(
                f"patient leakage between {left} and {right} in {fold.tag}: "
                f"{sorted(overlap)[:5]}"
            )

    if not require_all_classes:
        return
    for name, idx in (("train", fold.train_idx), ("test", fold.test_idx)):
        present = set(frame.iloc[idx]["cluster"].unique())
        missing = set(active_classes()) - present
        if missing:
            raise AssertionError(
                f"fold {fold.tag} {name} split is missing cluster(s) {sorted(missing)}. "
                "Reduce cv.n_splits or set cv.require_all_classes_per_fold=false."
            )


def folds_to_frame(frame: pd.DataFrame, folds: List[Fold]) -> pd.DataFrame:
    """Long-format fold assignment table, written alongside results for audit."""
    records = []
    for fold in folds:
        for split, indices in (
            ("train", fold.train_idx),
            ("val", fold.val_idx),
            ("test", fold.test_idx),
        ):
            for row_idx in indices:
                row = frame.iloc[int(row_idx)]
                records.append(
                    {
                        "repeat": fold.repeat,
                        "fold": fold.fold,
                        "split": split,
                        "slice_uid": row["slice_uid"],
                        "patient_uid": row["patient_uid"],
                        "cluster": row["cluster"],
                    }
                )
    return pd.DataFrame.from_records(records)


def save_fold_summary(frame: pd.DataFrame, folds: List[Fold], path: str) -> None:
    """Per-fold patient and class counts, for the paper's reproducibility appendix."""
    summary = []
    for fold in folds:
        entry: Dict[str, object] = {"repeat": fold.repeat, "fold": fold.fold}
        for split, indices in (
            ("train", fold.train_idx),
            ("val", fold.val_idx),
            ("test", fold.test_idx),
        ):
            subset = frame.iloc[indices]
            entry[f"{split}_slices"] = int(len(subset))
            entry[f"{split}_patients"] = int(subset["patient_uid"].nunique())
            for cluster in active_classes():
                entry[f"{split}_{cluster}"] = int((subset["cluster"] == cluster).sum())
        summary.append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(summary).to_csv(path, index=False)
    log.info("wrote fold summary to %s", path)


def iter_folds(
    folds: List[Fold],
    repeats: Optional[List[int]] = None,
    fold_ids: Optional[List[int]] = None,
) -> Iterator[Fold]:
    """Filter folds by repeat and/or fold index, for partial reruns."""
    for fold in folds:
        if repeats is not None and fold.repeat not in repeats:
            continue
        if fold_ids is not None and fold.fold not in fold_ids:
            continue
        yield fold


def fingerprint(folds: List[Fold]) -> str:
    """Stable hash of the split structure.

    Recorded with every result so that a paired test can refuse to compare two
    models that were not evaluated on identical folds.
    """
    import hashlib

    payload = json.dumps(
        [
            [f.repeat, f.fold, f.train_idx.tolist(), f.val_idx.tolist(), f.test_idx.tolist()]
            for f in folds
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


__all__ = [
    "Fold",
    "fingerprint",
    "folds_to_frame",
    "iter_folds",
    "make_folds",
    "patient_table",
    "save_fold_summary",
]
