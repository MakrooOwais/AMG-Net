"""Configurable target label space.

The 750-fold grid showed that on the 4-class task every deep model lands within
0.33-0.41 patient macro-F1 with a ±0.09-0.11 fold spread. From the observed
between-model standard error (~0.049), a difference must reach roughly **+0.10**
for nominal significance and **+0.15** to survive Holm across a large family. No
architectural change came close. That is the signature of an information-limited
task, not an architecture-limited one.

The response is to change the question rather than the network. This module makes
the target label space a configuration choice.

``cluster4``
    The original 4-way task: SDHx / VHL-EPAS1 / kinase / sporadic. Kept as the
    reference so previous results remain comparable.
``pseudohypoxic2``
    Cluster 1 (pseudohypoxic = SDHx + VHL/EPAS1) versus the rest. 427 slices /
    181 patients against 203 / 106 - far better balanced than a 4-way task whose
    smallest class has 39 patients. This is the target the clinical literature
    actually asks about (Yi et al. predict the pseudohypoxic subtype from CT
    radiomics in a multicentre cohort), so it also supplies an external
    comparator instead of only self-comparison, and cluster 1 membership is the
    axis that carries the higher metastatic risk.
``cluster3``
    Pseudohypoxic / kinase-signalling / sporadic. Collapses only 1a and 1b, which
    is where the sample sizes are thinnest, while keeping the clinically distinct
    kinase group separate.
``sdhx_vs_rest``
    SDHx versus everything else. The single most actionable split for
    surveillance decisions, and the majority class is large enough to estimate.

Mechanics: the manifest always stores the canonical 4-way cluster.
:func:`apply_label_space` rewrites ``cluster`` / ``cluster_idx`` in place to the
active space and preserves the original as ``cluster_raw`` / ``cluster_raw_idx``,
so splitting, datasets, losses and metrics need no changes - they simply see a
different number of classes.

The active space is process-global. Each training job runs in its own spawned
process with exactly one config, and each analysis command resolves one space
from the stored run config, so a module-level value is safe here and avoids
threading a ``classes`` argument through several dozen call sites where a missed
one would silently mislabel a column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ppgl import CLUSTERS
from ppgl.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LabelSpace:
    """A target vocabulary and the mapping onto it from the canonical clusters."""

    name: str
    classes: Tuple[str, ...]
    # Canonical cluster -> index in ``classes``, or None to exclude those rows.
    mapping: Dict[str, Optional[int]]
    display: Dict[str, str]
    description: str
    # True when a positive class is meaningful (binary tasks), enabling
    # sensitivity/specificity/PPV reporting alongside the macro metrics.
    positive_class: Optional[str] = None

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def index_of(self, cluster: str) -> Optional[int]:
        return self.mapping.get(cluster)


LABEL_SPACES: Dict[str, LabelSpace] = {
    "cluster4": LabelSpace(
        name="cluster4",
        classes=CLUSTERS,
        mapping={"SDHx": 0, "VHL_EPAS1": 1, "Kinase": 2, "Sporadic": 3},
        display={
            "SDHx": "SDHx",
            "VHL_EPAS1": "VHL/EPAS1",
            "Kinase": "Kinase",
            "Sporadic": "Sporadic",
        },
        description="Original 4-way genetic cluster task",
    ),
    "pseudohypoxic2": LabelSpace(
        name="pseudohypoxic2",
        classes=("Pseudohypoxic", "Non_pseudohypoxic"),
        mapping={"SDHx": 0, "VHL_EPAS1": 0, "Kinase": 1, "Sporadic": 1},
        display={
            "Pseudohypoxic": "Pseudohypoxic (cluster 1)",
            "Non_pseudohypoxic": "Non-pseudohypoxic",
        },
        description="Cluster 1 (SDHx + VHL/EPAS1) vs kinase signalling + sporadic",
        positive_class="Pseudohypoxic",
    ),
    "cluster3": LabelSpace(
        name="cluster3",
        classes=("Pseudohypoxic", "Kinase", "Sporadic"),
        mapping={"SDHx": 0, "VHL_EPAS1": 0, "Kinase": 1, "Sporadic": 2},
        display={
            "Pseudohypoxic": "Pseudohypoxic (cluster 1)",
            "Kinase": "Kinase (cluster 2)",
            "Sporadic": "Sporadic",
        },
        description="Collapse cluster 1a/1b only; keep kinase and sporadic separate",
    ),
    "sdhx_vs_rest": LabelSpace(
        name="sdhx_vs_rest",
        classes=("SDHx", "Non_SDHx"),
        mapping={"SDHx": 0, "VHL_EPAS1": 1, "Kinase": 1, "Sporadic": 1},
        display={"SDHx": "SDHx", "Non_SDHx": "Non-SDHx"},
        description="SDHx vs rest; the split most relevant to surveillance decisions",
        positive_class="SDHx",
    ),
}

DEFAULT_LABEL_SPACE = "cluster4"

# --------------------------------------------------------------------------- #
# Active space (process-global; see the module docstring for why)
# --------------------------------------------------------------------------- #
_ACTIVE: LabelSpace = LABEL_SPACES[DEFAULT_LABEL_SPACE]


def get_label_space(name: str) -> LabelSpace:
    if name not in LABEL_SPACES:
        raise ValueError(
            f"unknown label space {name!r}; available: {sorted(LABEL_SPACES)}"
        )
    return LABEL_SPACES[name]


def set_active_label_space(name: str) -> LabelSpace:
    """Set the process-wide target space. Call once, early, per job or analysis."""
    global _ACTIVE
    _ACTIVE = get_label_space(name)
    return _ACTIVE


def active_label_space() -> LabelSpace:
    return _ACTIVE


def active_classes() -> Tuple[str, ...]:
    """Class names of the active space. Replaces direct use of ``CLUSTERS``."""
    return _ACTIVE.classes


def n_active_classes() -> int:
    return _ACTIVE.n_classes


def display_name(class_name: str) -> str:
    return _ACTIVE.display.get(class_name, class_name.replace("_", " "))


# --------------------------------------------------------------------------- #
# Applying a space to a manifest
# --------------------------------------------------------------------------- #
def apply_label_space(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rewrite ``cluster`` / ``cluster_idx`` to the requested space.

    The canonical 4-way labels are preserved as ``cluster_raw`` /
    ``cluster_raw_idx`` so per-cluster breakdowns and the dataset table can still
    be reported against the original vocabulary.

    Rows whose cluster maps to ``None`` are dropped, and the count is logged.
    """
    space = get_label_space(name)
    out = frame.copy()

    if "cluster_raw" not in out.columns:
        out["cluster_raw"] = out["cluster"]
        out["cluster_raw_idx"] = out["cluster_idx"]

    mapped = out["cluster_raw"].map(space.mapping)
    unknown = out.loc[mapped.isna() & ~out["cluster_raw"].isin(space.mapping), "cluster_raw"]
    if len(unknown):
        raise ValueError(
            f"label space {name!r} has no mapping for cluster(s) "
            f"{sorted(unknown.unique())}"
        )

    dropped = int(mapped.isna().sum())
    if dropped:
        log.info("label space %r excludes %d slices by design", name, dropped)
        out = out.loc[mapped.notna()].copy()
        mapped = mapped.loc[mapped.notna()]

    out["cluster_idx"] = mapped.astype(int).to_numpy()
    out["cluster"] = out["cluster_idx"].map(dict(enumerate(space.classes)))

    if name != DEFAULT_LABEL_SPACE:
        counts = out["cluster"].value_counts().to_dict()
        patients = out.drop_duplicates("patient_uid")["cluster"].value_counts().to_dict()
        log.info(
            "label space %r: %d slices %s / %d patients %s",
            name,
            len(out),
            counts,
            out["patient_uid"].nunique(),
            patients,
        )
    return out.reset_index(drop=True)


def binary_metrics(
    y_true: Sequence[int],
    y_prob,
    space: Optional[LabelSpace] = None,
) -> Dict[str, float]:
    """Sensitivity / specificity / PPV / NPV for a binary space.

    Returns an empty dict for multi-class spaces. Reported alongside the macro
    metrics because a clinical reader of a binary pseudohypoxic model wants
    operating characteristics, not only macro-F1.
    """
    import numpy as np

    space = space or active_label_space()
    if space.n_classes != 2 or space.positive_class is None:
        return {}

    positive = space.classes.index(space.positive_class)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)

    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fp = int(((y_pred == positive) & (y_true != positive)).sum())
    tn = int(((y_pred != positive) & (y_true != positive)).sum())
    fn = int(((y_pred != positive) & (y_true == positive)).sum())

    def _safe(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "ppv": _safe(tp, tp + fp),
        "npv": _safe(tn, tn + fn),
        "positive_class": space.positive_class,
    }


def describe_spaces() -> List[str]:
    return [f"{s.name}: {s.description} ({s.n_classes} classes)" for s in LABEL_SPACES.values()]


__all__ = [
    "DEFAULT_LABEL_SPACE",
    "LABEL_SPACES",
    "LabelSpace",
    "active_classes",
    "active_label_space",
    "apply_label_space",
    "binary_metrics",
    "describe_spaces",
    "display_name",
    "get_label_space",
    "n_active_classes",
    "set_active_label_space",
]
