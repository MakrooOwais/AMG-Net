"""PPGL genetic-cluster classification from CT.

Codebase for the IEEE ISBI 2027 submission.

Design constraints baked into this package (see README.md):

* Patient identity is ``{CLASS}_{NUMERIC_ID}``. Numeric IDs are reused across
  classes (84 collisions in this archive), so grouping on the bare number
  silently merges different patients. Every split uses ``patient_uid``.
* Images are 8-bit RGBA TIFFs already windowed at 50/450 HU. Hounsfield units
  are unrecoverable, so no re-windowing and no HU-calibrated radiomics.
* Exactly one definition of each metric is applied to every model, including
  baselines. See :mod:`ppgl.metrics`.
"""

__version__ = "0.1.0"

# Canonical genetic-cluster label space. Order is fixed and must not change:
# stored per-fold prediction arrays are indexed by it.
CLUSTERS = ("SDHx", "VHL_EPAS1", "Kinase", "Sporadic")

# Raw on-disk class-directory prefix -> canonical cluster.
# SDHB and SDHx are separate directories in the archive but are the same
# cluster 1a in the paper's label space.
RAW_CLASS_TO_CLUSTER = {
    "SDHB": "SDHx",
    "SDHx": "SDHx",
    "VHL_EPAS1": "VHL_EPAS1",
    "Kinase": "Kinase",
    "Sporadic": "Sporadic",
}

# Canonical anatomical-region label space (routing experts).
REGIONS = ("head_neck", "chest", "abdomen_pelvis")

CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}

__all__ = [
    "__version__",
    "CLUSTERS",
    "REGIONS",
    "RAW_CLASS_TO_CLUSTER",
    "CLUSTER_TO_IDX",
    "REGION_TO_IDX",
]
