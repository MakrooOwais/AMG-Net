"""Clinical covariates: loading, per-fold preprocessing, and a blank template.

This exists to answer a specific question - whether the
imaging model adds anything beyond information a clinician already has. The
literature says those covariates carry real genotype signal: germline variant
presence tracks tumour topography, age under 45 and syndromic features (Lima et
al., Endocrine Connections 2023), and SDHx carriers show a predominance of
extra-adrenal tumours. So a clinical-only model is a serious baseline, and a
clinical+imaging fusion model is the most likely route to a difference large
enough to detect at this sample size.

**These variables are not in the archive.** They come from the chart review that
is already needed for the metastatic/multifocal accounting. Until that CSV
exists, models requiring it fail fast with instructions rather than training on
zeros. :func:`write_template` emits the blank sheet to fill in.

Leakage
-------
Imputation statistics and standardisation are fit on the **training rows of each
fold only**, never on the full table. That is why the preprocessing lives in
:class:`ClinicalPreprocessor` and is applied per fold by the runner, rather than
being baked into the manifest. Category vocabularies are taken from the whole
column, since the set of possible values is metadata and carries no outcome
information; only the fitted statistics are fold-local.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ppgl.utils.logging import get_logger

log = get_logger(__name__)

PREFIX = "clin_"


class Field:
    """One clinical variable: how to read it and how to encode it."""

    def __init__(self, name: str, kind: str, description: str, categories: Optional[Sequence[str]] = None) -> None:
        if kind not in ("numeric", "binary", "categorical"):
            raise ValueError(f"unknown field kind {kind!r}")
        self.name = name
        self.kind = kind
        self.description = description
        self.categories = tuple(categories or ())


# The variables worth collecting, chosen from the review feedback and the
# associations reported in the PPGL literature. All are optional: whatever the
# chart review yields is used, the rest is ignored.
SCHEMA: Tuple[Field, ...] = (
    Field("age_at_diagnosis", "numeric", "Age in years at PPGL diagnosis"),
    Field("sex", "categorical", "Sex", ("F", "M")),
    Field(
        "tumour_site",
        "categorical",
        "Primary tumour site",
        ("adrenal", "extra_adrenal_abdominal", "thoracic", "head_neck"),
    ),
    Field("multifocal", "binary", "More than one synchronous primary lesion (0/1)"),
    Field("metastatic", "binary", "Metastatic disease at any point (0/1)"),
    Field("recurrent", "binary", "Recurrence after resection (0/1)"),
    Field("max_tumour_size_mm", "numeric", "Largest primary tumour diameter, mm"),
    Field(
        "biochemical_phenotype",
        "categorical",
        "Secretory phenotype",
        ("noradrenergic", "adrenergic", "dopaminergic", "nonsecreting"),
    ),
    Field("normetanephrine_x_uln", "numeric", "Plasma normetanephrine, multiples of upper limit"),
    Field("metanephrine_x_uln", "numeric", "Plasma metanephrine, multiples of upper limit"),
    Field("methoxytyramine_x_uln", "numeric", "Plasma 3-methoxytyramine, multiples of upper limit"),
    Field("family_history_ppgl", "binary", "Family history of PPGL (0/1)"),
    Field("syndromic_features", "binary", "Syndromic features present (0/1)"),
)

SCHEMA_BY_NAME: Dict[str, Field] = {field.name: field for field in SCHEMA}
ID_COLUMN = "patient_uid"


# --------------------------------------------------------------------------- #
# Template
# --------------------------------------------------------------------------- #
def write_template(manifest: pd.DataFrame, path: str) -> str:
    """Write a blank clinical sheet, one row per patient, for the chart review."""
    patients = (
        manifest.drop_duplicates(ID_COLUMN)[[ID_COLUMN, "cluster_raw"]]
        if "cluster_raw" in manifest.columns
        else manifest.drop_duplicates(ID_COLUMN)[[ID_COLUMN, "cluster"]]
    ).copy()
    patients.columns = [ID_COLUMN, "genetic_cluster_for_reference"]

    for field in SCHEMA:
        patients[field.name] = ""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    patients.sort_values(ID_COLUMN).to_csv(path, index=False)

    legend = os.path.splitext(path)[0] + "_legend.txt"
    with open(legend, "w") as handle:
        handle.write("Clinical variables for the PPGL genotype study\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(
            "One row per patient, keyed by patient_uid. Leave a cell blank if unknown;\n"
            "blanks are imputed per cross-validation fold and flagged with a missing\n"
            "indicator, so partial completion is usable. Do not delete rows.\n\n"
            "The genetic_cluster_for_reference column is provided only to help locate\n"
            "the record. It is NOT read back as a feature.\n\n"
        )
        for field in SCHEMA:
            allowed = f"  allowed: {', '.join(field.categories)}" if field.categories else ""
            handle.write(f"{field.name:26s} [{field.kind}] {field.description}{allowed}\n")

    log.info("wrote clinical template for %d patients to %s (legend: %s)", len(patients), path, legend)
    return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_clinical_csv(path: str) -> pd.DataFrame:
    """Read a clinical sheet, keeping only recognised columns."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"clinical CSV not found: {path!r}")

    table = pd.read_csv(path)
    if ID_COLUMN not in table.columns:
        raise ValueError(f"{path!r} must contain a {ID_COLUMN!r} column")

    known = [c for c in table.columns if c in SCHEMA_BY_NAME]
    ignored = [c for c in table.columns if c not in SCHEMA_BY_NAME and c != ID_COLUMN]
    if ignored:
        log.info("ignoring unrecognised clinical column(s): %s", ignored)
    if not known:
        raise ValueError(
            f"{path!r} contains no recognised clinical variables. Expected any of: "
            f"{sorted(SCHEMA_BY_NAME)}"
        )

    out = table[[ID_COLUMN] + known].copy()
    out = out.replace({"": np.nan, " ": np.nan})

    # Normalise categoricals to the declared vocabulary; unknown values become NaN
    # rather than silently forming a new category.
    for name in known:
        field = SCHEMA_BY_NAME[name]
        if field.kind == "categorical":
            normalised = out[name].astype("string").str.strip().str.lower()
            allowed = {c.lower(): c for c in field.categories}
            unknown = sorted(set(normalised.dropna().unique()) - set(allowed))
            if unknown:
                log.warning("column %r has unrecognised value(s) %s; treated as missing", name, unknown)
            out[name] = normalised.map(allowed)
        elif field.kind == "binary":
            out[name] = (
                out[name]
                .astype("string")
                .str.strip()
                .str.lower()
                .map({"1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
                      "y": 1.0, "n": 0.0})
            )
        else:
            out[name] = pd.to_numeric(out[name], errors="coerce")

    duplicates = out[ID_COLUMN][out[ID_COLUMN].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{path!r} has duplicate patient_uid rows: {duplicates[:5]}")

    coverage = {name: float(out[name].notna().mean()) for name in known}
    log.info("loaded %d clinical rows; per-column coverage: %s",
             len(out), {k: f"{v:.0%}" for k, v in coverage.items()})
    return out


def attach_clinical(frame: pd.DataFrame, data_cfg, required: bool = False) -> pd.DataFrame:
    """Merge clinical columns onto the slice-level frame, prefixed ``clin_``."""
    path = getattr(data_cfg, "clinical_csv", None)

    if not path:
        if required:
            raise ValueError(
                "this model needs clinical covariates but data.clinical_csv is not set.\n"
                "Generate the blank sheet, have it completed from the chart review, then pass it:\n"
                "  python -m ppgl.cli clinical-template --out results/clinical_template.csv\n"
                "  ./run_all.sh --set data.clinical_csv=results/clinical.csv"
            )
        return frame

    table = load_clinical_csv(path)
    renamed = {c: f"{PREFIX}{c}" for c in table.columns if c != ID_COLUMN}
    table = table.rename(columns=renamed)

    merged = frame.merge(table, on=ID_COLUMN, how="left", validate="many_to_one")

    n_patients = frame[ID_COLUMN].nunique()
    covered = merged.loc[merged[list(renamed.values())].notna().any(axis=1), ID_COLUMN].nunique()
    log.info("clinical data covers %d/%d patients", covered, n_patients)
    if required and covered == 0:
        raise ValueError(
            f"{path!r} matched none of the {n_patients} patient_uid values. The key must be "
            "'{CLASS}_{NUMERIC_ID}', e.g. SDHB_00000."
        )
    if covered < n_patients:
        log.warning(
            "%d patient(s) have no clinical row; their features are imputed and flagged missing",
            n_patients - covered,
        )
    return merged


def clinical_columns(frame: pd.DataFrame) -> List[str]:
    """Recognised clinical columns present on the frame, in schema order."""
    return [f"{PREFIX}{field.name}" for field in SCHEMA if f"{PREFIX}{field.name}" in frame.columns]


# --------------------------------------------------------------------------- #
# Per-fold preprocessing
# --------------------------------------------------------------------------- #
class ClinicalPreprocessor:
    """Encode clinical columns to a dense matrix, fit on training rows only.

    Produces, per source column: the standardised value (numeric/binary) or a
    one-hot block (categorical), plus a missing-indicator column. Missingness in a
    chart review is itself informative, so it is modelled rather than hidden.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        self.feature_names: List[str] = []
        self._centre: Dict[str, float] = {}
        self._scale: Dict[str, float] = {}
        self._categories: Dict[str, List[str]] = {}
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> "ClinicalPreprocessor":
        self.feature_names = []
        for column in self.columns:
            field = SCHEMA_BY_NAME[column[len(PREFIX):]]
            series = frame[column]

            if field.kind == "categorical":
                # Vocabulary from the declared schema, so a category absent from
                # one training fold still occupies a stable column position.
                self._categories[column] = list(field.categories)
                self.feature_names.extend(f"{column}={c}" for c in field.categories)
            else:
                values = pd.to_numeric(series, errors="coerce")
                centre = float(values.median()) if values.notna().any() else 0.0
                spread = float(values.std()) if values.notna().sum() > 1 else 0.0
                self._centre[column] = centre
                self._scale[column] = spread if spread > 1e-8 else 1.0
                self.feature_names.append(column)

            self.feature_names.append(f"{column}__missing")

        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() on the training rows first")

        blocks: List[np.ndarray] = []
        for column in self.columns:
            field = SCHEMA_BY_NAME[column[len(PREFIX):]]
            series = frame[column]
            missing = series.isna().to_numpy(dtype=np.float32).reshape(-1, 1)

            if field.kind == "categorical":
                categories = self._categories[column]
                onehot = np.zeros((len(frame), len(categories)), dtype=np.float32)
                lookup = {c: i for i, c in enumerate(categories)}
                for row, value in enumerate(series.tolist()):
                    index = lookup.get(value)
                    if index is not None:
                        onehot[row, index] = 1.0
                blocks.append(onehot)
            else:
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
                values = np.where(np.isnan(values), self._centre[column], values)
                standardised = (values - self._centre[column]) / self._scale[column]
                blocks.append(standardised.astype(np.float32).reshape(-1, 1))

            blocks.append(missing)

        if not blocks:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return np.hstack(blocks).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def build_fold_matrices(
    frame: pd.DataFrame,
    train_idx: Sequence[int],
    splits: Dict[str, Sequence[int]],
) -> Tuple[Dict[str, np.ndarray], ClinicalPreprocessor]:
    """Fit on ``train_idx`` and transform every split. Returns matrices per split."""
    columns = clinical_columns(frame)
    preprocessor = ClinicalPreprocessor(columns).fit(frame.iloc[list(train_idx)])
    matrices = {
        name: preprocessor.transform(frame.iloc[list(indices)])
        for name, indices in splits.items()
    }
    return matrices, preprocessor


__all__ = [
    "ClinicalPreprocessor",
    "Field",
    "ID_COLUMN",
    "PREFIX",
    "SCHEMA",
    "SCHEMA_BY_NAME",
    "attach_clinical",
    "build_fold_matrices",
    "clinical_columns",
    "load_clinical_csv",
    "write_template",
]
