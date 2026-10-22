# Clinical covariate experiments

These four configs are **not** part of `configs/experiments/*.yaml` and are not
run by `run_all.sh` by default. They require a per-patient clinical sheet that
does not exist in the archive, so including them in the main glob would produce
guaranteed failures on every fold.

## Why these rows matter

Every architecture tried on the 4-class target lands within 0.33-0.41 patient
macro-F1. That pattern says the limit is the information in a single windowed
8-bit slice, not model capacity. Clinical covariates are the only genuinely new
information channel available without new imaging, which makes `81_fusion` the
most plausible route to an effect large enough to detect (roughly +0.10 patient
macro-F1 is needed for nominal significance at this fold count).

Reviewer #1 asked for the clinical comparison directly. It has to be reported
whatever it shows, including the outcome where covariates alone match or beat the
imaging model.

## Workflow

1. Emit the blank sheet, one row per patient:

   ```bash
   python -m ppgl.cli clinical-template --out results/clinical_template.csv
   ```

   This also writes a legend file describing every field and its accepted values.
   287 rows, keyed by `patient_uid` (the `{CLASS}_{ID}` form, because 84 numeric
   patient IDs collide across classes).

2. Fill it from chart review. Partial completion is fine: missing values get an
   explicit indicator column, because in chart review the fact that something was
   not recorded is itself informative. Unrecognised categories are mapped to
   missing with a warning rather than silently dropped.

3. Run the four rows:

   ```bash
   ./run_all.sh --clinical-csv results/clinical.csv --from 7
   ```

   or individually:

   ```bash
   python -m ppgl.cli train --config configs/clinical/81_fusion.yaml \
     --set data.clinical_csv=results/clinical.csv
   ```

## Leakage

`ClinicalPreprocessor` is fitted on the **training rows of each fold only** -
imputation statistics, standardisation and category vocabularies all come from
inside the fold. Category vocabularies are taken from the schema rather than from
the data so that a category absent from one fold still occupies a stable column
position. Fitting on the whole manifest was verified to give different numbers,
which is why it is not done.

## Reading the result

The comparison is three-way, not two-way:

| row | question it answers |
|---|---|
| `swin` (main grid) | what imaging alone achieves |
| `80_clinical_only` | what covariates alone achieve |
| `81_fusion` | whether the combination beats **both** |
| `82_clinical_net` | controls for the optimiser in that comparison |
| `83_clinical_radiomics` | strongest non-deep comparator with both channels |

Fusion beating only the image model does not establish that fusion helps, since
the covariates alone may already account for it.
