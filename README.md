# Attention Mass Guidance for PPGL genetic-cluster classification from CT

Predicting the genetic cluster of a pheochromocytoma or paraganglioma (PPGL) from a
single axial contrast-enhanced CT slice. This repository contains the full
experimental pipeline: the proposed method, every comparison model, the
cross-validation protocol, the statistics, and the figure and table generators.

---

## Data availability and ethics

**No data is released with this repository, and none will be.** This is a code-only
release: the method, the comparison models, the protocol, the statistics and the
figure generators, with nothing patient-derived attached.

The cohort is retrospective human-subjects CT acquired at the National Institutes of
Health under an approved institutional review board protocol. It is not ours to
redistribute, and access is governed by that protocol rather than by this repository,
so please do not open issues requesting it.

Nothing patient-derived is tracked: no imaging, no annotations, no run outputs, no
model checkpoints, and no rendered figures. Per-slice predictions carry patient
identifiers of the form `{CLASS}_{ID}`, and the attention-comparison figure overlays
attention maps on real CT crops, so both are excluded. Checkpoints are derived from
patient data and individually exceed GitHub's file-size limit. `.gitignore` enforces
all of this with deliberately broad patterns; please keep it that way.

The code is runnable against an equivalently structured cohort of your own.
`detection_data_NIH/PUT_DATA_HERE.txt` documents the layout the loader expects. The
cohort statistics quoted below are the aggregate counts reported in the paper.

Pretrained ImageNet backbones are not bundled either; `python -m ppgl.cli fetch-weights`
downloads them from the Hugging Face hub into a local `weights/` cache.

## Licence

**Not yet chosen — see the repository owner before reuse.** Note for whoever sets this:
contributions by NIH federal employees were made as part of their official duties and
are considered Works of the United States Government, which affects how this can be
licensed. Settle that with the NIH co-authors and your institution's tech-transfer
office before adding a `LICENSE` file. Until one exists, default copyright applies and
others have no grant to reuse the code.

---

## The idea

**The task looks saturated.** Five families of model — hand-crafted radiomics with a
gradient-boosted tree, a ResNet-101, a ViT-B/16, a supervised-contrastive ViT, and a
Swin-T — trained under identical 5×10 patient-grouped cross-validation land within
0.072 patient macro-F1 of each other:

| model | patient macro-F1 | patient macro AUC |
|---|---|---|
| majority-class floor (arithmetic) | 0.1655 | 0.500 |
| radiomics + GBM | 0.3083 | 0.6619 |
| ResNet-101 | 0.3291 | 0.6495 |
| ViT-B/16 + SupCon | 0.3656 | 0.6518 |
| ViT-B/16 | 0.3761 | 0.6491 |
| Swin-T | 0.3803 | 0.6733 |

Of the 40 paired comparisons between them (10 pairs × 4 metrics, corrected resampled
t-test), **not one survives Holm correction**. Capacity and inductive bias are not the
binding constraint.

**The diagnosis is where the encoder looks.** Train the recipe without the attention term
and the class token ends up no better than uniform attention at finding the lesion. Both
arms of one fold were saved with weights, so the two trained models can be compared
directly on the same held-out data. The stored checkpoint is the early-stopping selection
— `best_epoch=34` of 50 logged epochs for the control, 24 of 40 for AMG — so these are the
weights that would actually be used for prediction. On that fold's 94 held-out slices from
29 patients:

| | uniform-attention chance | control (λ=0) | AMG (λ=3) |
|---|---|---|---|
| slice level | 0.156 | 0.143 | 0.932 |
| patient level | 0.195 | 0.172 | 0.969 |

AMG beats the control on 91/94 slices and 29/29 patients. The control lands *at or below*
the uniform-attention level on both scales, so AMG is not merely accelerating something the
model would have reached on its own.

> **Reading `attention_loc` in `history.csv`.** The AMG runs log this column, and it is
> `-log M`, so `exp(-attention_loc)` recovers the mass. Two traps. It is a *training-set*
> average over each epoch, not a held-out measurement, so it is not comparable to the table
> above. And the epoch-0 row already includes a full epoch of gradient steps with the term
> active, so it does not measure an untrained encoder — nothing in the grid does. The
> control never logs the column at all, because at `λ=0` the term is never computed. Use
> the checkpoint comparison above for any claim about where attention ends up.

Reproduce with `scripts/attention_mass_comparison.py` (see below). Single fold only,
because that is where checkpoints exist.

**The method supervises that directly.** Let `α ∈ Δ^P` be the class token's attention
over the `P = 196` patch tokens of the final block, renormalised to sum to one, and let
`m_p ∈ [0,1]` be the fraction of patch `p` covered by the lesion bounding box
(average-pooled to the 14×14 patch grid). Attention Mass Guidance adds

```
L_AMG = -log( Σ_p α_p · m_p )
```

to the objective. It drives attention mass inside the lesion from **0.160 to 0.984**.

### Why *mass* and not a map

This is the part that matters, and it is why the obvious formulations fail. `α` is
softmax-normalised over 196 patches, so its entries are ≈ 0.005, while the mask is 0/1.

- **Dice** against the mask has numerator ≈ `2·0.005·k` against denominator ≈ `1+k`, so
  the index is pinned near zero and the loss sits near 1 almost regardless of where the
  model attends. It cannot express "correct".
- **MSE** against the mask has its minimum at `α_p = m_p`, which the simplex constraint
  forbids, so the gradient never vanishes even for perfect localisation.

Comparing a **scalar mass** against a scalar target puts both sides on the same scale
and makes the optimum reachable. The loss is zero exactly when all attention mass sits
on fully covered patches.

### What it buys

Against an otherwise identical control — same backbone, same LoRA adapters, same folds,
same four training seeds, `λ_AMG = 0` — on patient-level macro AUC, the pre-specified
primary endpoint:

| metric | proposed | control | Δ | p |
|---|---|---|---|---|
| **macro AUC** | **0.7308** | 0.6733 | **+0.0575** | **0.007** |
| macro-F1 | 0.4378 | 0.3889 | +0.0489 | 0.120 |

45 of 50 folds favour the method on AUC, and the effect is ≈6× the measured
between-seed standard deviation. The gain is concentrated in the three rare clusters
(+0.062 mean F1) and absent on the majority class (+0.011). Only the primary endpoint
reaches significance; the secondary metrics are consistent in direction but not
significant, and the method is not significantly better than the transformer baselines
on macro-F1.

The supervision weight `λ` matters only up to a point: performance rises to λ≈1 and is
then **flat from λ=1 to λ=300**, a 300-fold range within 1.1× the noise floor. There is
no hyperparameter to tune.

**The annotation budget is the class label and the lesion bounding box.** Nothing else.
The box was already needed for the crop, so AMG adds no annotation cost over any
baseline here.

---

## Repository layout

```
ppgl/
  cli.py              every entry point (doctor, inventory, grid, stats, tables, figures, ...)
  config.py           nested dataclass config, YAML loading with _base_ inheritance, CLI overrides
  labels.py           configurable target label space (4-class default, plus binary/3-class)
  engine.py           training loop, early stopping, TTA, prediction
  losses.py           cross-entropy variants, Tversky, logit adjustment, AttentionLocalisationLoss
  metrics.py          one definition of every metric, applied to every model
  mixing.py           mixup / CutMix
  scheduler.py        warmup + cosine, layer-wise LR decay
  data/
    inventory.py      build the manifest from the archive (source of truth for all counts)
    dataset.py        slice reading, cropping, the torch Dataset, lesion-mask construction
    transforms.py     image transforms, and the paired image+mask pipeline
    splits.py         patient-grouped stratified CV
    bags.py           patient-level bags for the MIL comparator
    radiomics.py      hand-crafted 2D features for the radiomics comparator
    clinical.py       clinical covariate loading with per-fold preprocessing
    location.py       anatomical region labels (see the note below)
  models/
    names.py          model registry and capability predicates, torch-free
    backbones.py      timm backbone factory, bundled-checkpoint loading
    attn_lora.py      THE PROPOSED MODEL: ViT + LoRA with supervised class-token attention
    lora.py           minimal LoRA for nn.Linear
    baselines.py      ResNet-101, ViT, ViT+SupCon, Swin, two-branch reimplementation, aux-head
    routed.py         anatomy-routed mixture of experts (comparison model)
    mil.py            gated-attention MIL over a patient's slices (comparison model)
    fusion.py         image + clinical late fusion, and a clinical-only net
    tabular.py        location-only, region-prior, radiomics+GBM, clinical controls
  experiments/
    dispatch.py       job dispatcher, one worker per visible GPU
    run_cv.py         deep-model CV runner
    run_tabular.py    non-deep CV runner
    results.py        on-disk results store, config fingerprinting, resume
    stats.py          corrected resampled t-test + Holm
    ensemble.py       post-hoc probability ensembling
    selective.py      risk-coverage / deferral analysis
    routing_analysis.py  analysis for the routed comparison model
    tables.py         CSV + LaTeX tables
    figures.py        publication figures, including the attention comparison
configs/              YAML experiment definitions, one per reported row
scripts/
  attention_mass_comparison.py  control-vs-AMG attention mass from stored checkpoints
packaging/build_zip.sh  builds a self-contained release archive, with correctness gates
run_all.sh            the whole pipeline end to end
```

---

## Models

All comparison models are retained and runnable. `attn_lora` is the proposed method;
everything else is a comparator or a control.

### Deep models

| name | what it is | config |
|---|---|---|
| `attn_lora` | **proposed**: ViT-B/16 + LoRA, class-token attention supervised by the lesion box | `experiments/40_amg.yaml` |
| `vit` | ViT-B/16, full fine-tune | `experiments/11_vit.yaml` |
| `vit_cl` | ViT-B/16 + supervised contrastive head | `experiments/12_vit_cl.yaml` |
| `swin` | Swin-T | `experiments/13_swin.yaml` |
| `resnet101` | ResNet-101 | `experiments/10_resnet101.yaml` |
| `ppgl_transformer` | two-branch reimplementation of prior PPGL work | `experiments/14_ppgl_transformer.yaml` |
| `single_head_aux` | one classifier + auxiliary region head | `experiments/20_single_head_aux.yaml` |
| `routed` | anatomy-routed mixture of experts, 5 routing variants | `experiments/3{0..4}_routed_*.yaml` |
| `mil` | gated-attention MIL over a patient's slices | `experiments/21_mil.yaml` |
| `fusion` | image + clinical late fusion, with modality dropout | `clinical/81_fusion.yaml` |
| `clinical_net` | clinical-only neural control | `clinical/82_clinical_net.yaml` |

### Non-deep controls

| name | what it is | config |
|---|---|---|
| `radiomics_gbm` | hand-crafted 2D radiomics + gradient boosting | `experiments/02_radiomics_gbm.yaml` |
| `radiomics_gbm_region` | the same, plus region as a feature | `experiments/03_radiomics_gbm_region.yaml` |
| `location_only` | lesion position and size only — the "is it just anatomy?" control | `experiments/01_location_only.yaml` |
| `region_majority` | per-region class prior — a trivial floor | `experiments/00_region_majority.yaml` |
| `clinical_only` | clinical covariates + logistic regression | `clinical/80_clinical_only.yaml` |
| `clinical_radiomics` | clinical + radiomics features | `clinical/83_clinical_radiomics.yaml` |

### Config families under `configs/attention/`

| family | varies | purpose |
|---|---|---|
| `A1`–`A3` | `runtime.seed` | between-seed noise floor, proposed arm |
| `E1`–`E3` | `runtime.seed` | between-seed noise floor, control arm |
| `B1`–`B7` | `loss.lambda_attention` | λ sweep: 0.1, 0.3, 1, 10, 30, 100, 300 |
| `C1`–`C4` | one mechanism each | relaxations: target 0.7, best-head, two-stage decay, rollout |
| `D1` | `model.finetune` | full fine-tuning, completing the adaptation × supervision 2×2 |
| `9*` | — | **superseded.** An earlier variant of the proposed model that carried an auxiliary region head. Kept for the development record; do not run or cite. See `configs/attention/README.md`. |

A seed replicate must be its own config file with a distinct `experiment:` value.
`runtime` is excluded from the config fingerprint, so `--set runtime.seed=N` changes
neither the fingerprint nor the run directory: the folds resolve to the already-complete
directory and every job reports `skipped`, with no work done and no error.

---

## Getting started

Full environment notes are in `INSTALL.md`. In short:

```bash
pip install -r requirements.txt
# put the archive under detection_data_NIH/original/ (see the placeholder file)
python -m ppgl.cli doctor            # environment + data check
python -m ppgl.cli fetch-weights     # pretrained backbones, if the host has internet
```

**Pretrained weights are not included here** — they are checkpoints, not code, and come
to about a gigabyte. `fetch-weights` pulls them into a local `weights/` cache. For a host
with no route to the Hugging Face hub, fetch them on a machine that does and build a
self-contained archive, which bundles `weights/` and verifies its checksums:

```bash
./packaging/build_zip.sh              # code + weights, ~1 GB
./packaging/build_zip.sh --no-weights # code only, ~270 KB
```

`--verify` on either extracts the archive and re-runs a set of correctness gates against
the clean copy — that every module imports, that every reported config resolves without
region labels, and that a config survives the serialisation round-trip the grid workers
depend on. Worth running before shipping anything.

Then the whole pipeline:

```bash
./run_all.sh                 # baselines, ablations, analysis, tables, figures
./run_all.sh --amg           # ... plus the seed replicates and the λ sweep
./run_all.sh --smoke         # wiring check in a few minutes, no real results
```

Or a single experiment, or a grid:

```bash
python -m ppgl.cli train --config configs/experiments/40_amg.yaml
python -m ppgl.cli grid --configs configs/experiments/*.yaml --output-dir results
```

GPU count is discovered at runtime; the dispatcher runs one worker per visible GPU and
pins it. Completed folds are skipped, so re-running resumes. Scope a run to specific
devices with `CUDA_VISIBLE_DEVICES`; several independent grid invocations can share one
output directory safely because each experiment writes only to its own `runs/<name>/`.

---

## Evaluation protocol

- **5 repeats × 10-fold patient-grouped stratified CV** = 50 folds, seed 1337, identical
  for every model, so all comparisons are paired.
- Patient identity is `{CLASS}_{NUMERIC_ID}`. Numeric IDs are reused across class
  directories (84 collisions in this archive), so grouping on the bare number silently
  merges different patients. **Slice-level splitting leaks badly**: 76% of slices belong
  to a multi-slice patient, and under 10-fold slice-level CV 73.5% of each held-out fold
  shares a patient with training.
- Patient-level prediction is the mean of that patient's slice probabilities. Patient
  level is the primary reporting level throughout.
- **Statistics**: corrected resampled t-test (Nadeau & Bengio 2003) over the 50 folds,
  with Holm–Bonferroni across families of comparisons. A naive paired t-test is not used:
  the folds share training data and it has inflated false-positive rates.
- Every fold writes its fully resolved config, per-slice predicted probabilities, and
  training history to disk, so any number can be traced to the settings that produced it.

---

## Reproducing the reported results

```bash
# the proposed method, its control, seed replicates, and the λ sweep
python -m ppgl.cli grid --configs \
  configs/experiments/40_amg.yaml configs/experiments/41_amg_control.yaml \
  configs/attention/A*_amg_seed_*.yaml configs/attention/E*_amg_control_seed_*.yaml \
  configs/attention/B*_amg_lambda_*.yaml \
  --output-dir results

# the ablations
python -m ppgl.cli grid --configs \
  configs/attention/C*_amg_*.yaml configs/attention/D1_amg_full_finetune.yaml \
  --output-dir results

# analysis
python -m ppgl.cli stats  --output-dir results --reference amg \
  --others amg_control,vit,swin,vit_cl,resnet101,radiomics_gbm
python -m ppgl.cli tables  --output-dir results
python -m ppgl.cli figures --output-dir results
```

`--others` is named explicitly on purpose: its default is every experiment in the store,
which would pull comparison rows that depend on region labels into the table.

### The attention comparison figure

This one needs trained weights, and weights are not persisted by default — a ViT-B/16
state_dict is ~340 MB and the full grid is over a thousand folds. Retrain one fold of
each arm with `runtime.save_checkpoint=true`, into a **fresh** output directory
(`runtime` is excluded from the fingerprint, so pointing it at a finished store would
just skip):

```bash
python -m ppgl.cli grid \
  --configs configs/experiments/40_amg.yaml configs/experiments/41_amg_control.yaml \
  --repeats 0 --folds 0 --set runtime.save_checkpoint=true \
  --output-dir results_ckpt

python -m ppgl.cli attention-comparison --output-dir results_ckpt \
  --proposed-checkpoint results_ckpt/runs/amg/folds/r0f0/model.pt \
  --control-checkpoint  results_ckpt/runs/amg_control/folds/r0f0/model.pt
```

It renders three rows — one per rare cluster — showing the crop with the annotated box,
the control's class-token attention, and the proposed model's. Examples are selected by
a stated rule rather than by eye: within each cluster, among that fold's **held-out**
slices, the slice whose attention-mass gain is the **median** for that cluster. The
quantity plotted is the one the loss supervises, read from the model's own output, not
Grad-CAM or a saliency proxy.

### Attention mass, control vs AMG

The figure above shows three slices. To get the same comparison over *every* held-out
slice of the fold — which is where the "training does not fix it" number comes from —
run the same checkpoints through:

```bash
python scripts/attention_mass_comparison.py \
  --output-dir results_ckpt \
  --proposed-checkpoint results_ckpt/runs/amg/folds/r0f0/model.pt \
  --control-checkpoint  results_ckpt/runs/amg_control/folds/r0f0/model.pt \
  --csv results_ckpt/attention_mass_r0f0.csv
```

It reuses `figures._patch_mass`, so the mass definition is the one the loss optimised
rather than a second implementation that could drift from it, and it inherits the
fold-fingerprint check that refuses an unpaired comparison. Beyond the figure it also
computes the **per-slice** chance level (uniform attention gives mass equal to the mean
of the pooled mask), so "below chance" is exact per slice instead of resting on a cohort
median. Per-slice values land in the CSV.

No p-value is reported: slices from one patient are correlated, and a single fold cannot
feed the corrected resampled t-test used elsewhere. To make this a 50-fold claim, log
`attention_loc` in the control arm rather than persisting 50 more 337 MB checkpoints —
though note that gives training-set mass, so a held-out figure needs the metric added to
the eval path.

---

## A note on anatomical region labels

The archive provides a class name and per-lesion bounding boxes. It contains **no
anatomical region annotation**. Every "region" label in this project came from an
unvalidated automatic estimator whose per-slice accuracy is unknown.

The proposed method does not use region labels: `40_amg.yaml` and everything inheriting
from it set `model.aux_region: false`, `loss.lambda_region: 0` and
`data.location_source: none`, so the dependency is absent by construction rather than by
convention. `packaging/build_zip.sh` re-checks this on every build and refuses to package
if it fails.

Several comparison models — `routed`, `single_head_aux`, `ppgl_transformer`, `mil`,
`location_only`, `region_majority`, `radiomics_gbm_region` — *do* consume those labels.
Their code is retained and runnable, but a result that depends on provisional labels
cannot support a claim, so their numbers are not reported. The mechanical test is whether
a run's `history.csv` carries a `region_ce` column.

If radiologist-reviewed labels ever become available, supply them and the same runs
become reportable:

```bash
python -m ppgl.cli locations --review-csv results/region_review.csv   # emit for review
./run_all.sh --location-csv reviewed.csv                             # re-run with them
```

---

## Data expectations

| | |
|---|---|
| slices | 630 |
| patients | 287 |
| studies | 296 |
| slices per patient | min 1, median 1, mean 2.20, max 22 |
| input | 8-bit pre-windowed RGBA TIFF, 512×512, single axial venous-phase slice |
| annotation | per-lesion bounding boxes in XML — no segmentation contours |

| cluster | slices | patients |
|---|---|---|
| SDHx | 353 | 142 |
| Sporadic | 104 | 51 |
| Kinase signalling | 99 | 55 |
| VHL/EPAS1 | 74 | 39 |

Because the images are pre-windowed 8-bit with no HU calibration, radiomics intensity
features here are **not** comparable to published HU-based radiomics.

`python -m ppgl.cli inventory` builds the manifest, which is the single source of truth
for every count above.

---

## Implementation details worth knowing

**Reading the attention at all.** This is the one non-obvious step in the whole method,
and it is worth spelling out because AMG is undefined without it. timm's `Attention`
routes through `F.scaled_dot_product_attention` whenever `fused_attn` is set. That kernel
is fast precisely because it never writes the `(B, heads, N, N)` attention matrix to
memory — it fuses the softmax and the value product in one pass, which is what turns the
quadratic memory cost into a linear one. No forward hook can observe a tensor that is
never created, so with the fused path enabled there is nothing for the loss to read.

Setting `attention.fused_attn = False` restores the explicit
`softmax → attn_drop → @v` path, and the post-softmax matrix then exists as a real
tensor. A forward hook on `attn_drop` reads it as `inputs[0]`:

```python
if hasattr(attention, "fused_attn"):
    attention.fused_attn = False
attention.attn_drop.register_forward_hook(self._make_hook(position))
```

Hooking the *input* to `attn_drop` rather than its output is deliberate: it yields the
true attention distribution rather than a dropout-perturbed sample of it.

The explicit path is slower and costs quadratic memory, so only the blocks actually being
observed are switched — the final block alone in the default configuration, leaving the
other eleven on the fast kernel. This is also why the rollout relaxation (`C2`) is the
expensive variant: it watches the last six blocks, so six of them lose the fused kernel
and the gradient runs through the whole product.

This detail was cut from the ISBI manuscript. It is an instrumentation constraint rather
than a modelling choice, and the paper describes the model; this README is where it
lives.

**The mask must follow the image.** `RandomAffine` and `RandomHorizontalFlip` draw their
parameters inside `__call__`, so running two independent pipelines would desynchronise
image and mask silently. `data/transforms.py` provides a paired pipeline that draws the
parameters once and applies them to both (bilinear for the image, nearest for the mask).

**Per-fold preprocessing.** The clinical preprocessor is fit on each fold's training rows
only; category vocabularies come from the declared schema rather than the data, so a
category absent from one fold still occupies a stable column position.

**Resume is fingerprinted.** A fold is reused only if its stored config fingerprint
matches. Bookkeeping fields (`output_dir`, `experiment`, and all of `runtime`) are excluded
so that changing the GPU count does not invalidate results, but anything that changes what
is computed does.

**Thread oversubscription.** Worker processes pin `OMP_NUM_THREADS` and the tabular
estimators run with `n_jobs=1`. Without both, a sub-second radiomics fit took 27 minutes
on a 96-core host.
