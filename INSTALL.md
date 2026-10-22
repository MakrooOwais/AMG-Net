# Setup and run

Unzip, install, point it at the data, run. Five steps.

---

## 1. Unzip

```bash
unzip ppgl_isbi2027_code.zip
cd ppgl_isbi2027
```

---

## 2. Install dependencies

Python 3.9 or newer. Install torch **first**, matching the server's CUDA, then
the rest.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# torch: pick the line matching the server's CUDA (check with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.1:  --index-url https://download.pytorch.org/whl/cu121
# CPU only:   --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

If the server already has a working torch (a module load, a conda env, a
container), use it and skip the torch line — `requirements.txt` lists
`torch>=2.1` but will not upgrade a satisfying install. Do not force-reinstall
torch over a vendor build.

`lightgbm`, `scikit-image` and `statsmodels` are optional. Without them the
radiomics baseline falls back to sklearn's `HistGradientBoostingClassifier`,
texture features return zeros, and Holm correction uses the built-in
implementation. Everything still runs.

### Pretrained weights are already included

`weights/` ships inside this archive, so **nothing is downloaded at run time** and
the deep models work on a host with no route to the Hugging Face hub:

| file | size | used by |
|---|---|---|
| `swin_tiny_patch4_window7_224.ms_in1k.safetensors` | 109 MB | `swin`, `single_head_aux`, all `routed_*` (the proposed model) |
| `vit_base_patch16_224.augreg2_in21k_ft_in1k.safetensors` | 330 MB | `vit`, `vit_cl`, `ppgl_transformer` |
| `resnet101.a1h_in1k.safetensors` | 170 MB | `resnet101` baseline |
| `vit_base_patch14_dinov2.lvd142m.safetensors` | 330 MB | `encoder_dinov2` (optional axis) |
| `resnet50.a1_in1k.safetensors` | 98 MB | `encoder_resnet50` (optional axis) |

That is every checkpoint the configs reference, ~1.04 GB, so both the main grid and
`--optional` run without touching the network.

Confirm they survived the transfer:

```bash
python -m ppgl.cli fetch-weights --verify-only     # checks SHA256SUMS
python -m ppgl.cli doctor --config configs/base.yaml
```

If `weights/` ends up somewhere else, point at it rather than moving it:

```bash
export PPGL_WEIGHTS_DIR=/mnt/efs/ppgl_weights
# or per run
./run_all.sh --set model.weights_dir=/mnt/efs/ppgl_weights
```

The one config that still needs a file you must supply yourself is
`configs/optional/41_encoder_radimagenet.yaml`. RadImageNet is distributed as
Keras/TF and has to be converted to a torch `state_dict`; the bundled
`resnet50.a1_in1k` is **ImageNet, not RadImageNet**, and does not substitute for it.
That config fails fast until the file it names exists. For a ResNet-50 encoder row
that runs today, use `42_encoder_resnet50.yaml` — same architecture, so the pair
becomes a clean ImageNet-vs-radiology comparison once you have the converted
weights.

Only as a last resort, you can run with randomly initialised backbones. It
exercises the pipeline but is **not publishable** — 28M parameters trained from
scratch on 630 slices learn nothing useful:

```bash
./run_all.sh --set model.allow_random_init=true
```

Without that flag a missing checkpoint fails loudly and immediately rather than
degrading silently, and the error names the fix.

---

## 3. Place the data

The code expects this layout, relative to the project root:

```
detection_data_NIH/
  original/
    Slices/{CLASS}_{ID}/{ID}_{STUDY}_Slice_{N}.tiff
    XMLs/  {CLASS}_{ID}/{ID}_{STUDY}_Slice_{N}.xml
```

**No data is distributed with this repository, and none will be.** The cohort is
retrospective human-subjects CT acquired at the NIH under an approved institutional
review board protocol; access is governed by that protocol, not by this repository.
The steps below are for running the pipeline against an equivalently structured
cohort of your own.

Point the loader at your copy by symlinking it:

```bash
ln -s /path/to/your/cohort detection_data_NIH
```

or by overriding the root and leaving the data where it is:

```bash
./run_all.sh --set data.root=/path/to/your/cohort/original
```

Only `original/` is read. If your layout has sibling `train/` and `test/`
directories, they are deliberately ignored: a flat split that strips the class
prefix cannot distinguish two patients who share a numeric ID, and numeric IDs are
reused across classes in this cohort (84 collisions). The patient key is
`{CLASS}_{ID}`, never `{ID}` alone.

---

## 4. Verify before committing GPU time

```bash
python -m ppgl.cli doctor --config configs/base.yaml
```

Prints torch version, CUDA version, every visible GPU with its memory, how many
fold-level workers would be used, and whether the data is where it should be.

Then a full end-to-end wiring check on a tiny subset — a few minutes, no valid
results, just proof that every stage runs:

```bash
./run_all.sh --smoke
```

Expect it to finish with a `DONE` banner and populate `results/tables/`. If it
does, the pipeline is sound.

---

## 5. Run

```bash
./run_all.sh
```

With radiologist region labels (see the warning below):

```bash
./run_all.sh --location-csv results/region_review.csv
```

Useful flags: `--n-parallel N` caps concurrent workers, `--optional` adds the
encoder and fine-tuning axes, `--clinical-csv FILE` adds the clinical and fusion
rows, `--label-space X` runs the whole grid on a coarser target (into its own
output directory, so the 4-class run is not overwritten), `--from N` resumes at a
stage, `--output-dir DIR` relocates output. Any config value can be overridden:
`--set optim.epochs=40 cv.n_repeats=3`.

Stage numbers changed when the clinical, ensemble and selective stages were
added — there are now 14. Check the list at the top of `run_all.sh` before using
`--from`.

Re-running resumes: completed folds are skipped, and the skip is config-aware, so
changing a setting recomputes the affected folds instead of silently reusing
stale ones.

### Long runs

`run_all.sh` will run for hours. Detach it:

```bash
nohup ./run_all.sh > run.out 2>&1 &
tail -f run.out
# or
tmux new -s ppgl './run_all.sh'
```

---

## Read these first, in this order

```
results/runs/location_only/summary.json      the decision gate
results/analysis/routing_routed_soft.json    the routing gap
results/tables/main_results.csv              the headline table
results/data_qa.json                         data-quality report
```

**The decision gate matters more than anything else in the run.** The
location-only model sees only which of three anatomical regions the lesion is in
— no image content at all. It runs at stage 5, before any GPU time is spent. If
it comes close to the deep models, the paper's premise does not hold and the
framing has to change; see `ISBI2027_plan.md` §8.

---

## One warning that affects every reported number

**Anatomical region labels do not exist in this archive.** The XMLs carry only a
class name and bounding boxes. The routed model, the location-only control and the
entire routing analysis depend on region labels.

Without `--location-csv`, a provisional heuristic estimates them and every result
is stamped `provisional`. Its marginal distribution was calibrated to approximate
the published 506 / 80 / 64 split, but per-slice accuracy is unknown and head &
neck is unreliable. To get real labels:

```bash
python -m ppgl.cli locations --review-csv results/region_review.csv
# a radiologist fills the radiologist_region column, then
./run_all.sh --location-csv results/region_review.csv
```

The review file is sorted lowest-confidence-first, so correcting the top of the
file buys the most.

---

## Rough runtime

Estimates, not measurements — scale to the actual hardware. Assumes 630 slices,
~570 training slices per fold, batch 16, up to 60 epochs with early stopping,
and 5 × 10 = 50 folds per experiment.

| | 1 modern GPU | 4 GPUs | 8 GPUs |
|---|---|---|---|
| Stages 1–5 (inventory, folds, gate) | ~5 min | ~5 min | ~5 min |
| Stage 6 (17 deep experiments × 50 folds) | ~2–3 days | ~12–18 h | ~6–9 h |
| Stage 7 (clinical, 4 experiments) | ~4–8 h | ~1–2 h | ~40–60 min |
| Stages 9, 12–14 (analysis, stats, tables, figures) | ~10 min | ~10 min | ~10 min |
| Stages 10–11 (ensembles, selective) | ~1 min | ~1 min | ~1 min |

Stage 6 grew from 11 to 17 deep experiments with the addition of the MIL row and
the five Tier-3 rows, so budget roughly 1.5× the earlier estimate.

Stages 1–5, 9 and 12–14 are CPU-bound and do not parallelise across GPUs. Stages
10 and 11 only read stored probabilities, so they are effectively free and can be
re-run whenever. ViT-B backbones cost roughly twice a Swin-T fold. MIL costs about
the same as a slice model per epoch despite processing bags, because each patient
appears once per epoch and mean bag size is only 2.2. TTA doubles inference cost,
which is a small share of the total; mixup adds a second loss evaluation but no
extra backbone pass.

To shrink a first pass: `--set cv.n_repeats=1` cuts stage 6 fivefold (at the cost
of noisier estimates and weaker paired tests), and `--set optim.epochs=30` roughly
halves it again.

---

## Troubleshooting

**`no pretrained weights available for '...'`, every deep model failing while the
tabular controls succeed.** The bundled `weights/` directory was not found.
`python -m ppgl.cli doctor` prints the directory it looked in and which files are
missing. Usually the archive was unpacked without `weights/`, or
`PPGL_WEIGHTS_DIR` points somewhere stale.

**`bundled checkpoint ... could not be loaded into ...`** in the log, followed by a
hub attempt. The file is present but does not match the architecture — a truncated
transfer, or a checkpoint bundled under the wrong name. Run
`python -m ppgl.cli fetch-weights --verify-only`, and re-fetch with `--force` if a
checksum fails.

**CUDA out of memory.** `--set optim.batch_size=8 optim.grad_accum_steps=2` keeps
the effective batch size while halving peak memory.

**`no CUDA device is visible`** from `runtime.device=cuda`. Either CUDA is genuinely
absent or `CUDA_VISIBLE_DEVICES` is empty. Leave `runtime.device=auto` and it
falls back to CPU on its own.

**`LoRA targets ... matched no nn.Linear layer`.** The backbone names its attention
projections differently. Inspect them and set `model.lora_targets` accordingly:

```python
import timm
m = timm.create_model("swin_tiny_patch4_window7_224")
print([n for n, _ in m.named_modules() if "attn" in n][:20])
```

**`manifest not found`.** Run `python -m ppgl.cli inventory --config configs/base.yaml`,
or just `./run_all.sh`, which does it at stage 2.

**A fold fails but the grid continues.** By design — one bad configuration should
not discard hours of work. Failures are logged as `FAILED <experiment>/rNfM` with
a traceback in that fold's `log.txt`. Re-run the same command; completed folds are
skipped and only the failures are retried.

**`refusing to compare X with Y: fold fingerprints differ`.** The two experiments
were evaluated on different splits, so a paired test would be invalid. Re-run the
lagging one, or pass `--allow-mismatched-folds` if you accept the caveat.

**Workers exit immediately with a CUDA error.** Something initialised CUDA before
the worker forked. The dispatcher uses the `spawn` start method to avoid this; if
you have wrapped it in your own multiprocessing, keep `spawn`.
