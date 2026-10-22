# Attention-supervision sweep

> ## ⚠️ Two generations of config live in this directory
>
> | family | base | region labels | status |
> |---|---|---|---|
> | `9*.yaml` | `_shared.yaml` | **consumed** (`lambda_region=0.1`) | **superseded — do not run, do not cite** |
> | `A*`, `B*`, `C*`, `D*` | `../experiments/40_amg.yaml` | none | **current — these are the reported runs** |
>
> The `9*` sweep is retained only so the audit trail is inspectable. Its results are
> in `results_lora/` and are documented, with a full contamination notice, in
> `RESULTS.md`. Anatomical region labels **do not exist in this archive**; the ones
> those rows consumed came from an unvalidated automatic estimator, so no `9*` row can
> support a claim.
>
> **Consequently `--configs configs/attention/*.yaml` is no longer safe** — it mixes
> the generations. Name the families explicitly. The current run commands are at the
> bottom of `RESULTS.md` and in `PAPER_PLAN.md` §0.1.

## The current (region-free) families

Every one of these inherits `../experiments/40_amg.yaml`, which sets
`model.aux_region: false`, `loss.lambda_region: 0` and `data.location_source: none`,
so region-independence is inherited rather than restated per file.

| family | files | base | varies | purpose |
|---|---|---|---|---|
| `A1`–`A3` | 3 | `40_amg` | `runtime.seed` | noise floor at λ=3.0 on the proposed arm, folds held fixed |
| `B1`–`B5` | 5 | `40_amg` | `loss.lambda_attention` | λ sweep at 0.1/0.3/1/10/30 (λ=3 is `40_amg`, λ=0 is `41_amg_control`) |
| `C1`–`C4` | 4 | `40_amg` | one mechanism each | relaxations of the constraint, at λ=3.0 |
| `D1` | 1 | `40_amg` | `model.finetune: llrd` | full-fine-tune cell of the adaptation × supervision 2×2 |
| `E1`–`E3` | 3 | `41_amg_control` | `runtime.seed` | noise floor on the **control** arm, so the headline comparison is 4 seeds against 4 rather than 4 against 1 |

`A*` + `B*` are the core run (10 configs with `40_amg` and `41_amg_control`, 500 folds).
`E*` is the highest-value addition (150 folds). `C*` + `D1` are the ablation extension
(250 folds).

## Why seed replicates are files, not `--set` overrides

`E1`–`E3` and `A1`–`A3` differ from their base **only** in `runtime.seed`, so it is
tempting to drop the files and pass `--set runtime.seed=7` instead. That does not work,
and it fails silently:

- `results.config_fingerprint()` deliberately drops the whole `runtime` section, so the
  seed does not change the fingerprint.
- `--set` does not change `experiment`, so the run resolves to the same directory.

The resume check therefore finds every fold already complete and the grid reports
`{'skipped': 50}` with no error and no work done. A distinct `experiment:` value is what
makes a seed replicate a separate row. `packaging/build_zip.sh` now fails the build if
any two eligible configs share an experiment name.

---

## The superseded `9*` sweep (historical)

Seven single-change ablations against `configs/experiments/28_attn_lora.yaml`, the
row that scored patient macro-F1 0.4312 on the 4-class target. Every `9*` config
inherits `_shared.yaml`, which pins that row's exact settings, and changes **one**
thing. All of them, and `28_attn_lora.yaml` itself, carry the auxiliary region head.

They lived outside `configs/experiments/` on purpose: they were a sweep, not result
rows, and adding seven more experiments to the default grid would have made every
full run 30% longer.

`_shared.yaml` is skipped automatically when globbing: `--configs` drops glob matches
whose filename starts with an underscore, since a shared fragment is not a runnable
experiment.

The rationale sections below are kept because the *reasoning* about each mechanism
still applies — it is what motivated the region-free `C1`–`C4` replacements. Only the
numbers are unusable.

## The rows

| config | change | why |
|---|---|---|
| `90_lambda_0p1` | λ 0.3 → 0.1 | weaker supervision |
| `91_lambda_1p0` | λ 0.3 → 1.0 | stronger |
| `92_lambda_3p0` | λ 0.3 → 3.0 | deliberately too strong, to bracket the curve |
| `93_target_0p7` | require 70% of mass inside, not 100% | the target is a *box*, ~20% background |
| `94_rollout` | supervise attention rollout over the last 6 blocks | raw single-layer attention is a weak proxy for information flow |
| `95_heads_max` | only the best head must localise | preserve head diversity |
| `96_warmup` | decay λ to 0 over the first half of training | two-stage schedule from the closest prior art |

## Why these four mechanisms

**Target mass (`93`).** The measured run drove attention mass inside the lesion
from 12% to **98%**. Chance is ~20%, since the lesion covers about a fifth of the
frame at `roi_context`. 98% is far past the point where the constraint is changing
behaviour, and because the annotation is a bounding box rather than a contour, it
is forcing attention onto background inside the box and forbidding peritumoral
context. `attention_target=0.7` makes the loss exactly zero once a row is
satisfied while still applying a large gradient to a row at chance level.

**Rollout (`94`).** Self-attention mixes information across layers and the
residual stream carries information around each attention operation, so the final
block's raw attention is not a reliable estimate of which patches the class token
depends on. Attention rollout — multiplying the per-block matrices after mixing in
the identity — correlates substantially better with genuine input importance
([Abnar & Zuidema, ACL 2020](https://arxiv.org/abs/2005.00928)). Supervising
rollout therefore constrains where the class token's information actually comes
from. Content rephrased for compliance with licensing restrictions.

**Best head (`95`).** Averaging 12 heads and pushing the average onto a
20%-of-frame target constrains every head. Heads specialise, and some useful
behaviour is global. Taking the per-head mass and using the maximum is provably a
weaker demand, so one head must localise and the rest stay free.

**Two-stage (`96`).** The closest prior art applies a lesion-focused attention
loss on last-layer attention from expert bounding boxes as a *pretraining*
objective, then fine-tunes with cross-entropy alone
([Di Stefano et al., 2026](https://pubmed.ncbi.nlm.nih.gov/41581326/)). Once
attention is on the lesion, continuing to constrain it only limits the classifier.
Content rephrased for compliance with licensing restrictions.

## One diagnostic worth reading first

Initial attention mass inside the lesion was **12%**, *below* the ~20% chance
level. That is consistent with the high-norm artifact tokens described in
[Darcet et al., "Vision Transformers Need Registers" (2024)](https://arxiv.org/abs/2309.16588):
pretrained ViTs repurpose low-information background patches for internal
computation, and those tokens attract attention away from the subject. Content
rephrased for compliance with licensing restrictions.

If that is what is happening here, part of what the supervision buys is simply
undoing an artifact rather than teaching anatomy — worth a sentence in the paper,
and worth checking on the attention maps. A register-token backbone
(`vit_base_patch14_reg4_dinov2`) would test it directly, but it needs a checkpoint
we do not currently bundle.

## What to report

Per row: patient macro-F1, MCC, macro AUC, **per-class F1**, and the final
attention mass (`exp(-attention_loc)` from `history.csv`). The per-class column is
the one that matters — the effect in the measured run was concentrated in the
three rare clusters (+0.046 to +0.067) rather than in SDHx (+0.028), and an
improvement that only lifts SDHx is not the improvement this paper needs.
