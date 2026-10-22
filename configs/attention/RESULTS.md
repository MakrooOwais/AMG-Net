# Sweep 1 results and what they change

> # ⚠️ SUPERSEDED — EVERY NUMBER ON THIS PAGE CONSUMES REGION LABELS
>
> **Do not report any figure from this document.** Every row in `results_lora/` was
> trained with an auxiliary region-classification head (`lambda_region = 0.1`) active
> in all 50 folds. The anatomical region labels it consumed **do not exist in this
> archive** — they came from an unvalidated automatic estimator, calibrated only so
> that its marginal distribution approximates a published 506/80/64 split. Per-slice
> accuracy is unknown.
>
> Verify for yourself:
> ```bash
> head -1 results_lora/runs/attn_lam_3p0/folds/r0f0/history.csv | grep -o region_ce
> ```
> A hit means the row is contaminated. Every run in this store hits.
>
> **What survives.** The attention arm and its control were contaminated *equally*,
> so the measured *comparison* is internally consistent and remains a sound guide to
> the expected **sign and magnitude** of the AMG effect. That is the only thing this
> page is still good for. The methodological conclusions — that λ tightening helps,
> that the four relaxations are neutral, that single runs cannot separate effects from
> run-to-run noise — also stand as reasoning, independent of the exact values.
>
> **Replacement.** Region-free configs: `../experiments/40_amg.yaml` (λ=3.0, the
> proposed method), `../experiments/41_amg_control.yaml` (λ=0), `A1`–`A3` (seed
> replicates), `B1`–`B5` (λ sweep), `C1`–`C4` (relaxations), `D1` (full fine-tune
> cell). All set `model.aux_region: false`, `loss.lambda_region: 0` and
> `data.location_source: none`. See "Next run" at the bottom and `PAPER_PLAN.md` §0.
>
> The `9*.yaml` configs in this directory are retained only so this audit trail is
> inspectable. Do not run them and do not cite them.

Run: `results_lora/`, 50 folds per row, patient-level metrics. **Region-contaminated
— pilot magnitudes only.**

## Read the reference correctly

`attn_lora` (0.4312) is **not** comparable to the sweep rows. The sweep code
renormalises the class-token attention over patches before the loss sees it —
slicing off the prefix tokens breaks the simplex, so without renormalising the
"mass inside the lesion" was expressed as a fraction of attention *including* the
class token's self-attention, and `-log(mass)` could never reach zero. That is a
change of objective at default settings, not a refactor.

`base` is the same settings re-run by the sweep code, so **`base` (0.4033) is the
reference**. Everything below is measured against it.

## The λ curve has not turned over

| λ | patient macro-F1 | MCC | AUC | SDHx | VHL | Kinase | Sporadic |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.3897 | 0.2714 | 0.6992 | 0.695 | 0.270 | 0.300 | 0.294 |
| 0.3 *(ref)* | 0.4033 | 0.2827 | 0.6911 | 0.689 | 0.305 | 0.283 | 0.336 |
| 1.0 | 0.4362 | 0.3279 | 0.7191 | 0.706 | 0.372 | 0.317 | 0.350 |
| **3.0** | **0.4453** | **0.3297** | **0.7357** | 0.698 | 0.368 | **0.384** | 0.332 |

Monotone increasing in macro-F1 and MCC. λ=3.0 was included as a
deliberately-too-strong bracketing point and it **won** — so the optimum is at
λ ≥ 3 and the sweep has to go further up. Hence `97_lambda_10` and `98_lambda_30`.

λ=3.0 is the best row in the whole project on macro-F1 (0.4453 vs `ens_routing`
0.4364) and on AUC (0.7357 vs `ens_tricks` 0.7196), and it has the highest Kinase
F1 anywhere (0.384 vs 0.359). The gains stay concentrated in the rare clusters,
which is the pattern that matters. λ=3.0 vs λ=0.1: d=+0.056, better on 36/50 folds,
p=0.26.

## All four mechanism tweaks are neutral

| variant | Δ vs ref | p |
|---|---|---|
| target 0.7 (hinge) | +0.0068 | 0.90 |
| best-head | +0.0063 | 0.90 |
| two-stage warm-down | +0.0062 | 0.90 |
| rollout (6 blocks) | −0.0186 | 0.71 |

Every one of them was an attempt to *relax* the constraint — cap the demand,
supervise a smoother signal, constrain fewer heads, stop constraining halfway. All
four came out flat or slightly worse, while raising λ (tightening the constraint)
is what moved the number. The consistent reading is that the term is acting as a
**regulariser** on a 630-image problem, not as gentle guidance: the more the
hypothesis space is restricted, the better. Rollout being the weakest fits — it is
the most diffuse target (final mass 0.941 against 0.98–0.99 for the rest).

Drop all four. They cost a config each and buy nothing.

## The finding that gates every claim

Nothing in this sweep is significant. Everything is p ≥ 0.45, p_holm = 1.0. And
the reason is measurable:

`attn_lora` and `base` differ by **+0.028** in mean patient macro-F1, with a
per-fold SD of **0.126** and **0 of 50 folds identical**. Some of that is the
objective change above, but the per-fold spread is pure run-to-run variance — two
independent draws each with fold-level SD ≈ 0.09. That puts the run-to-run standard
error of the 50-fold mean at roughly **0.018**.

Every effect in this sweep is 0.006–0.042. The noise floor is 0.018–0.028. So a
single run per configuration cannot separate an effect from a re-run, and the
apparent ranking of the four mechanism tweaks is not information.

`99a/b/c` are three seed replicates at λ=3.0 — same folds (`cv.seed` untouched),
different training randomness (`runtime.seed`). They measure the floor directly, so
the λ trend can be tested against it instead of against an assumption.

## Next run — region-free, and this is the one that matters

The `results_lora/` sweep is closed. Do not extend it; every additional row would
inherit the same contamination. The replacement run is region-free by construction.

**Core (required before the paper can be written)** — 10 configs, 500 folds, ~3 h:

```bash
python -m ppgl.cli grid --configs configs/experiments/40_amg.yaml \
  configs/experiments/41_amg_control.yaml \
  configs/attention/A*_amg_seed_*.yaml \
  configs/attention/B*_amg_lambda_*.yaml \
  --output-dir results_2
```

**Optional ablation extension** — 5 configs, 250 folds, ~1.5 h, same store so
comparisons stay paired:

```bash
python -m ppgl.cli grid --configs configs/attention/C*_amg_*.yaml \
  configs/attention/D1_amg_full_finetune.yaml --output-dir results_2
```

Note the glob `configs/attention/*.yaml` is **no longer safe** — it would pick up the
superseded `9*.yaml` rows. Name the `A*`/`B*`/`C*`/`D*` families explicitly.

Then confirm cleanliness across the whole store before reading anything off it:

```bash
for f in results_2/runs/amg*/folds/*/history.csv; do
  head -1 "$f" | grep -q region_ce && echo "CONTAMINATED: $f"
done; echo "audit done"
```

Silence means the store is clean.

**What to look for.** If λ=10 or 30 continues to improve, the optimum is further out
and the regulariser reading strengthens. If the seed replicates at λ=3.0 span more
than the λ=1.0→3.0 gap, the curve's upper end is not resolved and the paper should
report the trend with replicate error bars rather than a single best λ. One
difference from the pilot: the `C*` relaxations now run at λ=3.0 rather than 0.3, so
they read against `40_amg` directly — a relaxation of the *reported* method rather
than of an intermediate setting.
