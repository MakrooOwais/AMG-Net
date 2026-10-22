#!/usr/bin/env bash
#
# One script that runs everything, end to end, on the remote server.
#
#   ./run_all.sh                                  full pipeline, all visible GPUs
#   ./run_all.sh --smoke                          wiring check, a few minutes
#   ./run_all.sh --location-csv labels.csv        use radiologist region labels
#   ./run_all.sh --clinical-csv clinical.csv      also run the clinical + fusion rows
#   ./run_all.sh --label-space pseudohypoxic2     run the whole grid on a coarser target
#   ./run_all.sh --optional                       also run encoder/finetune axes
#   ./run_all.sh --amg                            also run the region-free AMG families
#   ./run_all.sh --amg-ablation                   ... and the AMG ablation extension
#   ./run_all.sh --fetch-weights                  pre-download backbones (needs internet)
#   ./run_all.sh --n-parallel 4                   cap concurrent workers
#   ./run_all.sh --from 5                         resume from a given stage
#
# GPU count is discovered at runtime. With N visible GPUs the dispatcher runs N
# fold-level workers, one pinned per device; with none it falls back to a single
# CPU/MPS worker. Completed folds are skipped, so re-running resumes.
#
# Stages
#   1 doctor          environment and data check
#   2 inventory       build the manifest (source of truth for all counts)
#   3 locations       region labels (radiologist CSV, or provisional heuristic)
#   4 folds           build and audit the CV splits
#   5 gate            location-only + region-prior controls  <-- READ THIS FIRST
#   6 grid            baselines, ablations, routed variants, MIL, Tier-3 rows
#     grid            region-free AMG seed/lambda families      (--amg only)
#   7 clinical        clinical-only and image+clinical fusion  (--clinical-csv only)
#   8 optional        encoder-comparison and fine-tuning axes  (--optional only)
#   9 analyse         routing gap, region estimator, expert utilisation
#  10 ensemble        post-hoc probability ensembles (no GPU)
#  11 selective       risk-coverage / deferral analysis (no GPU)
#  12 stats           paired corrected-resampled tests, Holm-corrected
#  13 tables          CSV + LaTeX
#  14 figures         confusion matrices, ROC, routing plots
#
# Stage numbers changed when stages 7, 10 and 11 were added. If you are using
# --from, check the list above rather than reusing an older number.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
PYTHON="${PYTHON:-python}"
OUTPUT_DIR=""
OUTPUT_DIR_EXPLICIT=0
DEFAULT_OUTPUT_DIR="results"
N_PARALLEL=0
SMOKE=0
RUN_OPTIONAL=0
RUN_AMG=0
RUN_AMG_ABLATION=0
FETCH_WEIGHTS=0
LOCATION_CSV=""
CLINICAL_CSV=""
LABEL_SPACE=""
START_STAGE=1
REFERENCE="routed_soft"
LOG_LEVEL="INFO"
EXTRA_SET=()

MAIN_CONFIGS="configs/experiments/*.yaml"
# 41_encoder_radimagenet is deliberately excluded: RadImageNet weights are not
# bundled (they need a manual Keras->torch conversion), so it would contribute
# only guaranteed failures. Run it explicitly once that checkpoint exists:
#   python -m ppgl.cli train --config configs/optional/41_encoder_radimagenet.yaml
# 60_reproduce_rejected is also excluded: it is a deliberate reproduction of the
# rejected setup, not part of the result grid.
OPTIONAL_CONFIGS="configs/optional/40_encoder_dinov2.yaml \
configs/optional/42_encoder_resnet50.yaml \
configs/optional/50_finetune_lora.yaml \
configs/optional/51_finetune_linear_probe.yaml"

# Clinical rows are excluded from the main glob because they need a chart-review
# sheet that is not in the archive; without it every fold would fail. They run
# only when --clinical-csv is given. See configs/clinical/README.md.
CLINICAL_CONFIGS="configs/clinical/80_clinical_only.yaml \
configs/clinical/81_fusion.yaml \
configs/clinical/82_clinical_net.yaml \
configs/clinical/83_clinical_radiomics.yaml"

# Region-free Attention Mass Guidance families. The proposed method (40_amg) and its
# matched control (41_amg_control) are already in MAIN_CONFIGS; these are the seed
# replicates and the lambda sweep, which live under configs/attention/ because they
# are a sweep rather than result rows.
#
# NOT globbed as configs/attention/*.yaml on purpose. That directory also holds the
# superseded 9*.yaml sweep, whose rows carry an auxiliary region-classification head.
# Anatomical region labels do not exist in this archive, so those rows cannot support
# a reported claim and must not be swept back in by a convenient wildcard. See
# configs/attention/README.md.
# The E* control replicates matter as much as the A* proposed replicates: without them
# the headline comparison is a 4-seed mean against a single run, which is an asymmetry a
# reviewer will challenge. They are separate FILES rather than `--set runtime.seed=...`
# overrides because config_fingerprint() drops the whole runtime section, so a seed
# override leaves the experiment name unchanged, the folds resolve to the already
# complete directory, and every job reports `skipped`.
AMG_CONFIGS="configs/attention/A1_amg_seed_7.yaml \
configs/attention/A2_amg_seed_2024.yaml \
configs/attention/A3_amg_seed_31337.yaml \
configs/attention/E1_amg_control_seed_7.yaml \
configs/attention/E2_amg_control_seed_2024.yaml \
configs/attention/E3_amg_control_seed_31337.yaml \
configs/attention/B1_amg_lambda_0p1.yaml \
configs/attention/B2_amg_lambda_0p3.yaml \
configs/attention/B3_amg_lambda_1p0.yaml \
configs/attention/B4_amg_lambda_10.yaml \
configs/attention/B5_amg_lambda_30.yaml \
configs/attention/B6_amg_lambda_100.yaml \
configs/attention/B7_amg_lambda_300.yaml"

# The ablation extension: four relaxations of the attention constraint plus the
# full-fine-tune cell of the adaptation x supervision 2x2. Separate because it is
# 250 extra folds and the paper survives without it.
AMG_ABLATION_CONFIGS="configs/attention/C1_amg_target_0p7.yaml \
configs/attention/C2_amg_rollout.yaml \
configs/attention/C3_amg_heads_max.yaml \
configs/attention/C4_amg_warmup.yaml \
configs/attention/D1_amg_full_finetune.yaml"

# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
usage() { sed -n '2,39p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)          SMOKE=1; shift ;;
    --optional)       RUN_OPTIONAL=1; shift ;;
    --amg)            RUN_AMG=1; shift ;;
    --amg-ablation)   RUN_AMG=1; RUN_AMG_ABLATION=1; shift ;;
    --fetch-weights)  FETCH_WEIGHTS=1; shift ;;
    --n-parallel)     N_PARALLEL="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; OUTPUT_DIR_EXPLICIT=1; shift 2 ;;
    --location-csv)   LOCATION_CSV="$2"; shift 2 ;;
    --clinical-csv)   CLINICAL_CSV="$2"; shift 2 ;;
    --label-space)    LABEL_SPACE="$2"; shift 2 ;;
    --from)           START_STAGE="$2"; shift 2 ;;
    --reference)      REFERENCE="$2"; shift 2 ;;
    --log-level)      LOG_LEVEL="$2"; shift 2 ;;
    --set)            shift; while [[ $# -gt 0 && "$1" != --* ]]; do EXTRA_SET+=("$1"); shift; done ;;
    -h|--help)        usage ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------------------------- #
# Output directory and label space
#
# A different label space is a different prediction task, and experiment names
# are reused across spaces ("swin", "routed_soft", ...). Writing two spaces into
# one output_dir would therefore have the second run overwrite the first run's
# folds. The stored config fingerprint means the results would be recomputed
# rather than silently misread, but the earlier space's numbers would be gone.
# So a non-default label space gets its own output directory unless the caller
# explicitly chose one.
# --------------------------------------------------------------------------- #
KNOWN_SPACES="cluster4 pseudohypoxic2 cluster3 sdhx_vs_rest"
if [[ -n "${LABEL_SPACE}" ]]; then
  if [[ " ${KNOWN_SPACES} " != *" ${LABEL_SPACE} "* ]]; then
    echo "unknown --label-space '${LABEL_SPACE}'; expected one of: ${KNOWN_SPACES}" >&2
    exit 2
  fi
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ -n "${LABEL_SPACE}" && "${LABEL_SPACE}" != "cluster4" ]]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_${LABEL_SPACE}"
  else
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
  fi
fi

RUN_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${RUN_LOG_DIR}"
RUN_LOG="${RUN_LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"

if [[ -n "${LABEL_SPACE}" && "${LABEL_SPACE}" != "cluster4" ]]; then
  {
    echo
    echo "LABEL SPACE ------------------------------------------------------------"
    echo "Running the whole grid on the '${LABEL_SPACE}' target instead of the"
    echo "4-way cluster. Results go to ${OUTPUT_DIR}/ so the 4-class run is not"
    if [[ "${OUTPUT_DIR_EXPLICIT}" -eq 1 ]]; then
      echo "overwritten (you set --output-dir explicitly; make sure it is not the"
      echo "same directory as a run on a different label space)."
    else
      echo "overwritten. Pass --output-dir to change this."
    fi
    echo "Numbers from different label spaces are NOT comparable to each other:"
    echo "the folds are stratified on the target, so the splits differ and the"
    echo "paired tests will correctly refuse to compare across spaces."
    echo "-----------------------------------------------------------------------"
    echo
  } | tee -a "${RUN_LOG}"
fi

# Common flags passed to every ppgl.cli invocation.
COMMON=(--output-dir "${OUTPUT_DIR}" --log-level "${LOG_LEVEL}")
SET_ARGS=()
if [[ -n "${LOCATION_CSV}" ]]; then
  SET_ARGS+=("data.location_source=csv" "data.location_csv=${LOCATION_CSV}")
fi
if [[ -n "${CLINICAL_CSV}" ]]; then
  SET_ARGS+=("data.clinical_csv=${CLINICAL_CSV}")
fi
if [[ -n "${LABEL_SPACE}" ]]; then
  SET_ARGS+=("data.label_space=${LABEL_SPACE}")
fi
if [[ ${#EXTRA_SET[@]} -gt 0 ]]; then
  SET_ARGS+=("${EXTRA_SET[@]}")
fi
# Argument lists are emitted by functions rather than expanded as arrays.
#
# Under `set -u`, expanding an empty array as "${arr[@]}" is an unbound-variable
# error on bash 3.2 (which is what macOS ships) while bash 4.4+ allows it. Going
# through a function keeps the script working on both, which matters because the
# whole point of --smoke is to be runnable anywhere before committing GPU time.
set_flags() {
  if [[ ${#SET_ARGS[@]} -gt 0 ]]; then printf '%s\n' --set "${SET_ARGS[@]}"; fi
}

smoke_flags() {
  if [[ "${SMOKE}" -eq 1 ]]; then printf '%s\n' --smoke-test; fi
}

run() {
  local stage="$1"; shift
  local title="$1"; shift
  if [[ "${stage}" -lt "${START_STAGE}" ]]; then
    printf '\n=== [%02d] %-28s SKIPPED (--from %s) ===\n' "${stage}" "${title}" "${START_STAGE}" | tee -a "${RUN_LOG}"
    return 0
  fi
  printf '\n=== [%02d] %-28s %s ===\n' "${stage}" "${title}" "$(date '+%H:%M:%S')" | tee -a "${RUN_LOG}"
  local started=${SECONDS}
  "$@" 2>&1 | tee -a "${RUN_LOG}"
  printf -- '--- [%02d] %s finished in %ss ---\n' "${stage}" "${title}" "$((SECONDS - started))" | tee -a "${RUN_LOG}"
}

# --------------------------------------------------------------------------- #
banner() {
  cat <<BANNER | tee -a "${RUN_LOG}"
==============================================================================
 PPGL genetic-cluster classification from CT  --  full pipeline
 output_dir : ${OUTPUT_DIR}
 label space: ${LABEL_SPACE:-cluster4 (4-way, default)}
 reference  : ${REFERENCE}
 smoke      : ${SMOKE}
 optional   : ${RUN_OPTIONAL}
 amg        : ${RUN_AMG} (ablation: ${RUN_AMG_ABLATION})
 locations  : ${LOCATION_CSV:-<provisional heuristic>}
 clinical   : ${CLINICAL_CSV:-<not supplied, clinical rows skipped>}
 log        : ${RUN_LOG}
==============================================================================
BANNER
}
banner

if [[ -z "${LOCATION_CSV}" ]]; then
  cat <<WARN | tee -a "${RUN_LOG}"

WARNING -------------------------------------------------------------------
No --location-csv was given, so anatomical region labels will be estimated
heuristically. Region labels do NOT exist in this archive. Heuristic labels
are adequate for validating the pipeline but MUST NOT be used for any number
reported in the paper: the routed model, the location-only control and the
routing analysis all depend on them.
Produce a review file, have it corrected, then re-run with --location-csv:
  python -m ppgl.cli locations --review-csv ${OUTPUT_DIR}/region_review.csv
---------------------------------------------------------------------------

WARN
fi

# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
run 1 "doctor" \
  "${PYTHON}" -m ppgl.cli doctor --config configs/base.yaml "${COMMON[@]}" $(set_flags)

# Optional: populate the pretrained-weight cache. Needs internet on this host.
# Stage 1 (doctor) reports whether the cache is already complete; if it lists
# MISSING backbones and this host has no egress, fetch on a machine that does and
# copy ~/.cache/huggingface across (or share it via HF_HOME).
if [[ "${FETCH_WEIGHTS}" -eq 1 ]]; then
  run 1 "fetch pretrained weights" \
    "${PYTHON}" -m ppgl.cli fetch-weights --config configs/base.yaml "${COMMON[@]}"
fi

run 2 "inventory" \
  "${PYTHON}" -m ppgl.cli inventory --config configs/base.yaml "${COMMON[@]}" $(set_flags)

if [[ -z "${LOCATION_CSV}" ]]; then
  run 3 "locations (provisional)" \
    "${PYTHON}" -m ppgl.cli locations --config configs/base.yaml "${COMMON[@]}" \
      --review-csv "${OUTPUT_DIR}/region_review.csv" $(set_flags)
else
  run 3 "locations (radiologist)" \
    "${PYTHON}" -m ppgl.cli locations --config configs/base.yaml "${COMMON[@]}" $(set_flags)
fi

run 4 "folds" \
  "${PYTHON}" -m ppgl.cli folds --config configs/base.yaml "${COMMON[@]}" $(smoke_flags) $(set_flags)

# Stage 5 is the decision gate. It costs seconds and can invalidate the paper's
# framing, so it runs before any GPU time is spent.
run 5 "gate: location-only" \
  "${PYTHON}" -m ppgl.cli grid --configs configs/experiments/00_region_majority.yaml \
    configs/experiments/01_location_only.yaml \
    "${COMMON[@]}" --n-parallel 1 $(smoke_flags) $(set_flags)

if [[ "${START_STAGE}" -le 5 ]]; then
  cat <<'GATE' | tee -a "${RUN_LOG}"

--------------------------------------------------------------------------
DECISION GATE. Inspect the location-only result now:
  cat results/runs/location_only/summary.json
If location-only is close to the deep models, the paper's premise does not
hold and the framing must change before more compute is spent. See
README.md, "A note on anatomical region labels". The pipeline continues regardless.
--------------------------------------------------------------------------

GATE
fi

run 6 "grid: baselines + routed" \
  "${PYTHON}" -m ppgl.cli grid --configs ${MAIN_CONFIGS} \
    "${COMMON[@]}" --n-parallel "${N_PARALLEL}" $(smoke_flags) $(set_flags)

# The proposed method and its control ride along in MAIN_CONFIGS above. What is added
# here is the evidence that makes their difference interpretable: seed replicates for
# the run-to-run noise floor, and the lambda sweep for dose-response. Opt-in because
# it is 400 folds and the baseline grid is useful without it.
if [[ "${RUN_AMG}" -eq 1 ]]; then
  run 6 "grid: AMG seeds + lambda" \
    "${PYTHON}" -m ppgl.cli grid --configs ${AMG_CONFIGS} \
      "${COMMON[@]}" --n-parallel "${N_PARALLEL}" $(smoke_flags) $(set_flags)
else
  run 6 "AMG families (no --amg, skipped)" true
fi

if [[ "${RUN_AMG_ABLATION}" -eq 1 ]]; then
  run 6 "grid: AMG ablation extension" \
    "${PYTHON}" -m ppgl.cli grid --configs ${AMG_ABLATION_CONFIGS} \
      "${COMMON[@]}" --n-parallel "${N_PARALLEL}" $(smoke_flags) $(set_flags)
else
  run 6 "AMG ablation (skipped)" true
fi

if [[ -n "${CLINICAL_CSV}" ]]; then
  if [[ ! -f "${CLINICAL_CSV}" ]]; then
    echo "--clinical-csv '${CLINICAL_CSV}' does not exist" >&2
    exit 2
  fi
  run 7 "clinical + fusion rows" \
    "${PYTHON}" -m ppgl.cli grid --configs ${CLINICAL_CONFIGS} \
      "${COMMON[@]}" --n-parallel "${N_PARALLEL}" $(smoke_flags) $(set_flags)
else
  run 7 "clinical (no --clinical-csv, skipped)" true
fi

if [[ "${RUN_OPTIONAL}" -eq 1 ]]; then
  run 8 "optional: encoder + finetune axes" \
    "${PYTHON}" -m ppgl.cli grid --configs ${OPTIONAL_CONFIGS} \
      "${COMMON[@]}" --n-parallel "${N_PARALLEL}" $(smoke_flags) $(set_flags)
else
  run 8 "optional (disabled)" true
fi

ROUTED_EXPERIMENTS="routed_hard,routed_soft,routed_anneal,routed_consistency,routed_shared_trunk"
run 9 "analyse routing" \
  "${PYTHON}" -m ppgl.cli analyse --config configs/base.yaml "${COMMON[@]}" \
    --experiments "${ROUTED_EXPERIMENTS}" $(set_flags)

# --------------------------------------------------------------------------- #
# Stage 10: post-hoc ensembles.
#
# Each ensemble is "every member of a declared family", never a subset chosen
# after looking at test scores - picking the best-performing subset afterwards is
# selection on the test set and would inflate the result. All families are
# reported, including the ones that lose.
#
# A family whose members were not all run (for example the Tier-3 rows under
# --smoke, or a fold that failed) is skipped with a warning rather than
# aborting the pipeline this late.
# --------------------------------------------------------------------------- #
ensemble_family() {
  local name="$1"; shift
  local members="$1"; shift
  if "${PYTHON}" -m ppgl.cli ensemble "${COMMON[@]}" \
       --members "${members}" --name "${name}" $(set_flags) 2>&1 | tee -a "${RUN_LOG}"; then
    return 0
  fi
  printf '    ensemble %s skipped (members incomplete)\n' "${name}" | tee -a "${RUN_LOG}"
  return 0
}

if [[ "${START_STAGE}" -le 10 ]]; then
  printf '\n=== [10] %-28s %s ===\n' "ensembles" "$(date '+%H:%M:%S')" | tee -a "${RUN_LOG}"
  ensemble_family "ens_encoders" "resnet101,vit,swin"
  ensemble_family "ens_routing" "routed_hard,routed_soft,routed_anneal,routed_consistency,routed_shared_trunk"
  ensemble_family "ens_tricks" "aux_mixup,aux_prototype,aux_logit_adjust,aux_tta,aux_all_tricks"
  ensemble_family "ens_deep" "resnet101,vit,vit_cl,swin,ppgl_transformer,single_head_aux"
  if [[ -n "${CLINICAL_CSV}" ]]; then
    ensemble_family "ens_fusion_image" "fusion,swin"
  fi
else
  printf '\n=== [10] %-28s SKIPPED (--from %s) ===\n' "ensembles" "${START_STAGE}" | tee -a "${RUN_LOG}"
fi

run 11 "selective prediction" \
  "${PYTHON}" -m ppgl.cli selective --config configs/base.yaml "${COMMON[@]}" \
    --level patient --score msp $(set_flags)

run 12 "stats" \
  "${PYTHON}" -m ppgl.cli stats --config configs/base.yaml "${COMMON[@]}" \
    --reference "${REFERENCE}" $(set_flags)

run 13 "tables" \
  "${PYTHON}" -m ppgl.cli tables --config configs/base.yaml "${COMMON[@]}" \
    --reference "${REFERENCE}" $(set_flags)

run 14 "figures" \
  "${PYTHON}" -m ppgl.cli figures --config configs/base.yaml "${COMMON[@]}" \
    --reference "${REFERENCE}" $(set_flags)

# --------------------------------------------------------------------------- #
cat <<SUMMARY | tee -a "${RUN_LOG}"

==============================================================================
 DONE in $((SECONDS / 60))m $((SECONDS % 60))s

 tables      ${OUTPUT_DIR}/tables/          (main_results.tex, routing_modes.tex, ...)
 figures     ${OUTPUT_DIR}/figures/
 analysis    ${OUTPUT_DIR}/analysis/        (routing gap, comparisons, Holm-corrected p)
 selective   ${OUTPUT_DIR}/selective/       (risk-coverage curves, deferral summary)
 per-fold    ${OUTPUT_DIR}/runs/<exp>/folds/r<R>f<F>/
 manifest    ${OUTPUT_DIR}/manifest.csv     data_qa.json, fold_summary.csv
 log         ${RUN_LOG}

 Read first:
   ${OUTPUT_DIR}/runs/location_only/summary.json      the decision gate
   ${OUTPUT_DIR}/analysis/routing_${REFERENCE}.json   the routing gap
   ${OUTPUT_DIR}/tables/main_results.csv              the headline table
   ${OUTPUT_DIR}/selective/selective_summary.csv      can the model defer? check
                                                      failure_auroc: 0.5 = no
==============================================================================
SUMMARY
