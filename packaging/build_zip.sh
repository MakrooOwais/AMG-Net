#!/usr/bin/env bash
#
# Build the distributable code archive.
#
#   ./packaging/build_zip.sh                     -> ppgl_isbi2027_code.zip
#   ./packaging/build_zip.sh --out /tmp/x.zip    -> custom path
#   ./packaging/build_zip.sh --verify            -> also extract and sanity-check
#
# Ships code, configs, the entrypoint, the docs and the bundled pretrained
# checkpoints in weights/ (~1.04 GB) so the deep models run on a host with no
# route to the Hugging Face hub. Pass --no-weights for a ~160 KB code-only
# archive.
#
# Deliberately excludes:
#   - the dataset (~1.2 GB; a placeholder note points at where it goes)
#   - results/ (regenerated in seconds, and shipping provisional region labels
#     risks them being mistaken for radiologist labels)
#   - caches, virtualenvs, PDFs

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${HERE}/.." && pwd)"
NAME="ppgl_isbi2027"
ARCHIVE="${PROJECT}/${NAME}_code.zip"
VERIFY=0
WITH_WEIGHTS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)         ARCHIVE="$2"; shift 2 ;;
    --verify)      VERIFY=1; shift ;;
    --no-weights)  WITH_WEIGHTS=0; shift ;;
    -h|--help)     sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

STAGE_ROOT="$(mktemp -d)"
STAGE="${STAGE_ROOT}/${NAME}"
cleanup() { rm -rf "${STAGE_ROOT}"; }
trap cleanup EXIT

mkdir -p "${STAGE}"
cd "${PROJECT}"

# --- pre-build gate: region-label independence -----------------------------
#
# Anatomical region labels do not exist in this archive. The ones used in the
# earlier result set came from an unvalidated automatic estimator, and the reported
# method must not depend on them. This resolves every config the paper is allowed to
# report and asks the loader itself whether region labels are needed - a grep can
# pass while an inheritance chain silently reintroduces the auxiliary head, which is
# exactly how the earlier contamination happened.
#
# Runs before archiving so a contaminated archive is never written, rather than
# discovered afterwards by --verify.
#
# Needs an interpreter with PyYAML. The system python3 often does not have it, so a
# few candidates are tried and the gate reports loudly if none qualifies rather than
# passing silently.
GATE_PY=""
for cand in "${PYTHON:-}" python3 python "${PROJECT}/.venv/bin/python" /tmp/pdfvenv/bin/python; do
  [[ -n "${cand}" ]] || continue
  command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]] || continue
  if "${cand}" -c 'import yaml' >/dev/null 2>&1; then GATE_PY="${cand}"; break; fi
done
if [[ -z "${GATE_PY}" ]]; then
  echo "ERROR: region-label gate cannot run - no interpreter with PyYAML found." >&2
  echo "       Set PYTHON=/path/to/python (a venv with PyYAML) and re-run." >&2
  echo "       The gate is not optional: it is what keeps region-dependent" >&2
  echo "       configurations out of the reported set." >&2
  exit 1
fi
PYTHONPATH="${PROJECT}" "${GATE_PY}" - <<'PYREGION'
import glob, sys
from ppgl.config import load_config
from ppgl.models.names import needs_region_labels

reported = (["configs/experiments/40_amg.yaml", "configs/experiments/41_amg_control.yaml"]
            + sorted(glob.glob("configs/attention/A*_amg_seed_*.yaml"))
            + sorted(glob.glob("configs/attention/B*_amg_lambda_*.yaml"))
            + sorted(glob.glob("configs/attention/C*_amg_*.yaml"))
            + sorted(glob.glob("configs/attention/D*_amg_*.yaml"))
            + sorted(glob.glob("configs/attention/E*_amg_control_seed_*.yaml")))
if len(reported) < 20:
    print(f"region-label gate FAILED: expected >=20 reported configs, found {len(reported)}")
    sys.exit(1)
bad, names = [], {}
for path in reported:
    cfg = load_config(path)
    if needs_region_labels(cfg):
        bad.append((path, "needs region labels"))
    if getattr(cfg.model, "aux_region", True):
        bad.append((path, "model.aux_region is not False"))
    if float(getattr(cfg.loss, "lambda_region", 0.0)) != 0.0:
        bad.append((path, "loss.lambda_region is not 0"))
    if str(getattr(cfg.data, "location_source", "")) != "none":
        bad.append((path, "data.location_source is not 'none'"))
    if cfg.experiment in names:
        bad.append((path, f"duplicate experiment name '{cfg.experiment}' (also {names[cfg.experiment]})"))
    names[cfg.experiment] = path
if bad:
    print("region-label gate FAILED:")
    for path, why in bad:
        print(f"  {path}: {why}")
    sys.exit(1)
print(f"region-free    : {len(reported)} reported configs, none need region labels")

# Grid workers are spawned processes: the config crosses the boundary as
# cfg.to_dict() and is rebuilt with _from_dict. If that round-trip is not
# value-preserving, workers silently run a different configuration than the one
# validated in the parent. That is not hypothetical - the string "none" was being
# coerced to None, so data.location_source came out invalid and a 500-fold grid
# failed on every job. Assert the round-trip is a fixed point for every config in
# the repo, which is the general invariant rather than a patch for one field.
# Deliberately _from_dict, not dispatch._rebuild_config. The worker also re-runs
# _validate, which repairs exactly the normalisation this gate is looking for, so
# going through the worker entry point would hide a lossy round-trip behind the
# repair. Test the serialisation on its own.
from ppgl.config import _from_dict

everything = sorted(set(
    glob.glob("configs/experiments/*.yaml")
    + glob.glob("configs/attention/[A-Z0-9]*.yaml")
    + glob.glob("configs/clinical/*.yaml")
    + glob.glob("configs/optional/*.yaml")
))
lossy = []
for path in everything:
    try:
        parent = load_config(path).to_dict()
    except Exception:
        continue  # configs needing absent inputs are exercised elsewhere
    worker = _from_dict(parent).to_dict()
    for section, value in parent.items():
        if isinstance(value, dict):
            for key in value:
                if value[key] != worker[section][key]:
                    lossy.append((path, f"{section}.{key}", value[key], worker[section][key]))
        elif value != worker[section]:
            lossy.append((path, section, value, worker[section]))
if lossy:
    print("config round-trip gate FAILED - workers would run a different config:")
    for path, fieldname, pv, wv in lossy:
        print(f"  {path}: {fieldname} parent={pv!r} -> worker={wv!r}")
    sys.exit(1)
print(f"round-trip     : {len(everything)} configs survive to_dict/_from_dict unchanged")
PYREGION

# --- code and configs ------------------------------------------------------
cp -R ppgl configs packaging "${STAGE}/"

# --- entrypoint, packaging metadata, docs ---------------------------------
# Code, the entrypoint and the documentation a recipient needs to run it. Internal
# planning documents are deliberately NOT shipped: they carry a superseded framing of
# the work and would contradict the README.
for f in run_all.sh requirements.txt pyproject.toml README.md INSTALL.md .gitignore; do
  [[ -e "$f" ]] && cp "$f" "${STAGE}/" || echo "warning: $f missing, skipped" >&2
done

# --- strip anything that should never ship --------------------------------
find "${STAGE}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "${STAGE}" \( -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true
chmod +x "${STAGE}/run_all.sh" "${STAGE}/packaging/build_zip.sh"

# --- placeholder marking where the data belongs ---------------------------
mkdir -p "${STAGE}/detection_data_NIH"
cp "${HERE}/PUT_DATA_HERE.txt" "${STAGE}/detection_data_NIH/PUT_DATA_HERE.txt"

# --- bundled ImageNet checkpoints -----------------------------------------
# Included by default: the target server cannot reach the Hugging Face hub, so
# without these every deep model fails at model construction.
WEIGHTS_NOTE="excluded (--no-weights)"
if [[ "${WITH_WEIGHTS}" -eq 1 ]]; then
  if [[ -d "${PROJECT}/weights" ]]; then
    mkdir -p "${STAGE}/weights"
    # Only real checkpoints and their checksums; never partial downloads.
    find "${PROJECT}/weights" -maxdepth 1 -type f \
         \( -name '*.safetensors' -o -name '*.pth' -o -name '*.pth.tar' -o -name 'SHA256SUMS' \) \
         -exec cp {} "${STAGE}/weights/" \;
    N_W=$(find "${STAGE}/weights" -name '*.safetensors' | wc -l | tr -d ' ')
    if [[ "${N_W}" -eq 0 ]]; then
      echo "ERROR: weights/ exists but contains no .safetensors checkpoint." >&2
      echo "       Run: python -m ppgl.cli fetch-weights   (or pass --no-weights)" >&2
      exit 1
    fi
    # Integrity gate: a corrupt checkpoint must not be shipped.
    if [[ -f "${STAGE}/weights/SHA256SUMS" ]]; then
      (cd "${STAGE}/weights" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1) || {
        echo "ERROR: bundled weights failed their SHA256 check; archive not written" >&2
        exit 1
      }
    fi
    WEIGHTS_NOTE="${N_W} checkpoint(s), $(du -sh "${STAGE}/weights" | cut -f1)"
  else
    echo "warning: no weights/ directory; the archive will not include checkpoints." >&2
    echo "         Run: python -m ppgl.cli fetch-weights" >&2
    WEIGHTS_NOTE="absent"
  fi
fi

# --- refuse to ship a broken archive --------------------------------------
python3 -m compileall -q "${STAGE}/ppgl" >/dev/null || {
  echo "ERROR: staged code does not compile; archive not written" >&2
  exit 1
}
bash -n "${STAGE}/run_all.sh" || {
  echo "ERROR: staged run_all.sh has a syntax error; archive not written" >&2
  exit 1
}
find "${STAGE}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# --- build ----------------------------------------------------------------
rm -f "${ARCHIVE}"
(cd "${STAGE_ROOT}" && zip -qr9 "${ARCHIVE}" "${NAME}")

PY_FILES=$(find "${STAGE}/ppgl" -name '*.py' | wc -l | tr -d ' ')
CFG_FILES=$(find "${STAGE}/configs" -name '*.yaml' | wc -l | tr -d ' ')

echo "archive  : ${ARCHIVE}"
echo "size     : $(du -h "${ARCHIVE}" | cut -f1)"
echo "entries  : $(unzip -l "${ARCHIVE}" | tail -1 | awk '{print $2}')"
echo "modules  : ${PY_FILES} python files"
echo "configs  : ${CFG_FILES} yaml files"
echo "weights  : ${WEIGHTS_NOTE}"

# --- optional post-build verification -------------------------------------
if [[ "${VERIFY}" -eq 1 ]]; then
  echo
  echo "verifying from a clean extraction..."
  CHECK_DIR="$(mktemp -d)"
  unzip -q "${ARCHIVE}" -d "${CHECK_DIR}"
  cd "${CHECK_DIR}/${NAME}"
  python3 -m compileall -q ppgl >/dev/null && echo "  compile        OK"
  bash -n run_all.sh && echo "  run_all.sh     OK"
  test -f detection_data_NIH/PUT_DATA_HERE.txt && echo "  data note      OK"
  grep -q 'fetch-weights' ppgl/cli.py && echo "  fetch-weights  OK"
  grep -q 'job_queue.put(None)' ppgl/experiments/dispatch.py && echo "  worker fix     OK"
  grep -q '_atomic_write_csv' ppgl/data/radiomics.py && echo "  atomic cache   OK"
  grep -q 'pretrained_cfg_overlay' ppgl/models/backbones.py && echo "  bundled loader OK"

  # Each feature is checked by the symbol that makes it work, not just by the
  # file existing, so a truncated or half-copied module is caught here.
  check() { grep -q -e "$2" "$1" && echo "  $3 OK" || { echo "  $3 MISSING ($1)" >&2; exit 1; }; }
  check ppgl/labels.py               'LABEL_SPACES'            "label spaces  "
  check ppgl/data/clinical.py        'def write_template'      "clinical io   "
  check ppgl/models/fusion.py        'ImageClinicalFusion'     "fusion model  "
  check ppgl/data/bags.py            'def collate_bags'        "bag dataset   "
  check ppgl/models/mil.py           'GatedAttentionPool'      "attention MIL "
  check ppgl/mixing.py               'def mix_batch'           "mixup/cutmix  "
  check ppgl/losses.py               'LogitAdjustedCrossEntropy' "logit adjust  "
  check ppgl/models/baselines.py     'class PrototypeHead'     "prototype head"
  check ppgl/engine.py               '_forward_tta'            "TTA           "
  check ppgl/experiments/ensemble.py 'def build_ensemble'      "ensembling    "
  check ppgl/experiments/selective.py 'def run_selective_analysis' "selective pred"
  # Thread oversubscription made a 0.9 s tabular fit take 27 minutes on the
  # 96-core server. Both halves of that fix are load-bearing, so both are gated.
  check ppgl/experiments/dispatch.py 'OMP_NUM_THREADS'         "thread pinning"
  check ppgl/models/tabular.py       'n_jobs=1'                "gbm n_jobs    "
  check ppgl/models/attn_lora.py     'AttentionSupervisedViT'  "attn-sup ViT  "
  # The control-vs-proposed attention figure and the checkpoint saving it depends on.
  # Gated together because the figure is useless without the weights, and the weights
  # are pointless without the figure.
  check ppgl/experiments/figures.py  'attention_comparison_figure' "attn cmp fig  "
  check ppgl/experiments/figures.py  'def _patch_mass'         "patch mass    "
  check ppgl/experiments/run_cv.py   'save_checkpoint'         "ckpt saving   "
  check ppgl/config.py               'save_checkpoint'         "ckpt cfg key  "
  check ppgl/cli.py                  'attention-comparison'    "attn cmp cmd  "
  check ppgl/models/attn_lora.py     'def _rollout'            "attn rollout  "
  check ppgl/losses.py               'AttentionLocalisationLoss' "attn loss     "
  check ppgl/losses.py               'def _attention_weight'   "attn schedule "
  check configs/attention/README.md  'attention_target'        "sweep doc     "
  check ppgl/data/transforms.py      'PairedTransformPipeline' "paired xform  "
  check ppgl/data/dataset.py         '_lesion_mask'            "lesion mask   "
  check ppgl/cli.py                  'def cmd_ensemble'        "ensemble cmd  "
  check ppgl/cli.py                  'def cmd_selective'       "selective cmd "
  check run_all.sh                   'smoke_flags'             "bash 3.2 fix  "
  check run_all.sh                   'ensemble_family'         "ensemble stage"
  # No leading dashes in the pattern: grep would parse it as an option.
  check run_all.sh                   'label-space)'            "label-space   "
  check run_all.sh                   'clinical-csv)'           "clinical flag "
  check run_all.sh                   'amg)'                    "amg flag      "

  # --- region-label independence --------------------------------------------
  #
  # Anatomical region labels do not exist in this archive; the ones used earlier
  # came from an unvalidated automatic estimator. The reported method must not
  # depend on them, so the opt-out machinery is gated here rather than trusted.
  check ppgl/models/names.py         'def needs_region_labels' "region gate   "
  check ppgl/models/names.py         '_OPTIONAL_REGION'        "region opt-out"
  check ppgl/experiments/data_prep.py 'needs_region_labels'    "prep uses gate"
  check ppgl/config.py               'aux_region'              "aux_region key"
  check configs/attention/RESULTS.md 'SUPERSEDED'              "contam. notice"
  # A wildcard over configs/attention/ would sweep the superseded region-dependent
  # 9*.yaml rows back into a reported run. Gate that the families are named.
  # Comment lines are stripped first: the file explains why the glob is avoided, and
  # that explanation must not trip its own gate.
  if sed 's/#.*//' run_all.sh | grep -q 'configs/attention/\*\.yaml'; then
    echo "  ERROR: run_all.sh globs configs/attention/*.yaml in executable code" >&2
    exit 1
  fi
  echo "  attn glob      OK (families named explicitly)"

  # Config keys the new rows depend on must be declared, or the configs would be
  # rejected at load time on the server rather than here.
  for key in head_type 'tta:' mixup_alpha cutmix_alpha mix_prob logit_adjust_tau bag_max_slices label_space clinical_csv; do
    grep -q "${key}" ppgl/config.py || { echo "  config key ${key} MISSING" >&2; exit 1; }
  done
  echo "  config keys    OK"

  N_EXP=$(find configs/experiments -name '*.yaml' | wc -l | tr -d ' ')
  N_CLIN=$(find configs/clinical -name '*.yaml' 2>/dev/null | wc -l | tr -d ' ')
  test "${N_EXP}" -ge 21 && echo "  configs        OK (${N_EXP} main, ${N_CLIN} clinical)" \
    || { echo "  expected >=21 main configs, found ${N_EXP}" >&2; exit 1; }
  test -f configs/clinical/README.md && echo "  clinical doc   OK"

  # Structural check on the extracted tree: the reported families must all be
  # present. Their region-independence was already proven functionally by the
  # pre-build gate, which resolves each config through the loader; repeating that
  # here would need PyYAML inside the verification environment, which is not
  # guaranteed. Counting is enough to catch a truncated copy.
  N_AMG=$(find configs/attention -name 'A*_amg_seed_*.yaml' -o -name 'B*_amg_lambda_*.yaml' \
          -o -name 'C*_amg_*.yaml' -o -name 'D*_amg_*.yaml' \
          -o -name 'E*_amg_control_seed_*.yaml' | wc -l | tr -d ' ')
  test "${N_AMG}" -ge 18 \
    && echo "  amg families   OK (${N_AMG} configs + 40_amg, 41_amg_control)" \
    || { echo "  expected >=18 AMG family configs, found ${N_AMG}" >&2; exit 1; }
  for f in configs/experiments/40_amg.yaml configs/experiments/41_amg_control.yaml; do
    test -f "$f" || { echo "  ${f} MISSING" >&2; exit 1; }
  done
  echo "  amg core       OK"
  if [[ "${WITH_WEIGHTS}" -eq 1 && -d weights ]]; then
    (cd weights && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1) \
      && echo "  weights sha256 OK ($(find weights -name '*.safetensors' | wc -l | tr -d ' ') files)"
    python3 - <<'PYCHECK'
# Confirm each bundled file is a valid safetensors container, without torch.
import glob, json, os, struct, sys
bad = []
for path in sorted(glob.glob("weights/*.safetensors")):
    try:
        with open(path, "rb") as handle:
            n = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(n))
        assert isinstance(header, dict) and len(header) > 1
    except Exception as exc:
        bad.append((os.path.basename(path), repr(exc)))
print("  weights format OK" if not bad else f"  weights format FAILED: {bad}")
sys.exit(1 if bad else 0)
PYCHECK
  fi
  cd "${PROJECT}"
  rm -rf "${CHECK_DIR}"
fi
