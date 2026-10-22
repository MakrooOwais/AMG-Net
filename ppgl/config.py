"""Configuration objects.

Config is a nested set of dataclasses with YAML loading and dotted-key CLI
overrides, e.g. ``--set optim.lr=3e-5 model.routing=anneal``.

Every run writes its fully-resolved config next to its results so that any
number in the paper can be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import os
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the data lives and how a slice becomes a tensor."""

    # Root containing ``Slices/`` and ``XMLs/`` subdirectories.
    root: str = "detection_data_NIH/original"
    # Manifest produced by ``ppgl.data.inventory``; treated as source of truth.
    manifest: str = "results/manifest.csv"

    # Target vocabulary. See ppgl.labels for the rationale: the 4-way task is
    # information-limited, and a clinically-motivated coarser target is far better
    # powered. One of cluster4 / pseudohypoxic2 / cluster3 / sdhx_vs_rest.
    label_space: str = "cluster4"

    # Optional clinical covariates keyed by patient_uid, for the fusion model and
    # the clinical-only baseline, so that imaging can be credited against covariates.
    clinical_csv: Optional[str] = None

    # Anatomical location labels. These do NOT exist in the archive.
    #   "csv"       - read authoritative radiologist labels from location_csv
    #   "heuristic" - use the provisional automatic estimator (research only)
    #   "none"      - no region labels; routed models cannot be trained
    location_source: str = "heuristic"
    location_csv: Optional[str] = None
    # Written by the heuristic labeller and read back when location_source=csv.
    location_cache: str = "results/location_labels_heuristic.csv"

    image_size: int = 224
    # "full"        - whole 512x512 slice, tumour location implicit
    # "roi"         - tight crop to the union of bounding boxes
    # "roi_context" - bbox union expanded by roi_context_scale (default)
    crop_mode: str = "roi_context"
    roi_context_scale: float = 2.0
    # Minimum crop side in source pixels, guards against 5-pixel lesions.
    min_crop_px: int = 48

    # Augmentation (train split only).
    aug_hflip: bool = True
    aug_vflip: bool = False
    aug_rotate_deg: float = 10.0
    aug_scale: float = 0.1
    aug_translate: float = 0.05
    aug_brightness: float = 0.1
    aug_contrast: float = 0.1
    # Histogram equalisation, as in the original paper. Off by default: the
    # images are already windowed, and equalisation destroys inter-scan
    # intensity comparability.
    hist_equalize: bool = False

    # Multiple-instance learning: cap the training bag at this many slices
    # (random subset each epoch, which also acts as bag-level augmentation).
    # 0 means use every slice. Evaluation always uses all of them.
    bag_max_slices: int = 8

    # Test-time augmentation: number of views averaged at inference.
    #   1 - off (identity only)
    #   2 - identity + horizontal flip
    # Only left/right flips are used. Vertical flips and rotations are not
    # anatomically plausible views of an axial CT slice, so they would test the
    # model on inputs from outside the data distribution.
    tta: int = 1

    num_workers: int = 4
    pin_memory: bool = True
    # Oversample rare clusters in the training loader.
    balanced_sampler: bool = True


@dataclass
class ModelConfig:
    """Which model, which encoder, how much of it to train."""

    # Registry key: see ppgl.models.build_model.
    name: str = "routed"

    # timm backbone name. Encoder-comparison axis:
    #   swin_tiny_patch4_window7_224        (ImageNet, primary)
    #   vit_base_patch14_dinov2.lvd142m     (DINOv2)
    #   resnet50 / resnet101                (ImageNet; RadImageNet via weights)
    backbone: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    # Optional path to a local state_dict, e.g. RadImageNet ResNet50 weights.
    backbone_weights: Optional[str] = None
    # If pretrained weights cannot be downloaded, fail (False) or fall back to
    # random initialisation (True). Random init at n=630 is not publishable; this
    # exists only to exercise the pipeline on an air-gapped machine.
    allow_random_init: bool = False
    # Directory holding the bundled ImageNet checkpoints. None resolves to
    # $PPGL_WEIGHTS_DIR, else <project root>/weights. These are tried before any
    # network access, so the deep models run on a host with no route to the hub.
    weights_dir: Optional[str] = None

    # "llrd"         - full fine-tune with layer-wise LR decay (default)
    # "lora"         - freeze backbone, inject LoRA into attention q,v
    # "linear_probe" - freeze backbone entirely
    finetune: str = "llrd"
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_targets: List[str] = field(default_factory=lambda: ["qkv", "q_proj", "v_proj"])

    hidden_dim: int = 256
    dropout: float = 0.3

    # Classifier head. "mlp" is the head described in the prior work.
    # "prototype" replaces the final linear layer with a scaled cosine
    # classifier against learnable class prototypes, which removes the
    # weight-norm degree of freedom that drives long-tail classifier bias.
    head_type: str = "mlp"

    # Auxiliary region-classification head. Applies to attn_lora and mil.
    #
    # Set False to remove the model's dependence on anatomical region labels
    # entirely - no head is built, no region loss is computed, and the data layer
    # is not asked for region labels at all. That matters because region labels do
    # NOT exist in this archive: with location_source="heuristic" they come from an
    # unvalidated automatic estimator, so any model carrying this head inherits a
    # dependency on provisional labels. A method intended for the paper should set
    # this False together with data.location_source="none".
    aux_region: bool = True

    # ---- attention supervision (attn_lora only) ----------------------------
    # Which attention signal the localisation loss supervises.
    #   "last"    - raw class-token attention in the final block
    #   "rollout" - attention rollout (Abnar & Zuidema, ACL 2020): attention
    #               matrices multiplied across blocks with the residual stream
    #               accounted for. Raw single-layer attention is a poor proxy for
    #               information flow precisely because residual connections carry
    #               information around the attention operation, so rollout is the
    #               better-founded target even though it costs more memory.
    attention_source: str = "last"
    # How the heads are combined before supervision.
    #   "mean" - every head must localise
    #   "max"  - only the best-matching head must, leaving the others free to
    #            attend globally. Heads specialise, and forcing all of them onto
    #            the lesion removes that diversity.
    attention_heads: str = "mean"
    # For attention_source="rollout": rolls out only the last N blocks. 0 = all.
    # Fewer blocks means less memory and a shorter gradient path.
    attention_rollout_layers: int = 0

    # ---- routed model only -------------------------------------------------
    # "hard"        - one-hot ground-truth routing at train time (kept because it
    #                 reproduces a train/inference mismatch worth measuring)
    # "soft"        - router softmax at both train and inference
    # "anneal"      - dense/soft -> sharp over training (EvoMoE-style)
    # "consistency" - soft routing + router entropy/consistency penalty
    routing: str = "soft"
    router_temperature: float = 1.0
    anneal_start_temp: float = 4.0
    anneal_end_temp: float = 0.5
    # Combine experts in logit space (as the prior work did) or in probability
    # space (the more standard mixture-of-experts formulation).
    router_aggregate: str = "logit"
    # Weight on the router-consistency term when routing="consistency".
    consistency_weight: float = 0.1
    # Share expert trunk weights and keep only per-region output layers.
    # Head & neck has 64 slices; independent 200k-param experts overfit.
    share_expert_trunk: bool = False


@dataclass
class OptimConfig:
    """Optimisation. Defaults are the corrected recipe, not the 1e-2 original."""

    optimizer: str = "adamw"
    lr: float = 1e-4
    # Layer-wise LR decay factor applied from the head downwards.
    llrd: float = 0.7
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    epochs: int = 60
    batch_size: int = 16
    grad_accum_steps: int = 1
    grad_clip: float = 1.0

    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 3
    min_lr_factor: float = 0.01

    # Mixup / CutMix. Both off by default. When both alphas are positive one is
    # chosen at random per batch. ``mix_prob`` is the chance that mixing fires at
    # all on a given batch. Mixing is skipped for bag (MIL) inputs, where
    # blending across patients would break the attention semantics.
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mix_prob: float = 0.5

    # Early stopping on the primary validation metric (patient-level macro-F1).
    early_stop_patience: int = 15
    early_stop_min_epochs: int = 20

    amp: bool = True


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    # Class weights inversely proportional to frequency ("inverse") or off.
    class_weighting: str = "inverse"
    label_smoothing: float = 0.0

    # Logit adjustment (Menon et al., ICLR 2021): add tau * log(class prior) to
    # the logits inside the cross-entropy term. tau=1 is balanced softmax; 0 is
    # off. This is an ALTERNATIVE to class_weighting="inverse", not a complement
    # - both correct the same prior. When tau > 0 the cross-entropy term drops
    # its class weights and the trainer logs a warning if weighting was also
    # requested. Tversky keeps its weights either way, since alpha/beta control
    # the recall/precision trade-off rather than the class prior.
    logit_adjust_tau: float = 0.0

    # Multi-class Tversky. alpha penalises false negatives, beta false
    # positives; alpha > beta favours recall on rare clusters.
    tversky_weight: float = 1.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    tversky_smooth: float = 1.0

    # Weight on the auxiliary region-classification loss (lambda in the paper).
    lambda_region: float = 0.1

    # Weight on the attention-localisation term, used by the attn_lora model:
    # -log of the class-token attention mass falling inside the lesion box. 0 is
    # off, which turns that model into a plain LoRA ViT and gives the control row.
    lambda_attention: float = 0.0

    # Required attention mass inside the lesion, in (0, 1]. The loss is zero once
    # this much mass is inside, so the model keeps freedom to attend elsewhere.
    #   1.0 - maximise mass (the original behaviour; drives it to ~98%)
    #   0.7 - "mostly look at the lesion", leaving budget for peritumoral context
    # The annotation is a BOX, roughly 21% of which is background for a round
    # lesion, so demanding all the mass inside is demanding attention on tissue
    # that is not tumour.
    attention_target: float = 1.0

    # Fraction of training over which lambda_attention decays linearly to zero.
    # 0 keeps it constant. Non-zero reproduces the two-stage schedule used by
    # Di Stefano et al. (2026), who supervise attention as a pretraining objective
    # and then fine-tune with cross-entropy alone.
    attention_warmup_frac: float = 0.0

    # Supervised contrastive term. Used only by the vit_cl baseline; the
    # proposed model has no contrastive component.
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07


@dataclass
class CVConfig:
    n_splits: int = 10
    # Repeated CV: single 10-fold at n=287 patients is too noisy for the
    # model comparisons this paper makes.
    n_repeats: int = 5
    seed: int = 1337
    # Fraction of each training split held out for early stopping.
    inner_val_fraction: float = 0.15
    # Abort if any fold is missing a cluster.
    require_all_classes_per_fold: bool = True


@dataclass
class RuntimeConfig:
    """Device and parallelism. Resolved at runtime, not hard-coded."""

    # "auto" | "cuda" | "mps" | "cpu"
    device: str = "auto"
    # Fold-level parallelism: how many worker processes to run concurrently.
    # 0 or "auto" -> one per visible GPU, or 1 on CPU/MPS.
    n_parallel_jobs: int = 0
    # Use DistributedDataParallel *within* a single fold instead of running
    # folds in parallel. Off by default: ~400 short runs parallelise better
    # across folds than within them.
    use_ddp: bool = False
    seed: int = 1337
    deterministic: bool = True
    # Skip a (model, repeat, fold) job if its results file already exists.
    resume: bool = True
    # Persist the best-epoch weights next to the fold's results.
    #
    # Off by default and deliberately so: a ViT-B/16 state_dict is ~340 MB and the
    # full grid is 1,250 folds, which is well over a terabyte for artefacts that no
    # reported number depends on. Switch it on for the handful of folds a qualitative
    # figure needs - attention maps require the trained weights, and nothing else in
    # the pipeline does.
    #
    # ``runtime`` is excluded from config_fingerprint(), so toggling this does NOT
    # invalidate completed folds. That is the desired behaviour (the metrics are
    # unchanged) but it also means enabling it will NOT re-run a fold that is already
    # complete: point such a run at a fresh --output-dir.
    save_checkpoint: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Human-readable experiment name; also the results subdirectory.
    experiment: str = "default"
    output_dir: str = "results"
    # Train on <= this many slices for a fast wiring check.
    smoke_test: bool = False

    # ---------------------------------------------------------------- helpers
    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, "runs", self.experiment)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Loading / overriding
# --------------------------------------------------------------------------- #
_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "optim": OptimConfig,
    "loss": LossConfig,
    "cv": CVConfig,
    "runtime": RuntimeConfig,
}


@functools.lru_cache(maxsize=None)
def _nullable_fields(cls: Any) -> frozenset:
    """Field names on ``cls`` whose declared type admits ``None``.

    This is what separates a null literal from a legitimate value spelled "none".
    ``data.location_source="none"`` is a real member of a three-way enum, whereas
    ``data.location_csv="none"`` means "no file". The strings are identical, so only
    the annotation can tell them apart - and getting it wrong silently replaces an
    enum member with ``None``, which is how a 500-fold grid came to fail with
    ``unknown location_source None``.

    ``from __future__ import annotations`` makes ``field.type`` a string, hence the
    resolve-then-fall-back-to-text approach.
    """
    try:
        hints = typing.get_type_hints(cls)
    except Exception:  # unresolvable forward reference; degrade to textual match
        hints = {f.name: f.type for f in dataclasses.fields(cls)}
    nullable = set()
    for name, hint in hints.items():
        if hint is type(None):
            nullable.add(name)
        elif type(None) in typing.get_args(hint):
            nullable.add(name)
        elif isinstance(hint, str) and ("Optional[" in hint or "None" in hint):
            nullable.add(name)
    return frozenset(nullable)


def _coerce(value: Any, target: Any, allow_none: bool = True) -> Any:
    """Coerce a string (from CLI or a serialised config) to the target's type.

    ``allow_none`` must be False for fields whose declared type does not admit
    ``None``; otherwise the string "none" is destroyed on the way in. Callers derive
    it from the dataclass annotation via :func:`_nullable_fields`. It defaults to
    True only so that direct calls keep their previous behaviour.
    """
    if not isinstance(value, str):
        return value
    if target is None:
        # Unknown target type (Optional[...] default None, or a field the YAML
        # explicitly nulled): best-effort parse.
        lowered = value.lower()
        if allow_none and lowered in ("none", "null", ""):
            return None
        if lowered in ("true", "false"):
            return lowered == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
    if isinstance(target, bool):
        return value.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(target, int) and not isinstance(target, bool):
        return int(float(value))
    if isinstance(target, float):
        return float(value)
    if isinstance(target, list):
        return [v.strip() for v in value.split(",") if v.strip() != ""]
    if allow_none and value.lower() in ("none", "null"):
        return None
    return value


def _assign(owner: Any, name: str, value: Any) -> None:
    """Set ``owner.name`` to ``value``, coerced against the field's declared type."""
    allow_none = name in _nullable_fields(type(owner))
    setattr(owner, name, _coerce(value, getattr(owner, name), allow_none=allow_none))


def _from_dict(payload: Dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in payload.items():
        if key in _SECTIONS:
            if not isinstance(value, dict):
                raise ValueError(f"config section '{key}' must be a mapping")
            section = getattr(cfg, key)
            for sub_key, sub_value in value.items():
                if not hasattr(section, sub_key):
                    raise ValueError(f"unknown config key '{key}.{sub_key}'")
                _assign(section, sub_key, sub_value)
        elif hasattr(cfg, key):
            _assign(cfg, key, value)
        else:
            raise ValueError(f"unknown config key '{key}'")
    return cfg


def load_config(
    path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> Config:
    """Load YAML config (optionally inheriting via ``_base_``) then apply overrides.

    Parameters
    ----------
    path
        YAML file. If ``None``, all defaults are used.
    overrides
        Dotted assignments such as ``["optim.lr=3e-5", "model.routing=anneal"]``.
    """
    payload: Dict[str, Any] = {}
    if path:
        payload = _load_yaml_with_base(path)

    cfg = _from_dict(payload)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override '{override}' must be key=value")
        dotted, raw_value = override.split("=", 1)
        parts = dotted.split(".")
        target: Any = cfg
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config section '{part}' in '{dotted}'")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"unknown config key '{dotted}'")
        _assign(target, leaf, raw_value)

    _validate(cfg)
    return cfg


def _load_yaml_with_base(path: str, _seen: Optional[set] = None) -> Dict[str, Any]:
    _seen = _seen or set()
    real = os.path.realpath(path)
    if real in _seen:
        raise ValueError(f"circular _base_ include at {path}")
    _seen.add(real)

    with open(path) as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload

    base_path = base_ref if os.path.isabs(base_ref) else os.path.join(os.path.dirname(path), base_ref)
    base_payload = _load_yaml_with_base(base_path, _seen)
    return _deep_merge(base_payload, payload)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate(cfg: Config) -> None:
    from ppgl.labels import LABEL_SPACES

    if cfg.data.label_space not in LABEL_SPACES:
        raise ValueError(
            f"data.label_space={cfg.data.label_space!r} invalid; "
            f"available: {sorted(LABEL_SPACES)}"
        )
    # ``location_source: none`` in YAML parses to Python None. Normalise it, for the
    # same reason as class_weighting: the archived config.json must be unambiguous
    # about whether region labels were used, and a typo must not silently disable
    # them.
    if cfg.data.location_source is None:
        cfg.data.location_source = "none"
    if cfg.data.location_source not in ("csv", "heuristic", "none"):
        raise ValueError(f"data.location_source={cfg.data.location_source!r} invalid")
    if cfg.data.crop_mode not in ("full", "roi", "roi_context"):
        raise ValueError(f"data.crop_mode={cfg.data.crop_mode!r} invalid")
    if cfg.model.finetune not in ("llrd", "lora", "linear_probe"):
        raise ValueError(f"model.finetune={cfg.model.finetune!r} invalid")
    if cfg.model.head_type not in ("mlp", "prototype"):
        raise ValueError(f"model.head_type={cfg.model.head_type!r} invalid")
    # ``class_weighting: none`` in YAML parses to Python None. Normalise it to the
    # string so the archived config.json is unambiguous about what was run, and so
    # a typo is caught here rather than silently disabling the weighting.
    if cfg.loss.class_weighting is None:
        cfg.loss.class_weighting = "none"
    if cfg.loss.class_weighting not in ("inverse", "none"):
        raise ValueError(
            f"loss.class_weighting={cfg.loss.class_weighting!r} invalid; "
            "expected 'inverse' or 'none'"
        )
    if int(cfg.data.tta) < 1:
        raise ValueError(f"data.tta={cfg.data.tta} must be >= 1 (1 means no TTA)")
    if cfg.optim.mixup_alpha < 0 or cfg.optim.cutmix_alpha < 0:
        raise ValueError("optim.mixup_alpha and optim.cutmix_alpha must be >= 0")
    if not 0.0 <= float(cfg.optim.mix_prob) <= 1.0:
        raise ValueError(f"optim.mix_prob={cfg.optim.mix_prob} must be in [0, 1]")
    if float(cfg.loss.logit_adjust_tau) < 0:
        raise ValueError("loss.logit_adjust_tau must be >= 0 (0 disables it)")
    if float(cfg.loss.lambda_attention) < 0:
        raise ValueError("loss.lambda_attention must be >= 0 (0 disables it)")
    if not 0.0 < float(cfg.loss.attention_target) <= 1.0:
        raise ValueError(
            f"loss.attention_target={cfg.loss.attention_target} must be in (0, 1]; "
            "it is a required attention-mass fraction, not a weight"
        )
    if not 0.0 <= float(cfg.loss.attention_warmup_frac) <= 1.0:
        raise ValueError("loss.attention_warmup_frac must be in [0, 1]")
    if cfg.model.attention_source not in ("last", "rollout"):
        raise ValueError(f"model.attention_source={cfg.model.attention_source!r} invalid")
    if cfg.model.attention_heads not in ("mean", "max"):
        raise ValueError(f"model.attention_heads={cfg.model.attention_heads!r} invalid")
    if int(cfg.model.attention_rollout_layers) < 0:
        raise ValueError("model.attention_rollout_layers must be >= 0 (0 = all blocks)")
    if float(cfg.loss.lambda_attention) > 0 and cfg.data.crop_mode == "roi":
        raise ValueError(
            "loss.lambda_attention > 0 with data.crop_mode='roi' is vacuous: a tight "
            "crop makes the lesion fill the frame, so 'attend to the lesion' is "
            "trivially satisfied. Use crop_mode='roi_context' (lesion covers ~21% of "
            "the frame) or 'full'."
        )
    if cfg.model.routing not in ("hard", "soft", "anneal", "consistency"):
        raise ValueError(f"model.routing={cfg.model.routing!r} invalid")
    if cfg.model.router_aggregate not in ("logit", "prob"):
        raise ValueError(f"model.router_aggregate={cfg.model.router_aggregate!r} invalid")
    if cfg.optim.lr >= 1e-2:
        raise ValueError(
            f"optim.lr={cfg.optim.lr} is >=1e-2. A fixed 1e-2 on a pretrained Swin "
            "degrades the backbone, and an earlier version of this work did exactly "
            "that. If you intend to reproduce it, remove this guard explicitly."
        )
    if cfg.cv.n_splits < 2:
        raise ValueError("cv.n_splits must be >= 2")
    if cfg.cv.n_repeats < 1:
        raise ValueError("cv.n_repeats must be >= 1")


__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "LossConfig",
    "CVConfig",
    "RuntimeConfig",
    "load_config",
]
