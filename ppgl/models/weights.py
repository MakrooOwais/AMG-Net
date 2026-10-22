"""Bundled pretrained weights.

The execution server has no route to the Hugging Face hub, so the ImageNet
checkpoints every deep model needs are shipped inside the project instead of
downloaded on first use. This module owns that: where the files live, how they
are fetched, how they are verified, and how ``create_backbone`` finds them.

Loading goes through timm's own ``file`` override rather than a manual
``load_state_dict``::

    timm.create_model(name, pretrained=True,
                      pretrained_cfg_overlay=dict(file=<local path>))

That path matters. timm resolves ``pretrained_cfg['file']`` as a first-class
weight source and still runs its checkpoint filter, which resamples positional
embeddings when the input resolution differs from the checkpoint's and adapts the
first conv and the classifier. Loading the raw tensors by hand would skip that
and silently break every ViT used at a non-native resolution.

Files are named by their full timm pretrained tag, so provenance is visible on
disk: ``swin_tiny_patch4_window7_224.ms_in1k.safetensors`` is unambiguous in a way
that ``swin.pth`` is not.

Nothing here imports torch, so ``fetch-weights`` and ``doctor`` work on a machine
with no ML stack installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from ppgl.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_WEIGHTS_DIR = "weights"
CHECKSUM_FILENAME = "SHA256SUMS"
_HF_URL = "https://huggingface.co/timm/{tag}/resolve/main/model.safetensors"


class WeightSource:
    """One bundled checkpoint: which timm tag it is and which configs need it."""

    def __init__(self, tag: str, size_mb: float, required: bool, used_by: str) -> None:
        self.tag = tag
        self.size_mb = size_mb
        self.required = required
        self.used_by = used_by

    @property
    def filename(self) -> str:
        return f"{self.tag}.safetensors"

    @property
    def url(self) -> str:
        return _HF_URL.format(tag=self.tag)


# Architecture -> checkpoint. Tags are the defaults timm resolves for these
# architectures; pinning them explicitly means a timm upgrade cannot silently
# change which weights a reported result was produced with.
WEIGHT_SOURCES: Dict[str, WeightSource] = {
    "swin_tiny_patch4_window7_224": WeightSource(
        tag="swin_tiny_patch4_window7_224.ms_in1k",
        size_mb=109.0,
        required=True,
        used_by="swin, single_head_aux, all routed_* variants (the proposed model)",
    ),
    "vit_base_patch16_224": WeightSource(
        tag="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        size_mb=330.2,
        required=True,
        used_by="vit, vit_cl, ppgl_transformer",
    ),
    "resnet101": WeightSource(
        tag="resnet101.a1h_in1k",
        size_mb=170.4,
        required=True,
        used_by="resnet101 baseline",
    ),
    # Needed only by the encoder-comparison axis in configs/optional/, so they are
    # not "required" for the main grid - but they are bundled, so that axis runs
    # offline too.
    "vit_base_patch14_dinov2.lvd142m": WeightSource(
        tag="vit_base_patch14_dinov2.lvd142m",
        size_mb=330.3,
        required=False,
        used_by="encoder_dinov2 (optional axis)",
    ),
    "resnet50": WeightSource(
        tag="resnet50.a1_in1k",
        size_mb=97.7,
        required=False,
        used_by="encoder_resnet50 (optional axis). This is ImageNet, NOT "
        "RadImageNet - encoder_radimagenet needs a separately converted "
        "checkpoint supplied via model.backbone_weights",
    ),
}

REQUIRED_BACKBONES: Tuple[str, ...] = tuple(
    name for name, source in WEIGHT_SOURCES.items() if source.required
)


# --------------------------------------------------------------------------- #
# Locating
# --------------------------------------------------------------------------- #
def resolve_weights_dir(configured: Optional[str] = None) -> str:
    """Where bundled weights live.

    Priority: explicit argument, then ``$PPGL_WEIGHTS_DIR``, then ``weights/``
    relative to the project root. The project root is derived from this file's
    location, so the directory is found regardless of the working directory the
    job was launched from - which matters because the dispatcher spawns workers
    that may not inherit the launch cwd.
    """
    if configured:
        return os.path.abspath(os.path.expanduser(configured))

    from_env = os.environ.get("PPGL_WEIGHTS_DIR")
    if from_env:
        return os.path.abspath(os.path.expanduser(from_env))

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(project_root, DEFAULT_WEIGHTS_DIR)
    if os.path.isdir(candidate):
        return candidate
    return os.path.abspath(DEFAULT_WEIGHTS_DIR)


def local_path(backbone: str, weights_dir: Optional[str] = None) -> Optional[str]:
    """Path to the bundled checkpoint for ``backbone``, or ``None`` if absent.

    Also accepts a full tag as ``backbone`` so a config can name a checkpoint
    directly.
    """
    directory = resolve_weights_dir(weights_dir)

    source = WEIGHT_SOURCES.get(backbone)
    candidates: List[str] = []
    if source is not None:
        candidates.append(source.filename)
    # Allow the architecture name itself, or an already-tagged name.
    candidates.append(f"{backbone}.safetensors")
    if "." in backbone:
        candidates.append(f"{backbone.split('.', 1)[0]}.safetensors")

    for filename in candidates:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path

    # Last resort: any file whose stem starts with the architecture name, which
    # covers a checkpoint bundled under a different tag than the pinned one.
    if os.path.isdir(directory):
        prefix = backbone.split(".", 1)[0]
        for filename in sorted(os.listdir(directory)):
            if filename.startswith(prefix) and filename.endswith(
                (".safetensors", ".pth", ".pth.tar", ".bin")
            ):
                return os.path.join(directory, filename)
    return None


def status(weights_dir: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    """Presence and size of every known checkpoint, for ``doctor``."""
    directory = resolve_weights_dir(weights_dir)
    out: Dict[str, Dict[str, object]] = {}
    for backbone, source in WEIGHT_SOURCES.items():
        path = local_path(backbone, directory)
        out[backbone] = {
            "tag": source.tag,
            "required": source.required,
            "used_by": source.used_by,
            "present": path is not None,
            "path": path,
            "size_mb": round(os.path.getsize(path) / 1048576, 1) if path else None,
            "expected_mb": source.size_mb,
        }
    return out


def missing(required_only: bool = True, weights_dir: Optional[str] = None) -> List[str]:
    directory = resolve_weights_dir(weights_dir)
    return [
        backbone
        for backbone, source in WEIGHT_SOURCES.items()
        if (source.required or not required_only) and local_path(backbone, directory) is None
    ]


# --------------------------------------------------------------------------- #
# Fetching and verifying
# --------------------------------------------------------------------------- #
def download(
    backbones: Optional[Sequence[str]] = None,
    weights_dir: Optional[str] = None,
    include_optional: bool = False,
    force: bool = False,
) -> Dict[str, str]:
    """Download checkpoints over HTTPS. No torch or timm required.

    Returns a backbone -> status map and never raises, so a partial fetch reports
    exactly what is missing.
    """
    import urllib.error
    import urllib.request

    directory = resolve_weights_dir(weights_dir)
    os.makedirs(directory, exist_ok=True)

    names = list(backbones) if backbones else [
        name
        for name, source in WEIGHT_SOURCES.items()
        if source.required or include_optional
    ]

    results: Dict[str, str] = {}
    for backbone in names:
        source = WEIGHT_SOURCES.get(backbone)
        if source is None:
            results[backbone] = "FAILED: unknown backbone, not in WEIGHT_SOURCES"
            continue

        target = os.path.join(directory, source.filename)
        if os.path.isfile(target) and not force:
            results[backbone] = "already present"
            continue

        temp = target + ".part"
        try:
            log.info("downloading %s (%.0f MB) -> %s", source.tag, source.size_mb, target)
            with urllib.request.urlopen(source.url, timeout=120) as response, open(temp, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if not _looks_like_safetensors(temp):
                raise ValueError("downloaded file is not a valid safetensors container")
            os.replace(temp, target)
            results[backbone] = "ok"
        except Exception as exc:  # noqa: BLE001 - report and continue
            results[backbone] = f"FAILED: {type(exc).__name__}: {exc}"
            log.error("could not download %s: %s", source.tag, exc)
        finally:
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except OSError:
                    pass
    return results


def _looks_like_safetensors(path: str) -> bool:
    """Validate the container without loading tensors.

    A safetensors file starts with a little-endian u64 header length followed by
    that many bytes of JSON. Guards against silently bundling an HTML error page
    or a truncated download.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return False
            header_length = struct.unpack("<Q", raw)[0]
            if header_length <= 0 or header_length + 8 > size:
                return False
            header = json.loads(handle.read(header_length))
        return isinstance(header, dict) and len(header) > 0
    except Exception:
        return False


def sha256_of(path: str, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_checksums(weights_dir: Optional[str] = None) -> Optional[str]:
    """Write ``SHA256SUMS`` for every checkpoint present."""
    directory = resolve_weights_dir(weights_dir)
    if not os.path.isdir(directory):
        return None
    lines = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".safetensors", ".pth", ".pth.tar", ".bin")):
            continue
        path = os.path.join(directory, filename)
        lines.append(f"{sha256_of(path)}  {filename}")
    if not lines:
        return None
    target = os.path.join(directory, CHECKSUM_FILENAME)
    with open(target, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    log.info("wrote %d checksums to %s", len(lines), target)
    return target


def verify(weights_dir: Optional[str] = None) -> Dict[str, str]:
    """Check files against ``SHA256SUMS``. Detects a corrupt transfer."""
    directory = resolve_weights_dir(weights_dir)
    checksum_file = os.path.join(directory, CHECKSUM_FILENAME)
    if not os.path.isfile(checksum_file):
        return {"_": "no SHA256SUMS present; nothing to verify"}

    expected: Dict[str, str] = {}
    with open(checksum_file) as handle:
        for line in handle:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                expected[parts[1].strip()] = parts[0].strip()

    out: Dict[str, str] = {}
    for filename, digest in expected.items():
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            out[filename] = "MISSING"
        elif sha256_of(path) == digest:
            out[filename] = "ok"
        else:
            out[filename] = "CHECKSUM MISMATCH"
    return out


__all__ = [
    "CHECKSUM_FILENAME",
    "DEFAULT_WEIGHTS_DIR",
    "REQUIRED_BACKBONES",
    "WEIGHT_SOURCES",
    "WeightSource",
    "download",
    "local_path",
    "missing",
    "resolve_weights_dir",
    "sha256_of",
    "status",
    "verify",
    "write_checksums",
]
