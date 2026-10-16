"""Model registry and provisioning.

Only allow-listed Hugging Face repositories are ever downloaded, and each is
pinned to a commit SHA so a compromised or force-pushed ``main`` cannot change
what runs. To move to newer weights, update the SHA here and re-run the tests.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import ModelError
from .hardware import free_disk_gb

# Space needed beyond the model itself: HF keeps partial blobs while downloading.
_DISK_HEADROOM = 1.15
_DISK_MARGIN_GB = 0.5

# Mirrors what faster-whisper itself fetches. preprocessor_config.json exists
# only in large-v3, and the vocabulary is .json there but .txt in older repos.
_WHISPER_PATTERNS = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")
_WHISPER_REQUIRED = ("config.json", "model.bin", "tokenizer.json")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    size_gb: float
    patterns: tuple[str, ...]
    required_files: tuple[str, ...]
    required_globs: tuple[str, ...] = ()  # each must match at least one file


def _whisper(name: str, revision: str, size_gb: float, extra_required: tuple[str, ...] = ()) -> ModelSpec:
    return ModelSpec(
        name=name,
        repo_id=f"Systran/faster-whisper-{name}",
        revision=revision,
        size_gb=size_gb,
        patterns=_WHISPER_PATTERNS,
        required_files=(*_WHISPER_REQUIRED, *extra_required),
        required_globs=("vocabulary.*",),
    )


WHISPER_MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        # large-v3 has 128 mel bins; without its preprocessor config faster-whisper silently assumes 80.
        _whisper("large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478", 3.09, ("preprocessor_config.json",)),
        _whisper("medium", "08e178d48790749d25932bbc082711ddcfdfbc4f", 1.53),
        _whisper("small", "536b0662742c02347bc0e980a01041f333bce120", 0.49),
    )
}

# Gated: needs a Hugging Face token whose owner accepted the model terms.
DIARIZATION_MODEL = ModelSpec(
    name="pyannote-community-1",
    repo_id="pyannote/speaker-diarization-community-1",
    revision="3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    size_gb=0.04,
    patterns=("config.yaml", "*.bin", "*.npz"),
    required_files=(
        "config.yaml",
        "embedding/pytorch_model.bin",
        "segmentation/pytorch_model.bin",
        "plda/plda.npz",
        "plda/xvec_transform.npz",
    ),
)

Downloader = Callable[..., str]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _snapshot_download(*, repo_id: str, revision: str, **kwargs) -> str:
    """``revision`` is required, so nothing can be fetched from an unpinned branch."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, revision=revision, **kwargs)


def _is_complete(directory: str, spec: ModelSpec) -> bool:
    root = Path(directory)
    return all((root / name).is_file() for name in spec.required_files) and all(
        any(root.glob(pattern)) for pattern in spec.required_globs
    )


def cached_path(
    spec: ModelSpec, token: str | None = None, downloader: Downloader | None = None
) -> str | None:
    """The snapshot directory if ``spec`` is fully cached, else None."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = (downloader or _snapshot_download)(
            repo_id=spec.repo_id,
            revision=spec.revision,
            allow_patterns=list(spec.patterns),
            local_files_only=True,
            token=token,
        )
    except LocalEntryNotFoundError:
        return None
    # An interrupted download can leave a snapshot folder with files missing.
    return path if _is_complete(path, spec) else None


def fetch_model(
    spec: ModelSpec,
    *,
    offline: bool = False,
    allow_download: bool = True,
    token: str | None = None,
    free_gb: float | None = None,
    log: Callable[[str], None] = _log,
    downloader: Downloader | None = None,
) -> str:
    """Return the local directory for ``spec``, downloading it if permitted."""
    download = downloader or _snapshot_download
    cached = cached_path(spec, token, download)
    if cached:
        return cached

    if offline or not allow_download:
        reason = "--offline" if offline else "--no-download"
        raise ModelError(
            f"model {spec.name!r} is not in the local cache and {reason} forbids downloading it. "
            f"Run once without {reason} to fetch ~{spec.size_gb:.1f} GB from {spec.repo_id}."
        )

    needed = spec.size_gb * _DISK_HEADROOM + _DISK_MARGIN_GB
    free = free_gb if free_gb is not None else _cache_free_gb()
    if free < needed:
        raise ModelError(
            f"not enough disk space for model {spec.name!r}: need about {needed:.1f} GB free "
            f"in the Hugging Face cache, found {free:.1f} GB."
        )

    log(f"Downloading {spec.name} (~{spec.size_gb:.2f} GB) from {spec.repo_id}@{spec.revision[:8]} ...")
    try:
        path = download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            allow_patterns=list(spec.patterns),
            token=token,
        )
    except Exception as exc:  # network, auth and gating errors all surface here
        raise ModelError(f"could not download model {spec.name!r}: {exc}") from exc
    if not _is_complete(path, spec):
        raise ModelError(f"download of model {spec.name!r} finished but files are missing in {path}")
    return path


def cache_dir() -> Path:
    """Where Hugging Face stores models (honours HF_HOME / HF_HUB_CACHE)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def _cache_free_gb() -> float:
    return free_disk_gb(cache_dir())


def _looks_like_path(text: str) -> bool:
    return "/" in text or "\\" in text or text.startswith(("~", "."))


def resolve_whisper_model(name_or_path: str, **kwargs) -> str:
    """Map a model name (or an explicit local directory) to a loadable directory.

    Registry names always win, so a folder named ``large-v3`` in the current
    directory can never replace the pinned download. A local model must be given
    as a path (absolute, ``~``, or containing a separator).
    """
    spec = WHISPER_MODELS.get(name_or_path)
    if spec is not None:
        return fetch_model(spec, **kwargs)
    if _looks_like_path(name_or_path):
        local = Path(name_or_path).expanduser()
        if local.is_dir():
            return str(local)
    known = ", ".join(sorted(WHISPER_MODELS))
    raise ModelError(
        f"unknown model {name_or_path!r}. Choose one of: {known}, or pass the path of an existing "
        "local CTranslate2 Whisper model directory."
    )
