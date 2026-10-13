"""Speaker diarization ("who spoke when") with pyannote's community-1 pipeline.

Diarization is optional and never fatal: without pyannote, without a suitable
Hugging Face token (the model is gated), or if the worker fails, it is skipped
and the transcript is written without speaker labels.

The neural part runs in a subprocess (:mod:`transcribe_media.diarize_worker`)
so a native crash cannot lose a long transcription, and GPU memory is fully
released afterwards. This module holds the parent-side logic and the pure
merge of speaker turns onto transcript segments.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import models
from .console import clean_text
from .environment import is_remote
from .errors import DiarizeError, ModelError

Turn = tuple[float, float, str]
Segment = dict[str, Any]

GATED_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
# A word that falls in a gap between turns is attached to a turn this close (seconds).
_MAX_GAP = 1.0


@dataclass(frozen=True)
class DiarizationSetup:
    pipeline_dir: str | None = None
    skip_reason: str | None = None
    will_download: bool = False  # dry run only: everything is in place to fetch the model

    @property
    def available(self) -> bool:
        return self.pipeline_dir is not None


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _hf_token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def _hf_auth_check(repo_id: str, token: str) -> None:
    from huggingface_hub import auth_check

    auth_check(repo_id, token=token)


def _access_problem(token: str, auth_check: Callable[[str, str], None]) -> str | None:
    """Why ``token`` cannot read the gated model, or None if it can."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        auth_check(models.DIARIZATION_MODEL.repo_id, token)
    except GatedRepoError:
        return f"your Hugging Face account has not accepted the model terms; accept them at {GATED_URL}"
    except RepositoryNotFoundError:
        return f"the Hugging Face token is invalid or cannot see the model; check it and the terms at {GATED_URL}"
    except HfHubHTTPError as exc:
        return f"the Hugging Face token was rejected ({clean_text(exc)})"
    except Exception as exc:  # huggingface_hub 2.x raises httpx errors that are not OSErrors
        return f"could not verify access to the diarization model ({type(exc).__name__}: {clean_text(exc)})"
    return None


def prepare_diarization(
    *,
    offline: bool,
    allow_download: bool,
    installed: Callable[[str], bool] = _installed,
    token_getter: Callable[[], str | None] = _hf_token,
    auth_check: Callable[[str, str], None] = _hf_auth_check,
    cached: Callable[..., str | None] = models.cached_path,
    fetch: Callable[..., str] = models.fetch_model,
    dry_run: bool = False,
) -> DiarizationSetup:
    """Decide whether diarization can run; if so, make the model available.

    With ``dry_run`` nothing is downloaded: the result reports whether a
    download would succeed (used by ``--check``).
    """
    spec = models.DIARIZATION_MODEL
    if not installed("pyannote.audio"):
        return DiarizationSetup(skip_reason="pyannote.audio is not installed (run scripts/setup.py --diarization)")
    path = cached(spec)
    if path:  # once downloaded, no token is needed
        return DiarizationSetup(pipeline_dir=path)
    if offline or not allow_download:
        return DiarizationSetup(skip_reason="the diarization model is not cached and downloading is disabled")

    token = token_getter()
    if not token and is_remote():
        return DiarizationSetup(skip_reason=(
            "no Hugging Face token in the cloud environment; create a Modal Secret holding HF_TOKEN and pass "
            f"--cloud-hf-secret NAME after accepting the model terms at {GATED_URL}"))
    if not token:
        return DiarizationSetup(
            skip_reason=(
                "no Hugging Face token found; set HF_TOKEN (or run `hf auth login`) and accept the "
                f"model terms at {GATED_URL}. The model is downloaded once, then needs no token"
            )
        )
    problem = _access_problem(token, auth_check)
    if problem:
        return DiarizationSetup(skip_reason=problem)
    if dry_run:
        return DiarizationSetup(will_download=True)
    try:
        return DiarizationSetup(pipeline_dir=fetch(spec, token=token))
    except ModelError as exc:
        return DiarizationSetup(skip_reason=str(exc))


# --------------------------------------------------------------------------- #
# Subprocess boundary
# --------------------------------------------------------------------------- #

Spawner = Callable[[list[str]], subprocess.CompletedProcess[str]]

# The worker only reads a local model directory, so it gets no credentials and no network:
# just what Python, CUDA and the Hugging Face cache locations need.
_WORKER_ENV_ALLOW = (
    "PATH", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP", "TMPDIR", "SYSTEMROOT", "WINDIR",
    "COMSPEC", "PATHEXT", "LANG", "LC_ALL", "LC_CTYPE", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "XDG_CACHE_HOME",
    "HF_HOME", "HF_HUB_CACHE", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES",
    "VIRTUAL_ENV",
)


def worker_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal environment for the pyannote worker.

    Drops every secret (HF_TOKEN, MODAL_*, API keys) and switches off pyannote's
    usage telemetry, which is on by default and would contact otel.pyannote.ai.
    """
    source = os.environ if base is None else base
    env = {key: value for key, value in source.items() if key in _WORKER_ENV_ALLOW}
    env.update({
        "PYANNOTE_METRICS_ENABLED": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    return env


# Isolated mode (-I) ignores PYTHONPATH, which is how a cloud container exposes this package, so the
# worker is told where the package lives instead: our own location, never the current directory.
PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
WORKER_BOOTSTRAP = (
    "import sys; sys.path.append(sys.argv.pop(1)); "
    "from transcribe_media.diarize_worker import main; sys.exit(main())"
)


def _spawn(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL,
                          env=worker_environment())


def run_worker(
    wav_path: str,
    pipeline_dir: str,
    *,
    device: str,
    device_index: int,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    spawn: Spawner = _spawn,
) -> tuple[list[Turn], str]:
    """Run the pyannote worker; returns (speaker turns, note about the device used)."""
    cmd = [
        # -I: never import code from the current directory or the user's site/PYTHON* settings.
        sys.executable, "-I", "-c", WORKER_BOOTSTRAP, PACKAGE_ROOT,
        "--wav", wav_path, "--pipeline", pipeline_dir,
        "--device", device, "--device-index", str(device_index),
    ]
    for flag, value in (("--num-speakers", num_speakers), ("--min-speakers", min_speakers),
                        ("--max-speakers", max_speakers)):
        if value is not None:
            cmd += [flag, str(value)]
    result = spawn(cmd)
    if result.returncode != 0:
        tail = (result.stderr.strip().splitlines() or ["no error output"])[-1]
        raise DiarizeError(f"the diarization worker failed (exit {result.returncode}): {tail}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        turns = [(float(s), float(e), str(label)) for s, e, label in payload["turns"]]
    except (IndexError, ValueError, KeyError, TypeError) as exc:
        raise DiarizeError(f"the diarization worker returned unreadable output: {exc}") from exc
    return turns, str(payload.get("note", ""))


# --------------------------------------------------------------------------- #
# Merging turns onto the transcript (pure)
# --------------------------------------------------------------------------- #

class _TurnIndex:
    """Fast lookup of the speaker for a time span, for sorted turns."""

    def __init__(self, turns: Sequence[Turn]):
        self.turns = sorted(turns, key=lambda t: (t[0], t[1]))
        self.names = self._friendly_names(self.turns)
        running_end = float("-inf")
        self._ends: list[float] = []
        for _, end, _ in self.turns:  # prefix maximum keeps the array sorted even if turns overlap
            running_end = max(running_end, end)
            self._ends.append(running_end)

    @staticmethod
    def _friendly_names(turns: Sequence[Turn]) -> dict[str, str]:
        names: dict[str, str] = {}
        for _, _, label in turns:
            names.setdefault(label, f"Speaker {len(names) + 1}")
        return names

    def speaker(self, start: float, end: float) -> str | None:
        best_label, best_overlap = None, 0.0
        i = bisect_right(self._ends, start)
        while i < len(self.turns) and self.turns[i][0] < end:
            t_start, t_end, label = self.turns[i]
            overlap = min(end, t_end) - max(start, t_start)
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
            i += 1
        if best_label is None:
            best_label = self._nearest(start, end)
        return self.names[best_label] if best_label else None

    def _nearest(self, start: float, end: float) -> str | None:
        """Closest turn within ``_MAX_GAP`` for a span that overlaps no turn.

        Only the turns on either side of the span are candidates, which is
        exact for the non-overlapping turns the exclusive pipeline produces.
        """
        i = bisect_right(self._ends, start)
        best_label, best_gap = None, _MAX_GAP
        for j in (i - 1, i):
            if 0 <= j < len(self.turns):
                t_start, t_end, label = self.turns[j]
                gap = max(t_start - end, start - t_end, 0.0)
                if gap < best_gap:
                    best_label, best_gap = label, gap
        return best_label


def _split_by_speaker(seg: Segment, index: _TurnIndex) -> list[Segment]:
    """Split one segment wherever the speaker changes between its words."""
    words = seg.get("words")
    if not words:
        return [{**seg, "speaker": index.speaker(seg["start"], seg["end"])}]
    runs: list[tuple[str | None, list[dict[str, Any]]]] = []
    for word in words:
        speaker = index.speaker(word["start"], word["end"])
        if runs and runs[-1][0] == speaker:
            runs[-1][1].append(word)
        else:
            runs.append((speaker, [word]))
    return [
        {
            **seg,
            "start": run[0]["start"],
            "end": run[-1]["end"],
            "text": "".join(w["word"] for w in run).strip(),
            "speaker": speaker,
            "words": run,
        }
        for speaker, run in runs
    ]


def merge_turns(segments: list[Segment], turns: Sequence[Turn]) -> list[Segment]:
    """Label ``segments`` with speakers from ``turns``.

    Segment boundaries are kept; a segment is split only where its words change
    speaker. With no turns the segments are returned unchanged.
    """
    if not turns:
        return segments
    index = _TurnIndex(turns)
    merged: list[Segment] = []
    for seg in segments:
        merged.extend(piece for piece in _split_by_speaker(seg, index) if piece["text"])
    return merged
