"""A transcription job that can run on any machine.

``run_job`` is the whole GPU-heavy part of the pipeline: pick the model for the
hardware *where it runs*, transcribe, and label speakers. The local backend
calls it in-process; the cloud backend ships the audio and these options to a
rented GPU and calls the same function there, so results are identical.

Options and results are plain JSON so they can cross a machine boundary, which
is also why options are validated: the receiving side must not trust them.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import diarize, models
from .console import Log, ProgressReporter, clean_text
from .cuda_libs import configure_cuda_environment
from .errors import DiarizeError, InputError, TranscribeError
from .hardware import detect_hardware
from .profiles import GPU_ONLY_COMPUTE_TYPES, AsrProfile, select_asr_profile
from .transcribe import DEFAULT_BEAM_SIZE, Transcriber, TranscriptionResult

Segment = dict[str, Any]

_LANGUAGE = re.compile(r"[a-z]{2,3}")
DEVICES = ("auto", "cuda", "cpu")
COMPUTE_TYPES = ("auto", "float16", "float32", "int8", "int8_float16")
DIARIZE_MODES = ("auto", "on", "off")
MAX_BEAM_SIZE = 20
MAX_SPEAKERS = 64
MAX_TEXT = 2000
DIARIZATION_STATES = ("off", "skipped", "applied")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _malformed(what: str) -> TranscribeError:
    return TranscribeError(f"the job result is malformed ({what})")


def _text(value: Any, name: str, limit: int) -> str:
    """Text from a result that may have come from another machine: one clean, bounded line."""
    if not isinstance(value, str):
        raise _malformed(f"{name} should be text")
    return clean_text(value, limit)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _malformed(f"{name} should be a number")
    return float(value)


def _diarization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("status") not in DIARIZATION_STATES:
        raise _malformed("diarization status")
    cleaned: dict[str, Any] = {"status": value["status"]}
    for key in ("reason", "note"):
        if value.get(key) is not None:
            cleaned[key] = _text(value[key], key, 300)
    if cleaned["status"] == "skipped":
        cleaned.setdefault("reason", "unknown")
    if "speakers" in value:
        if not _is_int(value["speakers"]) or not 0 <= value["speakers"] <= 10_000:
            raise _malformed("speaker count")
        cleaned["speakers"] = value["speakers"]
    return cleaned


@dataclass(frozen=True)
class JobOptions:
    language: str | None = None
    prompt: str | None = None
    beam_size: int = DEFAULT_BEAM_SIZE
    condition_on_previous_text: bool = False
    vad: bool = True
    model: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    diarize: str = "auto"  # auto | on | off
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    offline: bool = False
    no_download: bool = False

    def __post_init__(self) -> None:
        _check(self.language is None or (isinstance(self.language, str) and bool(_LANGUAGE.fullmatch(self.language))),
               f"language must be a code such as 'en' or 'zh', got {self.language!r}")
        _check(self.prompt is None or (isinstance(self.prompt, str) and len(self.prompt) <= MAX_TEXT),
               f"prompt must be text of at most {MAX_TEXT} characters")
        _check(_is_int(self.beam_size) and 1 <= self.beam_size <= MAX_BEAM_SIZE,
               f"beam size must be 1 to {MAX_BEAM_SIZE}")
        _check(self.model is None or (isinstance(self.model, str) and 0 < len(self.model) <= 300),
               "model must be a short name or path")
        _check(self.device in DEVICES, f"device must be one of {DEVICES}")
        _check(self.compute_type in COMPUTE_TYPES, f"compute type must be one of {COMPUTE_TYPES}")
        _check(self.diarize in DIARIZE_MODES, f"diarize must be one of {DIARIZE_MODES}")
        for name in ("condition_on_previous_text", "vad", "offline", "no_download"):
            _check(isinstance(getattr(self, name), bool), f"{name} must be true or false")
        for name in ("num_speakers", "min_speakers", "max_speakers"):
            count = getattr(self, name)
            _check(count is None or (_is_int(count) and 1 <= count <= MAX_SPEAKERS),
                   f"{name} must be 1 to {MAX_SPEAKERS}")
        if self.min_speakers is not None and self.max_speakers is not None:
            _check(self.min_speakers <= self.max_speakers, "min speakers exceeds max speakers")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobOptions:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown job options: {sorted(unknown)}")
        return cls(**data)


@dataclass
class JobResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model: str
    device: str
    compute_type: str
    diarization: dict[str, Any] = field(default_factory=lambda: {"status": "off"})
    backend: str = "local"
    hardware: str = ""  # the GPU (or "CPU") that actually ran the job
    warnings: list[str] = field(default_factory=list)  # raised where the job ran

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobResult:
        required = {"segments", "language", "language_probability", "duration", "model", "device", "compute_type"}
        missing = required - set(data)
        if missing:
            raise TranscribeError(f"the job result is incomplete (missing {sorted(missing)})")
        segments = data["segments"]
        if not isinstance(segments, list) or not all(isinstance(s, dict) for s in segments):
            raise _malformed("segments should be a list of objects")
        warnings = data.get("warnings", [])
        if not isinstance(warnings, list):
            raise _malformed("warnings should be a list")
        return cls(
            segments=segments,
            language=_text(data["language"], "language", 20),
            language_probability=_number(data["language_probability"], "language probability"),
            duration=_number(data["duration"], "duration"),
            model=_text(data["model"], "model", 100),
            device=_text(data["device"], "device", 20),
            compute_type=_text(data["compute_type"], "compute type", 40),
            diarization=_diarization(data.get("diarization", {"status": "off"})),
            backend=_text(data.get("backend", "local"), "backend", 20),
            hardware=_text(data.get("hardware", ""), "hardware", 120),
            warnings=[_text(w, "warning", 300) for w in warnings],
        )


def cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # a missing package or a driver problem both mean "no usable CUDA"
        return 0


def local_gpu_available(options: JobOptions) -> bool:
    """Whether this machine would run the job on a working CUDA GPU.

    A bad option combination raises its error here instead of being mistaken for
    "no GPU", which could otherwise send the audio to the cloud.
    """
    if options.device == "cpu":
        return False
    hw = detect_hardware(models.cache_dir())
    profile = select_asr_profile(hw, model=options.model, device=options.device, compute_type=options.compute_type)
    if profile.device != "cuda":
        return False
    configure_cuda_environment()
    return cuda_device_count() > 0


def choose_profile(options: JobOptions, log: Log) -> tuple[AsrProfile, str]:
    """The ASR profile for the machine this runs on, and a name for its hardware.

    Falls back to the CPU if CUDA turns out to be unusable.
    """
    hw = detect_hardware(models.cache_dir())
    profile = select_asr_profile(hw, model=options.model, device=options.device, compute_type=options.compute_type)
    if profile.device == "cuda":
        failures = configure_cuda_environment()
        if cuda_device_count() == 0:
            reason = failures[0] if failures else "CTranslate2 reports no CUDA device"
            if options.device == "cuda":
                raise TranscribeError(f"--device cuda was requested but CUDA is unusable: {reason}")
            log.warn(f"an NVIDIA GPU was found but CUDA is unusable ({reason}); using the CPU. "
                     "Run `transcribe-media --check` for fixes.")
            profile = select_asr_profile(dataclasses.replace(hw, gpus=()), model=options.model,
                                         device="cpu")  # the precision of a GPU request cannot follow
    if profile.device == "cpu" and profile.compute_type in GPU_ONLY_COMPUTE_TYPES:
        raise InputError(f"--compute-type {profile.compute_type} needs a GPU; on the CPU use int8 or float32")
    gpu = hw.best_gpu
    where = gpu.name if profile.device == "cuda" and gpu else "CPU"
    log.info(f"Using {profile.model} ({profile.compute_type}) on {where}")
    for note in profile.notes:
        log.warn(note)
    return profile, where


def load_transcriber(options: JobOptions, profile: AsrProfile) -> Transcriber:
    model_dir = models.resolve_whisper_model(
        profile.model, offline=options.offline, allow_download=not options.no_download
    )
    return Transcriber(model_dir, profile)


def _transcribe(wav: Path, options: JobOptions, profile: AsrProfile, log: Log) -> TranscriptionResult:
    transcriber = load_transcriber(options, profile)
    try:
        return transcriber.transcribe(
            str(wav), language=options.language, prompt=options.prompt, beam_size=options.beam_size,
            condition_on_previous_text=options.condition_on_previous_text, vad_filter=options.vad,
            progress=ProgressReporter(log),
        )
    finally:
        transcriber.close()  # frees the GPU before diarization needs it


def require_diarization(options: JobOptions) -> diarize.DiarizationSetup | None:
    """With ``--diarize on``, show that labelling can run before the slow transcription starts.

    Otherwise a missing token would only be noticed after hours of work, which would then be thrown away.
    """
    if options.diarize != "on":
        return None
    try:
        setup = diarize.prepare_diarization(offline=options.offline, allow_download=not options.no_download)
    except DiarizeError:
        raise
    except Exception as exc:
        raise DiarizeError(f"{type(exc).__name__}: {clean_text(exc, 300)}") from exc
    if not setup.available:
        raise DiarizeError(clean_text(setup.skip_reason or "unavailable", 300))
    return setup


def apply_diarization(
    options: JobOptions, wav: Path, segments: list[Segment], profile: AsrProfile, log: Log,
    setup: diarize.DiarizationSetup | None = None,
) -> tuple[list[Segment], dict[str, Any]]:
    """Label speakers when possible.

    Any failure at all skips the labels (the transcript is already finished and
    must not be lost), unless the user insisted with diarize == "on". ``setup`` is the result of
    :func:`require_diarization` when that already ran.
    """
    if options.diarize == "off":
        return segments, {"status": "off"}
    if not segments:
        return segments, {"status": "skipped", "reason": "no speech was transcribed, so there is nobody to label"}

    def skipped(reason: str) -> tuple[list[Segment], dict[str, Any]]:
        reason = clean_text(reason, 300)  # often another library's error text
        if options.diarize == "on":
            raise DiarizeError(reason)
        log.warn(f"speaker labels skipped: {reason}")
        return segments, {"status": "skipped", "reason": reason}

    try:
        if setup is None:
            setup = diarize.prepare_diarization(offline=options.offline, allow_download=not options.no_download)
        if not setup.available:
            return skipped(setup.skip_reason or "unavailable")
        log.info("Identifying speakers ...")
        turns, note = diarize.run_worker(
            str(wav), setup.pipeline_dir, device=profile.device, device_index=profile.device_index,
            num_speakers=options.num_speakers, min_speakers=options.min_speakers,
            max_speakers=options.max_speakers,
        )
        labelled = diarize.merge_turns(segments, turns)
    except DiarizeError as exc:
        return skipped(str(exc))
    except Exception as exc:  # network, parsing, anything: never lose the finished transcript
        return skipped(f"{type(exc).__name__}: {clean_text(exc)}")
    speakers = len({s["speaker"] for s in labelled if s["speaker"]})
    if not speakers:  # the worker ran but found nobody (music, silence): do not claim labels were applied
        reason = "no speakers were detected"
        log.warn(f"speaker labels skipped: {reason}")
        return segments, {"status": "skipped", "reason": reason}
    note = clean_text(note, 300)
    if note:
        log.warn(note)
    return labelled, {"status": "applied", "speakers": speakers, "note": note}


def run_job(wav: Path, options: JobOptions, log: Log, *, backend: str = "local") -> JobResult:
    """Transcribe ``wav`` (16 kHz mono) and label speakers, on this machine."""
    setup = require_diarization(options)
    profile, hardware = choose_profile(options, log)
    try:
        result = _transcribe(wav, options, profile, log)
    except TranscribeError as exc:
        if profile.device != "cuda" or options.device != "auto":
            raise
        # CUDA libraries load lazily, so a broken setup only shows up here. The CPU gives
        # the same accuracy, so retry rather than fail.
        log.warn(f"the GPU run failed ({clean_text(exc, 300)}); retrying on the CPU")
        profile, hardware = choose_profile(dataclasses.replace(options, device="cpu", compute_type="auto"), log)
        result = _transcribe(wav, options, profile, log)
    segments, diarization = apply_diarization(options, wav, result.segments, profile, log, setup=setup)
    return JobResult(
        segments=segments, language=result.language, language_probability=result.language_probability,
        duration=result.duration, model=result.model, device=result.device,
        compute_type=result.compute_type, diarization=diarization, backend=backend, hardware=hardware,
    )
