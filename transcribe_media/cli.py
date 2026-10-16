"""Command-line entry point: argument parsing and pipeline orchestration."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit, urlunsplit

from . import __version__, audio, captions, formats, job, models, preflight, quality, remote, sources
from .backends import BACKENDS, DEFAULT_MIN_CLOUD_MINUTES, Backend, choose_backend
from .backends.local import LocalBackend
from .backends.modal_backend import DEFAULT_MAX_MINUTES, ModalBackend, cloud_problem, validate_settings
from .cloud import settings as cloud_settings
from .console import Console, clean_text, clock
from .environment import detect_environment, harden_executable_search
from .errors import (
    BadRecordingError,
    InputError,
    OverwriteRefusedError,
    TranscribeError,
    TranscribeMediaError,
    WriteError,
)
from .hardware import detect_hardware
from .job import JobOptions, JobResult
from .profiles import select_asr_profile
from .transcribe import DEFAULT_BEAM_SIZE

DEFAULT_FORMATS = ("txt", "json")
BACKEND_ENV = "TRANSCRIBE_MEDIA_BACKEND"
DEFAULT_MAX_HOURS = 10.0
MAX_CLOUD_MINUTES = cloud_settings.MAX_JOB_SECONDS // 60
# Decoded audio shorter than this share of what the container declares means the file is cut off.
TRUNCATION_RATIO = 0.95


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

def _bounded_int(name: str, low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a whole number, got {text!r}") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{name} must be between {low} and {high}, got {value}")
        return value

    return parse


def _language_code(text: str) -> str:
    code = text.strip().lower()
    if not re.fullmatch(r"[a-z]{2,3}", code):
        raise argparse.ArgumentTypeError(f"{text!r} is not a language code such as 'en', 'de' or 'zh'")
    return code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe-media",
        description="Accuracy-first transcription of audio, video files and URLs.",
    )
    p.add_argument("input", nargs="?", help="path to a media file, or an http(s) URL (e.g. YouTube)")
    p.add_argument("-o", "--output", help="output file, base name or folder (end with / to make one); "
                                          "'-' prints the first format to stdout")
    p.add_argument("--format", help=f"comma-separated: {', '.join(formats.FORMATS)} (default: txt,json)")
    p.add_argument("--timestamps", action="store_true", help="prefix txt lines with [mm:ss]")
    p.add_argument("--language", type=_language_code,
                   help="force the spoken language (ISO code, e.g. en, de, zh); default: detect")
    p.add_argument("--model", help="large-v3, medium, small or the path of a local model folder "
                                   "(default: chosen by hardware)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--compute-type", default="auto",
                   choices=("auto", "float16", "float32", "int8", "int8_float16"))
    p.add_argument("--beam-size", type=_bounded_int("--beam-size", 1, job.MAX_BEAM_SIZE), default=DEFAULT_BEAM_SIZE)
    p.add_argument("--prompt", help="vocabulary to bias spelling in every window: names, jargon, product terms")
    p.add_argument("--condition-on-previous", action="store_true",
                   help="feed each window's text into the next (can cause repetition loops)")
    p.add_argument("--no-vad", action="store_true", help="transcribe everything instead of skipping non-speech")
    p.add_argument("--normalize", action="store_true", help="loudness-normalise quiet recordings")
    p.add_argument("--diarize", default="auto", choices=("auto", "on", "off"),
                   help="speaker labels: auto = if possible, on = required, off (default: auto)")
    count = _bounded_int("speaker counts", 1, job.MAX_SPEAKERS)
    p.add_argument("--num-speakers", type=count)
    p.add_argument("--min-speakers", type=count)
    p.add_argument("--max-speakers", type=count)

    cloud = p.add_argument_group(
        "where to run",
        "By default everything stays on this machine. The cloud options UPLOAD the audio to a GPU provider.")
    cloud.add_argument("--backend", metavar="|".join(BACKENDS),
                       help="local (default), modal = always use a rented cloud GPU, auto = a local GPU "
                            f"if there is one, else a cloud GPU for long audio (default can come from ${BACKEND_ENV})")
    cloud.add_argument("--cloud-gpu", metavar="GPU",
                       help=f"cloud GPU type, one of {', '.join(cloud_settings.ALLOWED_GPUS)} "
                            f"(default: first available of {', '.join(cloud_settings.DEFAULT_GPUS)})")
    cloud.add_argument("--cloud-max-minutes", type=_bounded_int("--cloud-max-minutes", 1, MAX_CLOUD_MINUTES),
                       default=DEFAULT_MAX_MINUTES,
                       help="cancel the cloud job after this long, which caps the cost (default: %(default)s)")
    cloud.add_argument("--cloud-hf-secret", metavar="NAME",
                       help="name of a Modal Secret you created that holds HF_TOKEN, for speaker labels in the cloud")
    cloud.add_argument("--min-cloud-minutes", type=float, default=DEFAULT_MIN_CLOUD_MINUTES,
                       help="with --backend auto, shorter audio stays local (default: %(default)s)")

    p.add_argument("--compare-captions", action="store_true",
                   help="for YouTube: fetch the video's captions and compare them with this transcript")
    p.add_argument("--offline", action="store_true", help="never touch the network; use cached models only")
    p.add_argument("--no-download", action="store_true", help="do not download models")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    p.add_argument("--on-bad", default="warn", choices=("warn", "skip", "fail"),
                   help="when the recording looks empty or garbled (default: warn)")
    p.add_argument("--allow-private-hosts", action="store_true",
                   help="allow URLs that resolve to private or loopback addresses")
    p.add_argument("--max-download-mb", type=_bounded_int("--max-download-mb", 1, 1_000_000), default=2048)
    p.add_argument("--max-duration-hours", type=float, default=DEFAULT_MAX_HOURS,
                   help="refuse longer media; also caps how much audio is decoded (default: %(default)s)")
    p.add_argument("--cookies-from-browser", choices=remote.BROWSERS, metavar="BROWSER",
                   help="use this browser's signed-in session if YouTube refuses anonymous access "
                        "(off by default; cookies are credentials)")
    p.add_argument("--keep-audio", help="also save the decoded 16 kHz audio to this .wav file")
    p.add_argument("--check", action="store_true", help="diagnose the environment and exit")
    p.add_argument("--self-test", action="store_true",
                   help="run a tiny inference on the chosen backend (uploads only a synthetic tone), then exit")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def resolve_backend_setting(args: argparse.Namespace) -> None:
    """Fill in ``args.backend`` from the flag, else the environment, else ``local``."""
    from_env = os.environ.get(BACKEND_ENV)
    args.backend_from_env = args.backend is None and bool(from_env)
    args.backend = args.backend or from_env or "local"


def job_options(args: argparse.Namespace) -> JobOptions:
    try:
        return JobOptions(
            language=args.language, prompt=args.prompt, beam_size=args.beam_size,
            condition_on_previous_text=args.condition_on_previous, vad=not args.no_vad, model=args.model,
            device=args.device, compute_type=args.compute_type, diarize=args.diarize,
            num_speakers=args.num_speakers, min_speakers=args.min_speakers, max_speakers=args.max_speakers,
            offline=args.offline, no_download=args.no_download,
        )
    except ValueError as exc:
        raise InputError(str(exc)) from exc


def validate_backend_args(args: argparse.Namespace) -> None:
    if args.backend not in BACKENDS:
        raise InputError(f"unknown backend {args.backend!r} (choose from {', '.join(BACKENDS)})")
    if args.max_duration_hours <= 0:
        raise InputError("--max-duration-hours must be positive")
    if args.backend == "local":
        return
    validate_settings(args.cloud_gpu, args.cloud_max_minutes)
    if args.offline:
        raise InputError(f"--offline forbids network use, so it cannot be combined with --backend {args.backend}")
    if args.device != "auto":
        raise InputError("--device picks a GPU on this machine; the cloud chooses its own (see --cloud-gpu)")
    if args.model and args.model not in models.WHISPER_MODELS:
        raise InputError("the cloud can only run the built-in models "
                         f"({', '.join(sorted(models.WHISPER_MODELS))}), not a local model folder")
    if args.backend == "modal":
        problem = cloud_problem()  # fail now, before downloading or decoding anything
        if problem:
            raise TranscribeError(f"--backend modal is not available: {problem}")


# --------------------------------------------------------------------------- #
# Output planning and writing
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OutputPlan:
    files: dict[str, Path]
    to_stdout: bool = False
    stdout_format: str = "txt"

    @property
    def base(self) -> Path | None:
        if not self.files:
            return None
        first = next(iter(self.files.values()))
        return first.with_suffix("")


def resolve_formats(arg: str | None, output: str | None) -> list[str]:
    if arg:
        wanted = [f.strip().lower() for f in arg.split(",") if f.strip()]
        unknown = [f for f in wanted if f not in formats.FORMATS]
        if unknown or not wanted:
            raise InputError(f"unknown --format {', '.join(unknown) or arg!r} (choose from {', '.join(formats.FORMATS)})")
        return list(dict.fromkeys(wanted))
    if output and output != "-":
        suffix = Path(output).suffix.lstrip(".").lower()
        if suffix in formats.FORMATS:
            return [suffix]
    return list(DEFAULT_FORMATS)


def check_writable(path: Path) -> None:
    """Fail now, not after an hour of transcription, if ``path`` could not be written.

    Creates the parent folder, and proves the file name is legal for this file
    system (length, characters) by creating and removing a probe of that name.
    """
    probe = path.with_name(path.name + ".probe")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except OSError as exc:
        raise WriteError(f"cannot write {path}: {exc.strerror or exc}") from exc
    try:
        probe.unlink()
    except OSError:
        pass  # the probe proved the name works; a leftover empty file is harmless


def names_a_folder(output: str) -> bool:
    """Whether ``-o`` names a folder (it exists, or ends with a separator) rather than a file or base name."""
    return output.endswith(("/", os.sep)) or Path(output).expanduser().is_dir()


def plan_known_output(output: str | None, wanted: Sequence[str], *, overwrite: bool) -> OutputPlan | None:
    """The output plan when it does not depend on the media's title, else None.

    An explicit file name can be checked (collisions, legal name, writable folder) before the download and
    the decode, instead of after them.
    """
    if output in (None, "-"):
        return None
    try:
        if names_a_folder(output):
            return None
    except OSError as exc:
        raise WriteError(f"cannot use the output location {clean_text(output, 200)}: {exc.strerror or exc}") from exc
    return plan_outputs(output, "transcript", wanted, overwrite=overwrite)


def plan_outputs(output: str | None, stem: str, wanted: Sequence[str], *, overwrite: bool) -> OutputPlan:
    if output == "-":
        return OutputPlan({}, to_stdout=True, stdout_format=wanted[0])
    if output:
        sources.refuse_network_path(output, "the output location")
    try:
        if output is None:
            base = Path.cwd() / f"{stem}.transcript"
        else:
            path = Path(output).expanduser()
            if names_a_folder(output):
                base = path / f"{stem}.transcript"  # a folder; created below if it does not exist yet
            elif path.suffix.lstrip(".").lower() in formats.FORMATS:
                base = path.with_suffix("")
            else:
                base = path
        files = {fmt: Path(f"{base}.{fmt}") for fmt in wanted}
        existing = [str(p) for p in files.values() if p.exists()]
    except OSError as exc:  # for example a name longer than the file system allows
        raise WriteError(f"cannot use the output location {clean_text(output, 200)}: {exc.strerror or exc}") from exc
    if existing and not overwrite:
        raise OverwriteRefusedError(f"output already exists: {', '.join(existing)} (pass --overwrite to replace)")
    for path in files.values():
        check_writable(path)
    return OutputPlan(files)


def plan_audio_copy(target: str | None, *, overwrite: bool) -> Path | None:
    """Validate ``--keep-audio`` up front: a .wav file name that is neither a link nor in the way."""
    if not target:
        return None
    sources.refuse_network_path(target, "the --keep-audio location")
    path = Path(target).expanduser()
    if path.suffix.lower() != ".wav":
        raise InputError("--keep-audio needs a file name ending in .wav")
    try:
        is_link, exists = path.is_symlink(), path.exists()
    except OSError as exc:
        raise WriteError(f"cannot use the --keep-audio location {clean_text(target, 200)}: "
                         f"{exc.strerror or exc}") from exc
    if is_link:
        raise InputError(f"--keep-audio will not write through the link {path}")
    if exists and not overwrite:
        raise OverwriteRefusedError(f"output already exists: {path} (pass --overwrite to replace)")
    check_writable(path)
    return path


def _atomic_write(path: Path, writer: Callable[[Any], None], mode: str, what: str) -> None:
    """Write via an unpredictable temp name in the same folder, then rename into place."""
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name[:40]}.", suffix=".tmp")
        with os.fdopen(fd, mode, **({"encoding": "utf-8", "newline": "\n"} if "b" not in mode else {})) as handle:
            writer(handle)
        os.replace(tmp_name, path)
    except OSError as exc:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass  # cleanup is best effort; report the original failure
        raise WriteError(f"could not write {what} {path}: {exc}") from exc


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 atomically: an interrupted run never leaves a half file."""
    _atomic_write(path, lambda handle: handle.write(text), "w", "")


def copy_audio(src: Path, dest: Path) -> None:
    """Copy the decoded audio next to the outputs, atomically."""
    def copy(out: Any) -> None:
        with open(src, "rb") as source:
            shutil.copyfileobj(source, out)

    _atomic_write(dest, copy, "wb", "the audio")


def public_url(source: sources.MediaSource) -> str | None:
    """The source URL without query strings, fragments or credentials (signed links, tokens)."""
    if source.url is None:
        return None
    info = source.info or {}
    ident = str(info.get("id") or "")
    if str(info.get("extractor_key", "")).lower().startswith("youtube") and re.fullmatch(r"[\w-]{6,20}", ident):
        return f"https://www.youtube.com/watch?v={ident}"
    parts = urlsplit(source.url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #

def select_backend(
    args: argparse.Namespace, options: JobOptions, duration: float | None
) -> tuple[Backend, str | None]:
    """Pick where the job runs, plus a note for the user about the choice."""
    requested = args.backend
    local_gpu = requested != "modal" and job.local_gpu_available(options)
    long_enough = duration is None or duration >= args.min_cloud_minutes * 60
    # Only look at the cloud setup when the answer can matter.
    problem = cloud_problem() if (requested != "local" or (not local_gpu and long_enough)) else None
    choice = choose_backend(
        requested, local_gpu=local_gpu, duration_s=duration, cloud_problem=problem,
        min_cloud_minutes=args.min_cloud_minutes, environment=detect_environment(),
    )
    note = choice.note
    if choice.name == "modal":
        if getattr(args, "backend_from_env", False):
            note = f"{note + ' ' if note else ''}(the cloud backend was selected by ${BACKEND_ENV}={args.backend})"
        return ModalBackend(gpu=args.cloud_gpu, max_minutes=args.cloud_max_minutes,
                            hf_secret=args.cloud_hf_secret, quiet=args.quiet), note
    return LocalBackend(), note


def handle_captions(
    args: argparse.Namespace, source: sources.MediaSource, result: JobResult, plan: OutputPlan, workdir: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Report YouTube captions, and compare them with our transcript if asked."""
    available = captions.find_captions(source.info) if source.info else None
    if not available or not available.any:
        if args.compare_captions:
            what = "this video has no YouTube captions" if source.url else "this input is a local file, not a YouTube video"
            return None, [f"--compare-captions: {what}, so there is nothing to compare with."]
        return None, []
    info = {"manual": list(available.manual), "auto": list(available.auto)}
    if not args.compare_captions:
        return info, [available.notice()]

    pick = available.pick(result.language)
    if pick is None:
        return info, [f"--compare-captions: YouTube has no caption track in the detected language "
                      f"('{result.language}'); available: {available.notice()}"]
    language, is_auto = pick
    kind = "auto-generated captions" if is_auto else "creator-uploaded subtitles"
    try:
        path = remote.download_caption(
            source.url, language, is_auto, workdir, cookies_from_browser=args.cookies_from_browser,
            allow_private_hosts=args.allow_private_hosts)
        cues = captions.load_json3(path)
    except (TranscribeMediaError, ValueError, OSError) as exc:
        return info, [f"--compare-captions: could not fetch the {kind}: {clean_text(exc, 300)}"]
    theirs = captions.cues_text(cues)
    comparison = captions.compare_texts(formats.to_prose(result.segments), theirs)
    info["comparison"] = {"track": language, "kind": "auto" if is_auto else "manual",
                          "similarity": round(comparison.similarity, 4),
                          "our_words": comparison.our_words, "youtube_words": comparison.youtube_words}
    notices = [comparison.summary(f"{kind} ({language})")]
    if plan.base is not None:
        saved = Path(f"{plan.base}.youtube-{'auto' if is_auto else 'manual'}-{language}.txt")
        try:
            if saved.exists() and not args.overwrite:
                notices.append(f"{saved} already exists, so the YouTube caption text was not saved "
                               "(pass --overwrite to replace it)")
            else:
                write_text(saved, theirs)
                notices.append(f"YouTube caption text saved to {saved}")
        except (WriteError, OSError) as exc:
            notices.append(clean_text(exc, 300))
    return info, notices


def build_meta(source: sources.MediaSource, result: JobResult, report: quality.QualityReport,
               caption_info: dict[str, Any] | None, notices: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "tool": f"transcribe-media {__version__}",
        "source": {"kind": "url" if source.url else "file", "title": source.title, "url": public_url(source)},
        "language": result.language,
        "language_probability": round(result.language_probability, 4),
        "duration_seconds": round(result.duration, 2),
        "asr": {"engine": "faster-whisper", "model": result.model, "device": result.device,
                "compute_type": result.compute_type, "hardware": result.hardware, "backend": result.backend},
        "diarization": result.diarization,
        "quality": report.as_dict(),
        "notices": notices,
    }
    if caption_info is not None:
        meta["youtube_captions"] = caption_info
    return meta


def summarize(plan: OutputPlan, result: JobResult, report: quality.QualityReport, notices: list[str]) -> str:
    where = "stdout" if plan.to_stdout else ", ".join(str(p) for p in plan.files.values())
    ran_on = f" on {result.hardware}" if result.hardware else ""
    lines = [
        f"Transcript: {where}",
        f"Language: {result.language} ({result.language_probability:.0%})   Duration: {clock(result.duration)}",
        f"Model: {result.model} ({result.compute_type}, {result.device}){ran_on} [{result.backend}]",
        report.summary().capitalize(),
    ]
    status = result.diarization
    if status["status"] == "applied":
        lines.append(f"Speakers: {status['speakers']}")
    elif status["status"] == "skipped":
        lines.append(f"Speaker labels: skipped - {status['reason']}")
    lines += [f"Note: {n}" for n in notices]
    return "\n".join(lines) + "\n"


def truncation_notice(declared: float | None, decoded: float | None, cap: float) -> str | None:
    """A warning when far less audio came out than the file claims to hold."""
    if not declared or not decoded:
        return None
    expected = min(declared, cap)
    if decoded < expected * TRUNCATION_RATIO:
        return (f"the file declares {clock(declared)} of audio but only {clock(decoded)} could be decoded; "
                "it may be cut off or damaged, so the transcript may be incomplete.")
    return None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def run_transcription(args: argparse.Namespace, console: Console) -> int:
    wanted = resolve_formats(args.format, args.output)
    if args.offline and (sources.is_url(args.input) or args.compare_captions):
        raise InputError("--offline forbids network use, so it cannot be combined with a URL or --compare-captions")
    options = job_options(args)  # every cheap check happens before anything is downloaded or decoded
    audio.require_ffmpeg()
    cap_seconds = args.max_duration_hours * 3600
    early_plan = plan_known_output(args.output, wanted, overwrite=args.overwrite)
    with tempfile.TemporaryDirectory(prefix="transcribe-media-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        audio_copy = plan_audio_copy(args.keep_audio, overwrite=args.overwrite)
        source = sources.resolve_input(
            args.input, workdir=workdir, allow_private_hosts=args.allow_private_hosts,
            max_download_mb=args.max_download_mb, cookies_from_browser=args.cookies_from_browser,
            log=console.info,
        )
        plan = early_plan or plan_outputs(args.output, source.stem, wanted, overwrite=args.overwrite)
        media = audio.ensure_audio_stream(source.path)
        if media.duration and media.duration > cap_seconds:
            raise InputError(f"{clock(media.duration)} of audio exceeds --max-duration-hours="
                             f"{args.max_duration_hours:g}; raise it if this is intended")
        wav = workdir / "audio-16k.wav"
        console.info("Decoding audio ...")
        audio.decode_to_wav(source.path, wav, normalize=args.normalize, max_seconds=cap_seconds)
        decoded = audio.probe_duration(wav)
        if audio_copy:
            copy_audio(wav, audio_copy)

        backend, backend_note = select_backend(args, options, decoded)
        notices: list[str] = [backend_note] if backend_note else []
        cut_off = truncation_notice(media.duration, decoded, cap_seconds)
        if cut_off:
            notices.append(cut_off)
        result = backend.run(wav, options, console)

        report = quality.assess(result.segments, result.duration)
        if report.verdict != quality.VERDICT_OK:
            console.warn(report.summary())
        if report.is_bad and args.on_bad == "fail":
            raise BadRecordingError(report.summary())
        if report.is_bad and args.on_bad == "skip":
            console.warn("not writing output (--on-bad skip)")
            return 0

        language_note = None if args.language else quality.language_note(result.language, result.language_probability)
        if language_note:
            notices.append(language_note)
        caption_info, caption_notices = handle_captions(args, source, result, plan, workdir)
        notices += caption_notices

        meta = build_meta(source, result, report, caption_info, notices)
        meta["timestamps"] = args.timestamps
        if plan.to_stdout:
            sys.stdout.write(formats.render(result.segments, plan.stdout_format, meta=meta, timestamps=args.timestamps))
        else:
            for fmt, path in plan.files.items():
                write_text(path, formats.render(result.segments, fmt, meta=meta, timestamps=args.timestamps))
        stream = sys.stderr if plan.to_stdout else sys.stdout
        print(summarize(plan, result, report, notices), file=stream, end="")
    return 0


def write_test_tone(path: Path, seconds: float = 3.0, rate: int = 16000) -> None:
    frames = b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def run_self_test(args: argparse.Namespace, console: Console) -> int:
    """Push a synthetic tone through the real model on the chosen backend.

    Voice-activity filtering would discard a tone before the decoder ever ran, so it is off and the
    language is fixed: that forces the decoder and beam search, the path that matters on a new GPU.
    """
    audio.require_ffmpeg()
    options = dataclasses.replace(job_options(args), diarize="off", vad=False, language="en")
    backend, _ = select_backend(args, options, None)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="transcribe-media-", ignore_cleanup_errors=True) as tmp:
        tone = Path(tmp) / "tone.wav"
        write_test_tone(tone)
        result = backend.run(tone, options, console)
    print(f"SELF-TEST PASSED: {result.model} ({result.compute_type}) on {result.hardware or result.device} "
          f"via {result.backend}; model load and inference took {time.monotonic() - started:.1f}s.")
    return 0


def run_check(args: argparse.Namespace) -> int:
    hw = detect_hardware(models.cache_dir())
    profile = select_asr_profile(hw, model=args.model, device=args.device, compute_type=args.compute_type)
    checks = preflight.run_checks(
        hw, profile, offline=args.offline, device_count=job.cuda_device_count,
        environment=detect_environment(), cloud_problem=cloud_problem(),
    )
    print(preflight.format_report(checks), end="")
    return 1 if preflight.has_failures(checks) else 0


def run(args: argparse.Namespace) -> int:
    console = Console(args.quiet)
    resolve_backend_setting(args)
    if args.check:
        return run_check(args)
    validate_backend_args(args)
    if args.self_test:
        return run_self_test(args, console)
    if not args.input:
        raise InputError("no input given: pass a media file or URL (see --help)")
    return run_transcription(args, console)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to a legacy codepage
        except (AttributeError, ValueError):
            pass
    harden_executable_search()
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except TranscribeMediaError as exc:
        print(f"error: {clean_text(exc, 1000)}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # a last resort: report cleanly instead of dumping a traceback
        if os.environ.get("TRANSCRIBE_MEDIA_DEBUG"):
            raise
        print(f"error: unexpected {type(exc).__name__}: {clean_text(exc, 500)} "
              "(set TRANSCRIBE_MEDIA_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 1
