#!/usr/bin/env python3
"""Create the transcribe-media environment for this machine.

Detects the hardware and installs only what it needs into a private virtual
environment (outside the skill folder, so updating the skill never disturbs it):

* always: faster-whisper, yt-dlp, huggingface-hub
* NVIDIA GPU: CUDA 12.8 cuBLAS/cuDNN wheels (required for RTX 50-series)
* speaker labels: PyTorch from the index matching the hardware, plus pyannote.
  Installed automatically only when a Hugging Face token is present, because
  the diarization model is gated and unusable without one. Force with
  --diarization, or skip with --no-diarization.
* cloud GPU: the Modal client, installed automatically only when Modal
  credentials are present. Force with --cloud, skip with --no-cloud.

Uses ``uv`` when available (it can also fetch a suitable Python), otherwise the
Python running this script (3.10+).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # hardware.py is standard-library only, so usable before installing

from transcribe_media.environment import harden_executable_search  # noqa: E402
from transcribe_media.hardware import detect_nvidia_gpus  # noqa: E402

MARKER = ".transcribe-media-ready"
PYTHON_VERSION = "3.12"
MIN_PYTHON = (3, 10)
TORCH_INDEX_CUDA = "https://download.pytorch.org/whl/cu128"
TORCH_INDEX_CPU = "https://download.pytorch.org/whl/cpu"


def venv_dir() -> Path:
    override = os.environ.get("TRANSCRIBE_MEDIA_VENV")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "transcribe-media" / "venv"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _pyproject_digest() -> str:
    return hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()


def environment_ready(venv: Path) -> bool:
    """Whether a previous setup finished: the marker is written last, only after every install succeeded.

    It also records this checkout's location (the install is editable, so moving the folder breaks the
    environment) and the dependency list, so changes to either trigger a repair on the next run.
    """
    try:
        stamp = json.loads((venv / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (venv_python(venv).exists() and stamp.get("root") == str(ROOT)
            and stamp.get("pyproject") == _pyproject_digest())


def _write_marker(venv: Path, steps: list[list[str]]) -> None:
    stamp = {"root": str(ROOT), "pyproject": _pyproject_digest(), "steps": steps}
    (venv / MARKER).write_text(json.dumps(stamp), encoding="utf-8")


def hf_token_present() -> bool:
    """Whether a Hugging Face token is configured (its value is never read)."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return (home / "token").is_file()


def modal_credentials_present() -> bool:
    """Whether Modal credentials are configured (their values are never read)."""
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return (Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml")).is_file()


def plan_install(*, nvidia: bool, diarization: bool, system: str, cloud: bool = False) -> list[list[str]]:
    """The installer argument lists to run, in order (pure, for testing)."""
    steps: list[list[str]] = []
    if diarization and system != "Darwin":  # PyPI's macOS torch is already right
        index = TORCH_INDEX_CUDA if nvidia else TORCH_INDEX_CPU
        # Torch alone from the PyTorch index; everything else from PyPI, never mixed.
        steps.append(["torch", "torchaudio", "--index-url", index])
    extras = [name for name, wanted in (("gpu", nvidia), ("diarize", diarization), ("cloud", cloud)) if wanted]
    target = f"{ROOT}[{','.join(extras)}]" if extras else str(ROOT)
    steps.append(["-e", target])
    return steps


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True)


def create_venv(venv: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        _run([uv, "venv", "--python", PYTHON_VERSION, str(venv)])
    elif sys.version_info[:2] >= MIN_PYTHON:
        _run([sys.executable, "-m", "venv", str(venv)])
    else:
        raise SystemExit(
            "Python 3.10+ is required. Install uv (https://docs.astral.sh/uv/), which fetches Python "
            "for you, or run this script with a newer Python."
        )


def install(venv: Path, steps: list[list[str]]) -> None:
    python = str(venv_python(venv))
    uv = shutil.which("uv")
    for step in steps:
        _run([uv, "pip", "install", "--python", python, *step] if uv else [python, "-m", "pip", "install", *step])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--diarization", action="store_true", help="install speaker-label support")
    group.add_argument("--no-diarization", action="store_true", help="never install speaker-label support")
    cloud_group = parser.add_mutually_exclusive_group()
    cloud_group.add_argument("--cloud", action="store_true", help="install the cloud GPU client (Modal)")
    cloud_group.add_argument("--no-cloud", action="store_true", help="never install the cloud GPU client")
    parser.add_argument("--cpu", action="store_true", help="ignore any NVIDIA GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without installing")
    args = parser.parse_args(argv)
    harden_executable_search()  # before nvidia-smi and uv are looked up by name

    nvidia = bool(detect_nvidia_gpus()) and not args.cpu
    if args.diarization:
        diarization = True
    elif args.no_diarization:
        diarization = False
    else:
        diarization = hf_token_present()
    cloud = args.cloud or (not args.no_cloud and modal_credentials_present())
    steps = plan_install(nvidia=nvidia, diarization=diarization, system=platform.system(), cloud=cloud)
    venv = venv_dir()
    print(f"Environment: {venv}\nNVIDIA GPU: {'yes' if nvidia else 'no'}   "
          f"Speaker labels: {'yes' if diarization else 'no'}   Cloud GPU client: {'yes' if cloud else 'no'}",
          file=sys.stderr)
    if not diarization and not args.no_diarization:
        print("Speaker labels need a Hugging Face token; add them later with "
              "`python scripts/setup.py --diarization`.", file=sys.stderr)
    if args.dry_run:
        for step in steps:
            print("install", " ".join(step))
        return 0

    (venv / MARKER).unlink(missing_ok=True)  # an interrupted run must never look finished
    try:
        if not venv_python(venv).exists():
            venv.parent.mkdir(parents=True, exist_ok=True)
            create_venv(venv)
        install(venv, steps)
    except subprocess.CalledProcessError as exc:
        print(f"error: the installation failed (exit {exc.returncode}); fix the problem shown above and run "
              "`python scripts/setup.py` again.", file=sys.stderr)
        return 1
    _write_marker(venv, steps)
    print("Setup complete. Verify with: python scripts/transcribe.py --check", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
