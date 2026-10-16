#!/usr/bin/env python3
"""Run transcribe-media from this checkout inside its own environment.

On first use the environment is created for the detected hardware (see
setup.py), and it is repaired if an earlier setup was interrupted or the
checkout moved. Set TRANSCRIBE_MEDIA_NO_AUTOSETUP=1 to disable that and fail
instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup as env_setup  # noqa: E402  (scripts/setup.py, not setuptools)

EXIT_NO_ENVIRONMENT = 15  # distinct from the package's own exit codes (3 is FFmpeg missing)
EXIT_SETUP_FAILED = 16
INTERRUPTED = 130


def run_package(python: Path, args: list[str]) -> int:
    """Run the package under ``python``.

    ``-I`` keeps the current directory and the user's PYTHON* settings out of the import path, so a
    stray file named like a standard module in the working folder cannot run as code.
    """
    command = [str(python), "-I", "-m", "transcribe_media", *args]
    if os.name == "posix":
        os.execv(command[0], command)  # replaces this process, so signals reach the real run directly
    process = subprocess.Popen(command)
    try:
        return process.wait()
    except KeyboardInterrupt:  # Ctrl-C also reaches the child, which cleans up (cancels cloud jobs, ...)
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            return INTERRUPTED


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    env_setup.harden_executable_search()
    venv = env_setup.venv_dir()
    if not env_setup.environment_ready(venv):
        if os.environ.get("TRANSCRIBE_MEDIA_NO_AUTOSETUP"):
            print(f"error: no usable environment at {venv}; run scripts/setup.py first.", file=sys.stderr)
            return EXIT_NO_ENVIRONMENT
        print("Preparing the transcription environment (one time, a few minutes) ...", file=sys.stderr)
        if env_setup.main([]):
            return EXIT_SETUP_FAILED
    return run_package(env_setup.venv_python(venv), args)


if __name__ == "__main__":
    sys.exit(main())
