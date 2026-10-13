"""Recognise where we are running, so notices and defaults fit the place.

Cloud notebooks and agent sandboxes differ from a desktop: the disk is
ephemeral (environments and models are rebuilt each session), there may be no
GPU, and outbound network access can be restricted.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Mapping

# (environment variable, human name, is the filesystem ephemeral?)
_SIGNALS = (
    ("MODAL_TASK_ID", "a Modal container", True),
    ("COLAB_RELEASE_TAG", "Google Colab", True),
    ("COLAB_GPU", "Google Colab", True),
    ("KAGGLE_KERNEL_RUN_TYPE", "a Kaggle notebook", True),
    ("RUNPOD_POD_ID", "a RunPod pod", True),
    ("CODESPACES", "GitHub Codespaces", False),
    ("CLAUDE_CODE_REMOTE", "a Claude Code cloud session", True),
    ("IS_SANDBOX", "a cloud sandbox", True),
)


@dataclass(frozen=True)
class Environment:
    name: str | None = None  # None on an ordinary machine
    ephemeral: bool = False

    @property
    def is_cloud(self) -> bool:
        return self.name is not None


ORDINARY_MACHINE = Environment()


def harden_executable_search(platform: str = sys.platform) -> None:
    """Stop Windows looking for programs in the current folder first.

    Without this, a planted ``ffmpeg.exe``, ``nvidia-smi.exe`` or ``uv.exe`` in the folder the user
    happens to be in (Downloads, say) would run when we start those programs by name.
    """
    if platform == "win32":
        os.environ["NoDefaultCurrentDirectoryInExePath"] = "1"


def detect_environment(env: Mapping[str, str] | None = None) -> Environment:
    env = os.environ if env is None else env
    for variable, name, ephemeral in _SIGNALS:
        if env.get(variable):
            return Environment(name, ephemeral)
    return Environment()


def is_remote() -> bool:
    """True inside the cloud GPU container, where local remedies (flags, tokens) do not apply."""
    return bool(os.environ.get("TRANSCRIBE_MEDIA_REMOTE"))
