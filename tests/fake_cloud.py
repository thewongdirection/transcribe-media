"""A stand-in for the Modal pieces the cloud backend touches (names verified against modal 1.6)."""

from __future__ import annotations

import contextlib
import json
import time
from types import SimpleNamespace
from typing import Any, Callable


class AuthError(Exception):
    pass


class ModalTimeoutError(Exception):
    pass


class FakeCall:
    def __init__(self, cloud: FakeCloud, job_id: str):
        self.cloud, self.job_id = cloud, job_id
        self.cancelled_with: bool | None = None

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        self.cloud.timeouts.append(timeout)
        return self.cloud.behaviour(self.cloud, self.job_id)

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled_with = terminate_containers
        self.cloud.cancelled.append(terminate_containers)


class FakeFunction:
    def __init__(self, cloud: FakeCloud, options: dict[str, Any] | None = None):
        self.cloud, self.options = cloud, options or {}

    def with_options(self, **options: Any) -> FakeFunction:
        self.cloud.option_calls.append(options)
        return FakeFunction(self.cloud, options)

    def spawn(self, job_id: str) -> FakeCall:
        call = FakeCall(self.cloud, job_id)
        self.cloud.spawned.append((job_id, self.options))
        self.cloud.calls.append(call)
        return call


class FakeVolume:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.mtimes: dict[str, float] = {}

    @contextlib.contextmanager
    def batch_upload(self):
        volume = self

        class Batch:
            def put_file(self, source: Any, remote: str) -> None:
                data = source.read() if hasattr(source, "read") else open(source, "rb").read()
                volume.files[remote] = data

        yield Batch()

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def remove_file(self, path: str, recursive: bool = False) -> None:
        doomed = [p for p in self.files if p == path or (recursive and p.startswith(path + "/"))]
        if not doomed:
            raise FileNotFoundError(f"{path} not found")
        for p in doomed:
            del self.files[p]

    def listdir(self, path: str):
        names = sorted({p[len(path) + 1:].split("/")[0] for p in self.files if p.startswith(path + "/")})
        return [SimpleNamespace(path=f"{path.lstrip('/')}/{n}", mtime=self.mtimes.get(f"{path}/{n}", time.time()))
                for n in names]


def success(cloud: FakeCloud, job_id: str) -> dict[str, Any]:
    cloud.volume.files[f"/jobs/{job_id}/result.json"] = json.dumps(cloud.result).encode()
    return {"ok": True}


class FakeCloud:
    """Looks like ``transcribe_media.cloud.modal_app`` (modal, app, run_remote, volume)."""

    def __init__(self, behaviour: Callable[[FakeCloud, str], dict[str, Any]] = success):
        self.behaviour = behaviour
        self.volume = FakeVolume()
        self.result: dict[str, Any] = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": None}],
            "language": "en", "language_probability": 0.99, "duration": 1.0,
            "model": "large-v3", "device": "cuda", "compute_type": "float16",
            "diarization": {"status": "off"}, "backend": "modal", "hardware": "NVIDIA L4",
            "warnings": ["remote warning"],
        }
        self.spawned: list[Any] = []
        self.calls: list[FakeCall] = []
        self.cancelled: list[bool] = []
        self.timeouts: list[float | None] = []
        self.option_calls: list[dict[str, Any]] = []
        self.secret_requests: list[Any] = []
        self.output_enabled = 0
        cloud = self

        class Secret:
            @staticmethod
            def from_name(name: str, required_keys: list[str] | None = None):
                cloud.secret_requests.append((name, required_keys))
                return ("secret", name)

        @contextlib.contextmanager
        def enable_output():
            cloud.output_enabled += 1
            yield

        self.modal = SimpleNamespace(
            Secret=Secret,
            enable_output=enable_output,
            exception=SimpleNamespace(AuthError=AuthError, TimeoutError=ModalTimeoutError),
        )
        self.app = SimpleNamespace(run=lambda: contextlib.nullcontext())
        self.run_remote = FakeFunction(self)


def fake_modal_module(*, token_id: str | None, token_secret: str | None) -> Any:
    """What ``cloud_problem`` inspects."""
    values = {"token_id": token_id, "token_secret": token_secret}
    return SimpleNamespace(config=SimpleNamespace(config=SimpleNamespace(get=values.get)))


__all__ = ["FakeCloud", "AuthError", "ModalTimeoutError", "fake_modal_module", "success"]
