import re
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from transcribe_media import models
from transcribe_media.errors import ModelError
from transcribe_media.models import (
    DIARIZATION_MODEL,
    WHISPER_MODELS,
    fetch_model,
    resolve_whisper_model,
)

SPEC = WHISPER_MODELS["small"]


def populate(directory: Path, spec=SPEC, skip=(), extra=("vocabulary.txt",)) -> str:
    for name in (*spec.required_files, *extra):
        if name in skip:
            continue
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    return str(directory)


class FakeHub:
    """Stands in for snapshot_download; records calls."""

    def __init__(self, tmp_path: Path, cached=False, cached_skip=(), download_error=None, download_skip=()):
        self.tmp_path = tmp_path
        self.cached = cached
        self.cached_skip = cached_skip
        self.download_error = download_error
        self.download_skip = download_skip
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("local_files_only"):
            if not self.cached:
                raise LocalEntryNotFoundError("not cached")
            return populate(self.tmp_path / "cached", skip=self.cached_skip)
        if self.download_error:
            raise self.download_error
        return populate(self.tmp_path / "downloaded", skip=self.download_skip)


def test_registry_is_pinned_and_allow_listed():
    for spec in (*WHISPER_MODELS.values(), DIARIZATION_MODEL):
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.name
        assert spec.repo_id.startswith(("Systran/", "pyannote/")), spec.name


@pytest.mark.parametrize("name, layout", [
    # Regression: the small/medium repos have no preprocessor_config.json and use vocabulary.txt.
    ("small", ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")),
    ("medium", ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")),
    ("large-v3", ("config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json")),
])
def test_real_repo_layouts_count_as_complete(tmp_path, name, layout):
    for file in layout:
        (tmp_path / file).write_bytes(b"x")
    assert models._is_complete(str(tmp_path), WHISPER_MODELS[name])


def test_a_snapshot_without_any_vocabulary_is_incomplete(tmp_path):
    for file in ("config.json", "model.bin", "tokenizer.json"):
        (tmp_path / file).write_bytes(b"x")
    assert not models._is_complete(str(tmp_path), SPEC)


def test_cached_model_is_returned_without_downloading(tmp_path):
    hub = FakeHub(tmp_path, cached=True)
    path = fetch_model(SPEC, downloader=hub, free_gb=0.0)
    assert path.endswith("cached")
    assert [c["local_files_only"] for c in hub.calls] == [True]


def test_download_uses_pinned_revision_and_patterns(tmp_path):
    hub = FakeHub(tmp_path)
    logs = []
    path = fetch_model(SPEC, downloader=hub, free_gb=100, log=logs.append, token="tok")
    assert path.endswith("downloaded")
    call = hub.calls[-1]
    assert call["repo_id"] == SPEC.repo_id and call["revision"] == SPEC.revision
    assert call["token"] == "tok" and "local_files_only" not in call
    assert "vocabulary.*" in call["allow_patterns"]
    assert logs and SPEC.revision[:8] in logs[0]


@pytest.mark.parametrize("flags, word", [({"offline": True}, "--offline"), ({"allow_download": False}, "--no-download")])
def test_refuses_to_download_when_forbidden(tmp_path, flags, word):
    hub = FakeHub(tmp_path)
    with pytest.raises(ModelError, match=word):
        fetch_model(SPEC, downloader=hub, free_gb=100, **flags)
    assert all(c.get("local_files_only") for c in hub.calls)


def test_insufficient_disk_blocks_download(tmp_path):
    hub = FakeHub(tmp_path)
    with pytest.raises(ModelError, match="disk space"):
        fetch_model(WHISPER_MODELS["large-v3"], downloader=hub, free_gb=1.0)
    assert all(c.get("local_files_only") for c in hub.calls)


def test_incomplete_cache_triggers_redownload(tmp_path):
    hub = FakeHub(tmp_path, cached=True, cached_skip=("model.bin",))
    path = fetch_model(SPEC, downloader=hub, free_gb=100, log=lambda m: None)
    assert path.endswith("downloaded")


def test_download_failure_is_wrapped(tmp_path):
    hub = FakeHub(tmp_path, download_error=OSError("connection reset"))
    with pytest.raises(ModelError, match="connection reset"):
        fetch_model(SPEC, downloader=hub, free_gb=100, log=lambda m: None)


def test_download_that_leaves_files_missing_is_an_error(tmp_path):
    hub = FakeHub(tmp_path, download_skip=("tokenizer.json",))
    with pytest.raises(ModelError, match="missing"):
        fetch_model(SPEC, downloader=hub, free_gb=100, log=lambda m: None)


def test_resolve_passes_local_directory_through(tmp_path):
    assert resolve_whisper_model(str(tmp_path)) == str(tmp_path)


def test_resolve_rejects_unknown_names_instead_of_fetching_arbitrary_repos():
    with pytest.raises(ModelError, match="unknown model"):
        resolve_whisper_model("someone-else/evil-whisper")


def test_resolve_known_name_delegates_to_fetch(tmp_path):
    hub = FakeHub(tmp_path, cached=True)
    assert resolve_whisper_model("small", downloader=hub).endswith("cached")


def test_default_free_space_probe_reads_hf_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_cache_free_gb", lambda: 0.1)
    hub = FakeHub(tmp_path)
    with pytest.raises(ModelError, match="disk space"):
        fetch_model(SPEC, downloader=hub)


# --- the pinned large-v3 needs its feature-extractor settings ------------------------------- #

def test_large_v3_is_incomplete_without_preprocessor_config(tmp_path):
    """Without it faster-whisper silently assumes 80 mel bins, which large-v3 (128) garbles."""
    for file in ("config.json", "model.bin", "tokenizer.json", "vocabulary.json"):
        (tmp_path / file).write_bytes(b"x")
    assert not models._is_complete(str(tmp_path), WHISPER_MODELS["large-v3"])
    (tmp_path / "preprocessor_config.json").write_bytes(b"x")
    assert models._is_complete(str(tmp_path), WHISPER_MODELS["large-v3"])


# --- a name never turns into a local folder by accident --------------------------------------- #

def test_a_folder_named_like_a_model_in_the_working_directory_cannot_replace_the_pinned_download(
        tmp_path, monkeypatch):
    (tmp_path / "large-v3").mkdir()  # planted next to the user's files
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(models, "fetch_model", lambda spec, **kwargs: f"pinned:{spec.repo_id}@{spec.revision}")
    resolved = resolve_whisper_model("large-v3")
    assert resolved == f"pinned:Systran/faster-whisper-large-v3@{WHISPER_MODELS['large-v3'].revision}"


def test_a_bare_name_that_is_not_in_the_registry_is_not_looked_up_as_a_folder(tmp_path, monkeypatch):
    (tmp_path / "mymodel").mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ModelError, match="unknown model"):
        resolve_whisper_model("mymodel")


@pytest.mark.parametrize("spelling", ["absolute", "relative", "home"])
def test_a_local_model_must_be_given_as_a_path(tmp_path, monkeypatch, spelling):
    (tmp_path / "mymodel").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    text = {"absolute": str(tmp_path / "mymodel"), "relative": "./mymodel", "home": "~/mymodel"}[spelling]
    assert Path(resolve_whisper_model(text)).resolve() == (tmp_path / "mymodel").resolve()


def test_a_path_that_is_not_a_directory_is_an_unknown_model(tmp_path):
    with pytest.raises(ModelError, match="unknown model"):
        resolve_whisper_model(str(tmp_path / "missing"))


def test_the_real_downloader_refuses_to_run_without_a_pinned_revision():
    with pytest.raises(TypeError, match="revision"):
        models._snapshot_download(repo_id="Systran/faster-whisper-small")
