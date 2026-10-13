"""Tests for the bootstrap scripts (scripts/ is not a package, so load by path)."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def setup_script():
    return load("setup")


def test_plan_for_the_nvidia_box_with_speaker_labels(setup_script):
    steps = setup_script.plan_install(nvidia=True, diarization=True, system="Linux")
    assert steps[0] == ["torch", "torchaudio", "--index-url", setup_script.TORCH_INDEX_CUDA]
    assert steps[1][0] == "-e" and steps[1][1].endswith("[gpu,diarize]")
    assert "cu128" in setup_script.TORCH_INDEX_CUDA  # Blackwell needs CUDA 12.8 builds


def test_torch_comes_from_its_own_index_and_nothing_else_does(setup_script):
    steps = setup_script.plan_install(nvidia=True, diarization=True, system="Windows")
    assert sum("--index-url" in s or "--extra-index-url" in s for s in steps) == 1
    assert "--extra-index-url" not in [a for s in steps for a in s]


def test_cpu_only_without_speaker_labels_installs_just_the_core(setup_script):
    steps = setup_script.plan_install(nvidia=False, diarization=False, system="Linux")
    assert steps == [["-e", str(setup_script.ROOT)]]


def test_cpu_with_speaker_labels_uses_the_cpu_torch_build(setup_script):
    steps = setup_script.plan_install(nvidia=False, diarization=True, system="Windows")
    assert steps[0][-1] == setup_script.TORCH_INDEX_CPU and steps[1][1].endswith("[diarize]")


def test_gpu_without_speaker_labels_skips_torch_entirely(setup_script):
    steps = setup_script.plan_install(nvidia=True, diarization=False, system="Linux")
    assert len(steps) == 1 and steps[0][1].endswith("[gpu]")


def test_macos_uses_pypi_torch(setup_script):
    steps = setup_script.plan_install(nvidia=False, diarization=True, system="Darwin")
    assert len(steps) == 1 and steps[0][1].endswith("[diarize]")


def test_the_cloud_client_is_an_extra_when_wanted(setup_script):
    steps = setup_script.plan_install(nvidia=False, diarization=False, system="Linux", cloud=True)
    assert steps == [["-e", f"{setup_script.ROOT}[cloud]"]]
    full = setup_script.plan_install(nvidia=True, diarization=True, system="Linux", cloud=True)
    assert full[-1][1].endswith("[gpu,diarize,cloud]")


def test_modal_credentials_detection_never_reads_values(setup_script, monkeypatch, tmp_path):
    for var in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / "none.toml"))
    assert not setup_script.modal_credentials_present()
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    assert not setup_script.modal_credentials_present()  # half a credential is not enough
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-x")
    assert setup_script.modal_credentials_present()
    for var in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(var)
    (tmp_path / "none.toml").write_text("[default]\n")
    assert setup_script.modal_credentials_present()


def test_dry_run_adds_the_cloud_client_when_credentials_exist(setup_script, monkeypatch, capsys):
    monkeypatch.setattr(setup_script, "modal_credentials_present", lambda: True)
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: ())
    setup_script.main(["--dry-run", "--no-diarization"])
    assert "[cloud]" in capsys.readouterr().out
    setup_script.main(["--dry-run", "--no-diarization", "--no-cloud"])
    assert "[cloud]" not in capsys.readouterr().out


def test_venv_location_is_overridable_and_outside_the_checkout(setup_script, monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIBE_MEDIA_VENV", str(tmp_path / "v"))
    assert setup_script.venv_dir() == tmp_path / "v"
    monkeypatch.delenv("TRANSCRIBE_MEDIA_VENV")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(setup_script.sys, "platform", "linux")
    assert setup_script.venv_dir() == tmp_path / "cache" / "transcribe-media" / "venv"


def test_venv_python_path_per_platform(setup_script, monkeypatch, tmp_path):
    monkeypatch.setattr(setup_script.sys, "platform", "linux")
    assert setup_script.venv_python(tmp_path) == tmp_path / "bin" / "python"
    monkeypatch.setattr(setup_script.sys, "platform", "win32")
    assert setup_script.venv_python(tmp_path) == tmp_path / "Scripts" / "python.exe"


def test_hf_token_detection_never_needs_the_value(setup_script, monkeypatch, tmp_path):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert not setup_script.hf_token_present()
    (tmp_path / "token").write_text("hf_x")
    assert setup_script.hf_token_present()
    (tmp_path / "token").unlink()
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert setup_script.hf_token_present()


def test_dry_run_installs_nothing_and_explains_diarization(setup_script, monkeypatch, capsys):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent")
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: ())
    monkeypatch.setattr(setup_script, "_run", lambda cmd: pytest.fail("dry run must not execute anything"))
    assert setup_script.main(["--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("install -e ")
    assert "Hugging Face token" in captured.err


def test_token_present_turns_speaker_labels_on_automatically(setup_script, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: (object(),))
    setup_script.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "torch torchaudio --index-url" in out and "[gpu,diarize]" in out


def test_no_diarization_flag_overrides_the_token(setup_script, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: ())
    setup_script.main(["--dry-run", "--no-diarization"])
    assert "torch" not in capsys.readouterr().out


def test_cpu_flag_ignores_the_gpu(setup_script, monkeypatch, capsys):
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: (object(),))
    setup_script.main(["--dry-run", "--cpu", "--no-diarization"])
    assert "[gpu" not in capsys.readouterr().out


def test_install_uses_uv_when_present(setup_script, monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(setup_script, "_run", commands.append)
    monkeypatch.setattr(setup_script.shutil, "which", lambda tool: "/usr/bin/uv")
    setup_script.install(tmp_path, [["-e", "/repo"]])
    assert commands[0][:3] == ["/usr/bin/uv", "pip", "install"] and "--python" in commands[0]


def test_install_falls_back_to_pip(setup_script, monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(setup_script, "_run", commands.append)
    monkeypatch.setattr(setup_script.shutil, "which", lambda tool: None)
    setup_script.install(tmp_path, [["-e", "/repo"]])
    assert commands[0][1:4] == ["-m", "pip", "install"]


def test_old_python_without_uv_gets_a_clear_instruction(setup_script, monkeypatch, tmp_path):
    monkeypatch.setattr(setup_script.shutil, "which", lambda tool: None)
    monkeypatch.setattr(setup_script.sys, "version_info", (3, 9, 0))
    with pytest.raises(SystemExit, match="uv"):
        setup_script.create_venv(tmp_path / "v")


# --- launcher + environment stamp ------------------------------------------- #

@pytest.fixture
def launcher():
    return load("transcribe")


def test_environment_is_not_ready_until_setup_finishes(setup_script, monkeypatch, tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    monkeypatch.setattr(setup_script.sys, "platform", "linux")
    assert not setup_script.environment_ready(venv)  # a bare venv, as left by an interrupted first install
    setup_script._write_marker(venv, [["-e", "x"]])
    assert setup_script.environment_ready(venv)


def test_a_moved_checkout_or_changed_dependencies_invalidate_the_environment(setup_script, monkeypatch, tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    monkeypatch.setattr(setup_script.sys, "platform", "linux")
    setup_script._write_marker(venv, [])
    monkeypatch.setattr(setup_script, "ROOT", tmp_path / "elsewhere")
    assert not setup_script.environment_ready(venv)
    monkeypatch.undo()
    monkeypatch.setattr(setup_script.sys, "platform", "linux")
    monkeypatch.setattr(setup_script, "_pyproject_digest", lambda: "changed")
    assert not setup_script.environment_ready(venv)


def test_a_failed_install_is_reported_cleanly_and_leaves_no_ready_marker(setup_script, monkeypatch, tmp_path, capsys):
    import subprocess

    monkeypatch.setenv("TRANSCRIBE_MEDIA_VENV", str(tmp_path / "v"))
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: ())
    monkeypatch.setattr(setup_script, "create_venv", lambda venv: (venv / "bin").mkdir(parents=True))

    def boom(venv, steps):
        raise subprocess.CalledProcessError(2, "pip")

    monkeypatch.setattr(setup_script, "install", boom)
    assert setup_script.main(["--no-diarization", "--no-cloud"]) == 1
    assert "installation failed" in capsys.readouterr().err
    assert not (tmp_path / "v" / setup_script.MARKER).exists()


def test_a_successful_setup_writes_the_marker_last(setup_script, monkeypatch, tmp_path):
    venv = tmp_path / "v"
    monkeypatch.setenv("TRANSCRIBE_MEDIA_VENV", str(venv))
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: ())
    monkeypatch.setattr(setup_script, "create_venv", lambda path: (path / "bin").mkdir(parents=True))
    monkeypatch.setattr(setup_script, "install", lambda path, steps: (path / "bin" / "python").write_text(""))
    monkeypatch.setattr(setup_script.sys, "platform", "linux")
    assert setup_script.main(["--no-diarization", "--no-cloud"]) == 0
    assert setup_script.environment_ready(venv)


def test_launcher_replaces_itself_on_posix_and_isolates_imports(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.env_setup, "environment_ready", lambda venv: True)
    monkeypatch.setattr(launcher.env_setup, "venv_python", lambda venv: tmp_path / "python")
    seen = {}

    def fake_execv(path, argv):
        seen.update(path=path, argv=argv)
        raise SystemExit(0)  # the real execv replaces the process and never returns

    monkeypatch.setattr(launcher.os, "name", "posix")
    monkeypatch.setattr(launcher.os, "execv", fake_execv)
    with pytest.raises(SystemExit):
        launcher.main(["talk.mp3", "--language", "de"])
    # -I: a file named like a standard module in the working folder must not be imported.
    assert seen["argv"] == [str(tmp_path / "python"), "-I", "-m", "transcribe_media", "talk.mp3", "--language", "de"]


def test_launcher_on_windows_waits_for_the_child_and_survives_ctrl_c(launcher, monkeypatch, tmp_path):
    class Child:
        calls = 0

        def wait(self, timeout=None):
            Child.calls += 1
            if Child.calls == 1:
                raise KeyboardInterrupt
            return 130

    monkeypatch.setattr(launcher.os, "name", "nt")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda command: Child())
    assert launcher.run_package(tmp_path / "python", ["x"]) == 130


def test_launcher_stops_a_child_that_ignores_ctrl_c(launcher, monkeypatch, tmp_path):
    import subprocess

    class Stubborn:
        terminated = False

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            raise subprocess.TimeoutExpired("x", timeout)

        def terminate(self):
            Stubborn.terminated = True

    monkeypatch.setattr(launcher.os, "name", "nt")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda command: Stubborn())
    assert launcher.run_package(tmp_path / "python", []) == launcher.INTERRUPTED and Stubborn.terminated


def test_launcher_repairs_an_unready_environment_first(launcher, monkeypatch, tmp_path):
    state = {"ready": False}
    monkeypatch.setattr(launcher.env_setup, "environment_ready", lambda venv: state["ready"])
    monkeypatch.setattr(launcher.env_setup, "venv_python", lambda venv: tmp_path / "python")
    monkeypatch.delenv("TRANSCRIBE_MEDIA_NO_AUTOSETUP", raising=False)
    monkeypatch.setattr(launcher.env_setup, "main", lambda argv: state.update(ready=True) or 0)
    ran = []
    monkeypatch.setattr(launcher, "run_package", lambda python, args: ran.append(args) or 0)
    assert launcher.main(["x.mp3"]) == 0 and state["ready"] and ran == [["x.mp3"]]


def test_launcher_can_refuse_to_install_anything(launcher, monkeypatch, capsys):
    monkeypatch.setattr(launcher.env_setup, "environment_ready", lambda venv: False)
    monkeypatch.setenv("TRANSCRIBE_MEDIA_NO_AUTOSETUP", "1")
    monkeypatch.setattr(launcher.env_setup, "main", lambda argv: pytest.fail("must not install"))
    assert launcher.main(["x.mp3"]) == launcher.EXIT_NO_ENVIRONMENT
    assert "setup.py" in capsys.readouterr().err


def test_launcher_reports_a_failed_setup_with_its_own_exit_code(launcher, monkeypatch):
    monkeypatch.setattr(launcher.env_setup, "environment_ready", lambda venv: False)
    monkeypatch.delenv("TRANSCRIBE_MEDIA_NO_AUTOSETUP", raising=False)
    monkeypatch.setattr(launcher.env_setup, "main", lambda argv: 1)
    monkeypatch.setattr(launcher, "run_package", lambda *a: pytest.fail("must not run"))
    assert launcher.main(["x.mp3"]) == launcher.EXIT_SETUP_FAILED


# --- the current folder must not be searched for programs on Windows -------------------------------------------- #

def test_windows_stops_searching_the_current_folder_for_programs(monkeypatch):
    from transcribe_media.environment import harden_executable_search

    monkeypatch.setenv("NoDefaultCurrentDirectoryInExePath", "x")
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath")  # so the test leaves the process as it found it
    harden_executable_search("linux")
    harden_executable_search("darwin")
    assert "NoDefaultCurrentDirectoryInExePath" not in os.environ  # not a Windows concept
    harden_executable_search("win32")
    assert os.environ["NoDefaultCurrentDirectoryInExePath"] == "1"


def test_setup_hardens_the_search_before_it_starts_any_program(setup_script, monkeypatch):
    """setup.py looks nvidia-smi and uv up by name."""
    order = []
    monkeypatch.setattr(setup_script, "harden_executable_search", lambda: order.append("hardened"))
    monkeypatch.setattr(setup_script, "detect_nvidia_gpus", lambda: order.append("nvidia-smi") or ())
    setup_script.main(["--dry-run", "--no-diarization", "--no-cloud"])
    assert order[:2] == ["hardened", "nvidia-smi"]


def test_the_launcher_hardens_the_search_before_it_checks_or_repairs_anything(launcher, monkeypatch):
    order = []
    monkeypatch.setattr(launcher.env_setup, "harden_executable_search", lambda: order.append("hardened"))
    monkeypatch.setattr(launcher.env_setup, "environment_ready", lambda venv: order.append("checked") or True)
    monkeypatch.setattr(launcher, "run_package", lambda python, args: 0)
    assert launcher.main(["x.mp3"]) == 0
    assert order == ["hardened", "checked"]
