"""The documentation must describe the tool that exists: flags, exit codes and models."""

import re
from pathlib import Path

import pytest

from transcribe_media import cli, errors
from transcribe_media.cloud import settings

ROOT = Path(__file__).resolve().parent.parent
DOCS = {
    "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
    "SKILL.md": (ROOT / "skills" / "transcribe-media" / "SKILL.md").read_text(encoding="utf-8"),
}
# Options of scripts/setup.py, and of other programs the docs mention.
OTHER_PROGRAMS = {"--diarization", "--no-diarization", "--cloud", "--no-cloud", "--cpu", "--dry-run",
                  "--version", "--help", "--index-url", "--plugin-dir"}


def parser_options():
    return {opt for action in cli.build_parser()._actions for opt in action.option_strings if opt.startswith("--")}


@pytest.mark.parametrize("name", DOCS)
def test_every_option_the_docs_mention_exists(name):
    # "--cloud-*" is a family written as a wildcard, not an option
    mentioned = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*[a-z0-9](?![\w*-])", DOCS[name]))
    unknown = mentioned - parser_options() - OTHER_PROGRAMS
    assert not unknown, f"{name} mentions options that do not exist: {sorted(unknown)}"


def test_the_exit_codes_in_the_skill_match_the_code():
    line = re.search(r"Exit codes:(.*?)\n\n", DOCS["SKILL.md"], re.S).group(1)
    documented = {int(code) for code in re.findall(r"`(\d+)`", line)}
    in_code = {cls.exit_code for cls in errors.TranscribeMediaError.__subclasses__()}
    assert in_code <= documented, f"undocumented exit codes: {sorted(in_code - documented)}"
    assert documented - in_code <= {0, 1, 15, 16, 130}  # success, --check, launcher setup codes, interrupted


def test_the_launchers_own_exit_codes_are_documented():
    launcher = (ROOT / "scripts" / "transcribe.py").read_text(encoding="utf-8")
    codes = {int(code) for code in re.findall(r"^(?:EXIT_\w+|INTERRUPTED) = (\d+)", launcher, re.M)}
    assert codes >= {15, 16, 130}
    line = re.search(r"Exit codes:(.*?)\n\n", DOCS["SKILL.md"], re.S).group(1)
    documented = {int(code) for code in re.findall(r"`(\d+)`", line)}
    assert codes <= documented, f"undocumented launcher exit codes: {sorted(codes - documented)}"


def test_every_gpu_the_docs_offer_is_allowed():
    allowed = set(settings.ALLOWED_GPUS)
    for gpu in re.findall(r"`(T4|L4|A10|L40S|A100|H100)`", DOCS["SKILL.md"]):
        assert gpu in allowed


def test_the_default_cloud_cap_in_the_docs_is_the_real_one():
    from transcribe_media.backends.modal_backend import DEFAULT_MAX_MINUTES

    for text in DOCS.values():
        assert f"default {DEFAULT_MAX_MINUTES}" in text
    assert f"at most {settings.MAX_JOB_SECONDS // 60}" in DOCS["SKILL.md"]


def test_the_license_file_matches_the_declared_license():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+; the project itself supports 3.10

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert project["license"] == "MIT" and project["license-files"] == ["LICENSE"]
    assert license_text.startswith("MIT License\n") and "Permission is hereby granted, free of charge" in license_text
    assert "[MIT](LICENSE)" in DOCS["README.md"] and "No license file" not in DOCS["README.md"]


# --- the plugin packaging ------------------------------------------------------------------------------------ #

def test_the_plugin_manifest_agrees_with_the_package_and_the_skill():
    import json

    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "transcribe-media" == project["name"]
    assert manifest["version"] == project["version"] and manifest["license"] == project["license"]
    skill = DOCS["SKILL.md"]
    assert skill.startswith("---\nname: transcribe-media\n")


def test_the_skill_finds_the_scripts_from_inside_the_plugin_layout():
    skill_dir = ROOT / "skills" / "transcribe-media"
    commands = re.findall(r'python "\$\{CLAUDE_SKILL_DIR\}/([^"]+)"', DOCS["SKILL.md"])
    assert commands, "the skill should run its scripts through ${CLAUDE_SKILL_DIR}"
    for relative in commands:
        assert (skill_dir / relative).is_file(), relative
    assert "<skill-dir>" not in DOCS["SKILL.md"]
