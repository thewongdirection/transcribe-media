"""Checks against the real `modal` package (skipped when it is not installed).

The cloud backend is exercised elsewhere with a fake provider; these tests make
sure the names the fake stands in for really exist, so an API change upstream
fails here instead of on someone's first cloud run.
"""

import inspect

import pytest

from transcribe_media.cloud import settings

modal = pytest.importorskip("modal")


def params(callable_):
    return set(inspect.signature(callable_).parameters)


def test_every_provider_call_the_backend_makes_exists():
    assert callable(modal.enable_output) and callable(modal.App.run)
    assert {"name", "required_keys"} <= params(modal.Secret.from_name)
    assert {"gpu", "secrets"} <= params(modal.Function.with_options)
    assert callable(modal.Function.spawn)
    assert "timeout" in params(modal.FunctionCall.get)
    assert "terminate_containers" in params(modal.FunctionCall.cancel)
    for method in ("batch_upload", "read_file", "remove_file", "listdir"):
        assert callable(getattr(modal.Volume, method)), method
    assert "recursive" in params(modal.Volume.remove_file)
    assert callable(modal.config.config.get)


def test_the_exception_types_we_catch_are_distinct_classes():
    # modal's TimeoutError is not the builtin one, so catching only the builtin would miss it.
    assert not issubclass(modal.exception.TimeoutError, TimeoutError)
    assert issubclass(modal.exception.AuthError, Exception)


def test_the_app_module_builds_without_credentials():
    from transcribe_media.cloud import modal_app

    assert modal_app.app.name == settings.APP_NAME
    assert callable(modal_app.run_remote.with_options)
    assert set(settings.DEFAULT_GPUS) <= set(settings.ALLOWED_GPUS)


def test_the_remote_function_is_a_module_level_modal_function():
    # Modal's container finds the function by importing this module and looking it up by
    # name, so it must be a module-level attribute (closures would need serialisation).
    from transcribe_media.cloud import modal_app

    assert isinstance(modal_app.run_remote, modal.Function)


def test_settings_do_not_import_modal():
    import subprocess
    import sys

    code = "import sys; import transcribe_media.cloud.settings; sys.exit('modal' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_the_per_call_options_the_backend_sets_are_accepted_by_modal():
    # The backend always sets a server-side timeout and optionally a GPU and secrets.
    assert {"gpu", "secrets", "timeout"} <= params(modal.Function.with_options)


def test_the_remote_function_refuses_a_job_id_that_could_be_a_path():
    from transcribe_media.cloud import modal_app

    with pytest.warns(UserWarning, match="executing locally"):  # modal's note that no volume is mounted
        with pytest.raises(ValueError, match="invalid job id"):
            modal_app.run_remote.local("../../etc/passwd")


def test_the_container_image_switches_telemetry_off_and_knows_it_is_remote():
    # The image's layers are not inspectable without building it, so check the definition itself.
    from transcribe_media.cloud import modal_app

    source = inspect.getsource(modal_app)
    for setting in ('"PYANNOTE_METRICS_ENABLED": "0"', '"HF_HUB_DISABLE_TELEMETRY": "1"',
                    '"TRANSCRIBE_MEDIA_REMOTE": "1"'):
        assert setting in source
