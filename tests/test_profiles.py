import pytest

from transcribe_media.errors import InputError
from transcribe_media.profiles import select_asr_profile

from .conftest import make_gpu, make_hw


def test_rtx5080_gets_large_v3_float16(rtx5080_box):
    p = select_asr_profile(rtx5080_box)
    assert (p.model, p.device, p.compute_type, p.device_index) == ("large-v3", "cuda", "float16", 0)
    assert p.notes == ()


def test_8gb_non_blackwell_gets_float16():
    hw = make_hw(gpus=[make_gpu("RTX 3070 Ti", 8192, (8, 6))])
    assert select_asr_profile(hw).compute_type == "float16"


def test_4gb_non_blackwell_uses_int8_to_keep_large_v3():
    hw = make_hw(gpus=[make_gpu("GTX 1650", 4096, (7, 5))])
    p = select_asr_profile(hw)
    assert (p.model, p.compute_type) == ("large-v3", "int8_float16")
    assert p.notes


def test_4gb_blackwell_never_uses_int8_and_drops_to_medium():
    hw = make_hw(gpus=[make_gpu("RTX 5050", 4096, (12, 0))])
    p = select_asr_profile(hw)
    assert (p.model, p.compute_type) == ("medium", "float16")


def test_tiny_gpu_falls_back_to_cpu_with_explanation():
    hw = make_hw(gpus=[make_gpu("MX150", 2048, (6, 1))], ram_gb=32)
    p = select_asr_profile(hw)
    assert (p.device, p.model, p.compute_type) == ("cpu", "large-v3", "int8")
    assert any("too small" in n for n in p.notes)


@pytest.mark.parametrize(
    "ram_gb, model", [(64, "large-v3"), (8, "large-v3"), (6, "medium"), (4, "medium"), (2, "small")]
)
def test_cpu_model_scales_with_ram(ram_gb, model):
    p = select_asr_profile(make_hw(ram_gb=ram_gb))
    assert (p.device, p.model, p.compute_type) == ("cpu", model, "int8")


def test_apple_silicon_runs_on_cpu_with_note():
    hw = make_hw(system="Darwin", machine="arm64", ram_gb=32)
    p = select_asr_profile(hw)
    assert p.device == "cpu" and any("Apple" in n for n in p.notes)


def test_device_cpu_overrides_present_gpu(rtx5080_box):
    p = select_asr_profile(rtx5080_box, device="cpu")
    assert p.device == "cpu"


def test_device_cuda_without_gpu_is_an_error():
    with pytest.raises(InputError, match="no NVIDIA GPU"):
        select_asr_profile(make_hw(), device="cuda")


def test_unknown_device_is_an_error():
    with pytest.raises(InputError):
        select_asr_profile(make_hw(), device="tpu")


def test_model_override(rtx5080_box):
    assert select_asr_profile(rtx5080_box, model="medium").model == "medium"


def test_explicit_int8_on_blackwell_is_rejected(rtx5080_box):
    with pytest.raises(InputError, match="Blackwell"):
        select_asr_profile(rtx5080_box, compute_type="int8_float16")


def test_explicit_int8_on_cpu_is_fine():
    assert select_asr_profile(make_hw(), compute_type="int8").compute_type == "int8"


def test_multi_gpu_picks_largest_and_reports_its_index():
    hw = make_hw(gpus=[make_gpu("small", 8000, (8, 6), index=0), make_gpu("big", 24000, (8, 9), index=1)])
    assert select_asr_profile(hw).device_index == 1


def test_an_explicit_cuda_request_on_a_card_that_is_too_small_is_an_error():
    hw = make_hw(gpus=[make_gpu("NVIDIA GeForce GT 1030", 2048, (6, 1))])
    with pytest.raises(InputError, match="too small"):
        select_asr_profile(hw, device="cuda")


def test_the_same_card_quietly_uses_the_cpu_when_the_device_is_automatic():
    hw = make_hw(gpus=[make_gpu("NVIDIA GeForce GT 1030", 2048, (6, 1))])
    profile = select_asr_profile(hw)
    assert profile.device == "cpu" and any("too small" in note for note in profile.notes)
