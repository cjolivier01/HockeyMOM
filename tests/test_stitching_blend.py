"""Seam blend selection: vocabulary, resolution, and what reaches the stitcher."""

from __future__ import annotations

import argparse

import pytest
from stitching_fixtures import write_generation

from hmlib.stitching.blend import (
    BLEND_MODES,
    DEFAULT_BLEND_LEVELS,
    DEFAULT_FEATHER_FRACTION,
    GPU_BLEND_MODES,
    BlendSettings,
    normalize_blend_mode,
    normalize_feather_fraction,
    resolve_blend_settings,
)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("laplacian", "laplacian"),
        ("Laplacian", "laplacian"),
        ("alpha", "alpha"),
        ("ALPHA", "alpha"),
        ("gpu-hard-seam", "gpu-hard-seam"),
        # HockeyMONStream normalizes '_' to '-' and accepts the short spellings.
        ("gpu_hard_seam", "gpu-hard-seam"),
        ("hard-seam", "gpu-hard-seam"),
        ("hard", "gpu-hard-seam"),
        ("multiblend", "multiblend"),
    ],
)
def should_accept_the_same_blend_spellings_as_hstream(raw, expected):
    assert normalize_blend_mode(raw) == expected


@pytest.mark.parametrize("raw", ["", "pyramid", "alpha-blend", "laplace", None])
def should_reject_an_unknown_blend_mode_instead_of_falling_back(raw):
    with pytest.raises(ValueError, match="Unsupported stitching blend mode"):
        normalize_blend_mode(raw)


@pytest.mark.parametrize("raw", [-0.01, 1.5, float("nan"), float("inf"), True, [0.1]])
def should_reject_an_out_of_range_feather_fraction(raw):
    with pytest.raises(ValueError, match="blend_feather_fraction"):
        normalize_feather_fraction(raw)


def should_accept_an_explicit_zero_feather_fraction():
    # 0 is a legal width meaning a hard seam, and must stay distinguishable from unset.
    assert normalize_feather_fraction(0) == 0.0
    assert resolve_blend_settings({"blend_feather_fraction": 0.0}).feather_fraction == 0.0
    assert resolve_blend_settings({}).feather_fraction == DEFAULT_FEATHER_FRACTION


def should_resolve_blend_settings_from_config_with_overrides_winning():
    config = {"blend_mode": "laplacian", "blend_feather_fraction": 0.2, "max_blend_levels": 7}
    assert resolve_blend_settings(config) == BlendSettings("laplacian", 0.2, 7)
    assert resolve_blend_settings(config, blend_mode="alpha").mode == "alpha"
    assert resolve_blend_settings(config, blend_feather_fraction=0.01).feather_fraction == 0.01
    assert resolve_blend_settings(config, max_blend_levels=3).max_levels == 3


def should_default_a_missing_or_non_positive_level_count():
    assert resolve_blend_settings({}).max_levels == DEFAULT_BLEND_LEVELS
    assert resolve_blend_settings({"max_blend_levels": 0}).max_levels == DEFAULT_BLEND_LEVELS
    assert resolve_blend_settings({"max_blend_levels": -1}).max_levels == DEFAULT_BLEND_LEVELS


@pytest.mark.parametrize(
    "mode,expected",
    [("laplacian", 9), ("alpha", 0), ("gpu-hard-seam", 0), ("multiblend", 0)],
)
def should_only_spend_pyramid_levels_on_laplacian(mode, expected):
    assert BlendSettings(mode=mode, max_levels=9).levels == expected


def should_reject_a_gpu_unrenderable_mode():
    assert set(GPU_BLEND_MODES) == set(BLEND_MODES) - {"multiblend"}
    for mode in GPU_BLEND_MODES:
        assert BlendSettings(mode=mode).require_gpu_mode().mode == mode
    with pytest.raises(ValueError, match="no GPU implementation"):
        BlendSettings(mode="multiblend").require_gpu_mode()


def _capture_native(monkeypatch, tmp_path):
    from hmlib.stitching import blender2

    write_generation(tmp_path)
    captured = {}

    def native(directory, batch_size, levels, *sizes, **kwargs):
        captured.update(levels=levels, **kwargs)
        return "stitcher"

    for name in (
        "CudaStitchPanoU8",
        "CudaStitchPanoF32",
        "CudaStitchPanoNU8",
        "CudaStitchPanoNF32",
    ):
        monkeypatch.setattr(blender2, name, native)
    return blender2, captured


@pytest.mark.parametrize("use_cuda_pano_n", [False, True])
def should_forward_alpha_and_its_feather_width_to_the_native_stitcher(
    monkeypatch, tmp_path, use_cuda_pano_n
):
    blender2, captured = _capture_native(monkeypatch, tmp_path)
    assert (
        blender2.create_stitcher(
            str(tmp_path),
            batch_size=1,
            device=torch.device("cuda"),
            dtype=torch.uint8,
            left_image_size_wh=(4, 3),
            right_image_size_wh=(4, 3),
            python_blender=False,
            use_cuda_pano_n=use_cuda_pano_n,
            blend_mode="alpha",
            feather_fraction=0.125,
            levels=11,
        )
        == "stitcher"
    )
    assert captured["blend_mode"] == "alpha"
    assert captured["feather_fraction"] == 0.125
    # Alpha carries its width in the fraction; the pyramid depth is not spent.
    assert captured["levels"] == 0


def should_keep_laplacian_levels_and_the_historical_hard_seam_encoding(monkeypatch, tmp_path):
    blender2, captured = _capture_native(monkeypatch, tmp_path)
    common = dict(
        batch_size=1,
        device=torch.device("cuda"),
        dtype=torch.uint8,
        left_image_size_wh=(4, 3),
        right_image_size_wh=(4, 3),
        python_blender=False,
        levels=7,
    )
    blender2.create_stitcher(str(tmp_path), blend_mode="laplacian", **common)
    assert (captured["blend_mode"], captured["levels"]) == ("laplacian", 7)
    blender2.create_stitcher(str(tmp_path), blend_mode="gpu_hard_seam", **common)
    assert (captured["blend_mode"], captured["levels"]) == ("gpu-hard-seam", 0)


def should_refuse_multiblend_on_the_gpu_path_instead_of_rendering_a_hard_seam(
    monkeypatch, tmp_path
):
    blender2, _ = _capture_native(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no GPU implementation"):
        blender2.create_stitcher(
            str(tmp_path),
            batch_size=1,
            device=torch.device("cuda"),
            dtype=torch.uint8,
            left_image_size_wh=(4, 3),
            right_image_size_wh=(4, 3),
            python_blender=False,
            blend_mode="multiblend",
        )


def should_refuse_alpha_on_the_python_blender_path(monkeypatch, tmp_path):
    blender2, _ = _capture_native(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="GPU-only"):
        blender2.create_stitcher(
            str(tmp_path),
            batch_size=1,
            device=torch.device("cuda"),
            dtype=torch.float32,
            left_image_size_wh=(4, 3),
            right_image_size_wh=(4, 3),
            python_blender=True,
            use_cuda_pano=False,
            blend_mode="alpha",
        )


def should_expose_the_blend_choice_on_the_command_line():
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--blend-mode", "GPU_HARD_SEAM", "--blend-feather-fraction", "0.2"])
    assert args.blend_mode == "gpu-hard-seam"
    assert args.blend_feather_fraction == 0.2
    with pytest.raises(SystemExit):
        parser.parse_args(["--blend-mode", "pyramid"])


def should_carry_the_blend_keys_in_the_shared_baseline():
    from hmlib.config import get_config

    stitching = get_config(game_id=None)["stitching"]
    assert normalize_blend_mode(stitching["blend_mode"]) == "laplacian"
    assert normalize_feather_fraction(stitching["blend_feather_fraction"]) == 0.05
