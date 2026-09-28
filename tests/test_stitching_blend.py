"""Seam blend selection: vocabulary, resolution, and what reaches the stitcher."""

from __future__ import annotations

import argparse

import pytest
from stitching_fixtures import write_generation

from hmlib.stitching.blend import (
    BLEND_MODES,
    DEFAULT_BLEND_LEVELS,
    DEFAULT_BLEND_MODE,
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


def should_reject_a_mode_the_selected_renderer_cannot_run():
    from hmlib.stitching.blend import PYTHON_BLEND_MODES

    for mode in GPU_BLEND_MODES:
        assert BlendSettings(mode=mode).require_gpu_mode().mode == mode
    for mode in PYTHON_BLEND_MODES:
        assert BlendSettings(mode=mode).require_python_mode().mode == mode
    # multiblend names the calibration-time enblend/multiblend binaries; neither
    # video path can render it, so neither may suggest the other.
    assert "multiblend" not in set(GPU_BLEND_MODES) | set(PYTHON_BLEND_MODES)
    for check in ("require_gpu_mode", "require_python_mode"):
        with pytest.raises(ValueError, match="cannot be rendered") as excinfo:
            getattr(BlendSettings(mode="multiblend"), check)()
        assert "python-blender" not in str(excinfo.value)
    with pytest.raises(ValueError, match="drop --python-blender"):
        BlendSettings(mode="alpha").require_python_mode()


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
    with pytest.raises(ValueError, match="cannot be rendered by the GPU blender"):
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
    with pytest.raises(ValueError, match="cannot be rendered by the Python blender"):
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


def should_reject_naming_laplacian_with_no_pyramid_levels():
    # `mode="gpu-hard-seam"` is how a caller asks for no blending; laplacian with
    # no levels is a mistake that used to render a hard seam under a wrong name.
    with pytest.raises(ValueError, match="at least one pyramid level"):
        BlendSettings(mode="laplacian", max_levels=0)
    with pytest.raises(ValueError, match="must not be negative"):
        BlendSettings(mode="alpha", max_levels=-1)
    assert BlendSettings(mode="gpu-hard-seam", max_levels=0).levels == 0
    # The dataclass canonicalizes, so no caller can hold an alias spelling.
    assert BlendSettings(mode="GPU_HARD_SEAM").mode == "gpu-hard-seam"


def should_reject_a_non_finite_level_count_as_a_value_error():
    # int(inf) raises OverflowError, which is not a ValueError; a YAML `.inf`
    # must not escape as a bare OverflowError from the stitching plugin.
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_blend_settings(max_blend_levels=float("inf"))
    with pytest.raises(ValueError, match="must be a number"):
        normalize_feather_fraction(10**400)


def should_treat_an_explicit_null_blend_mode_as_inherit():
    # HStream writes `blend_mode: null` to mean inherit; str(None) would make it
    # the literal "None" and reject it.
    assert resolve_blend_settings({"blend_mode": None}).mode == DEFAULT_BLEND_MODE
    assert resolve_blend_settings(blend_mode=None).mode == DEFAULT_BLEND_MODE


def should_split_the_renderable_modes_by_blender():
    from hmlib.stitching.blend import PYTHON_BLEND_MODES

    assert "alpha" in GPU_BLEND_MODES and "alpha" not in PYTHON_BLEND_MODES
    # Every renderable mode is a known mode, but not the reverse: multiblend is
    # vocabulary both applications accept and neither video path can run.
    assert set(GPU_BLEND_MODES) | set(PYTHON_BLEND_MODES) < set(BLEND_MODES)


@pytest.mark.parametrize("raw", [None, "", "   "])
def should_read_a_blank_configured_mode_as_unset(raw):
    from hmlib.stitching.blend import config_blend_mode

    # An empty scalar means inherit, as it does elsewhere in the config; it must
    # not become a combo entry, and it must not abort the run.
    assert config_blend_mode(raw) is None
    assert resolve_blend_settings({"blend_mode": raw}).mode == DEFAULT_BLEND_MODE


def should_fold_a_configured_mode_before_comparing_it():
    from hmlib.stitching.blend import config_blend_mode

    # Two spellings of one unknown mode must not read as two different modes.
    assert config_blend_mode("Pyramid") == config_blend_mode(" pyramid ") == "pyramid"
    assert config_blend_mode("GPU_Hard_Seam") == "gpu-hard-seam"


def should_reject_a_laplacian_stitcher_with_no_levels_before_touching_artifacts(
    monkeypatch, tmp_path
):
    blender2, captured = _capture_native(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="at least one pyramid level"):
        blender2.create_stitcher(
            str(tmp_path),
            batch_size=1,
            device=torch.device("cuda"),
            dtype=torch.uint8,
            left_image_size_wh=(4, 3),
            right_image_size_wh=(4, 3),
            python_blender=False,
            blend_mode="laplacian",
            levels=0,
        )
    assert not captured


def should_key_the_geometry_revision_to_the_normalized_blend_mode():
    from hmlib.aspen.plugins.stitching_plugin import StitchingPlugin

    imgs = [torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)]
    blended = torch.zeros(1, 3, 4, 8)

    def revision(**kwargs):
        plugin = StitchingPlugin(**kwargs)
        # Without a PTO or a native revision this falls back to a process-local
        # token, which would mask any difference the blend keys make.
        plugin._geometry_source = lambda _context: {"kind": "test"}
        return plugin._make_geometry_revision(
            context={"shared": {"game_id": "g"}},
            imgs=imgs,
            blended=blended,
            applied_rotation=0.0,
        )

    # The revision drives a mask cache that deletes entries under other
    # revisions, so spellings that render identically must hash identically.
    assert revision(blend_mode="gpu_hard_seam") == revision(blend_mode="GPU-Hard-Seam")
    assert revision(blend_mode=None) == revision(blend_mode="laplacian")
    assert revision(blend_mode="laplacian") != revision(blend_mode="alpha")
    # Only alpha's width moves pixels, so only alpha carries it in the key -
    # which is what leaves every already-cached Laplacian mask valid.
    assert revision(blend_mode="alpha", blend_feather_fraction=0.2) != revision(
        blend_mode="alpha", blend_feather_fraction=0.3
    )
    assert revision(blend_mode="laplacian", blend_feather_fraction=0.2) == revision(
        blend_mode="laplacian", blend_feather_fraction=0.3
    )


def should_reject_an_unrenderable_mode_at_plugin_construction():
    from hmlib.aspen.plugins.stitching_plugin import StitchingPlugin

    # The stitch UI deliberately lets an operator select a mode this path cannot
    # run, so the graph must refuse to build rather than dying on the first batch.
    with pytest.raises(ValueError, match="cannot be rendered by the GPU blender"):
        StitchingPlugin(blend_mode="multiblend")
    with pytest.raises(ValueError, match="cannot be rendered by the Python blender"):
        StitchingPlugin(blend_mode="alpha", python_blender=True)
    with pytest.raises(ValueError, match="Unsupported stitching blend mode"):
        StitchingPlugin(blend_mode="pyramid")
    assert StitchingPlugin(blend_mode="alpha", blend_feather_fraction=0.2)._blend.mode == "alpha"
