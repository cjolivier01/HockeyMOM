from __future__ import annotations

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")

CLASSES = (
    "Blue Line",
    "Center Ice Circle",
    "Center Line",
    "Crease",
    "Faceoff Dot",
    "Field",
    "Goal",
    "Goal Line",
    "Slot Box",
    "Trapezoid",
    "Zone Circle",
)
PALETTE = [
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
]

HEIGHT = 40
WIDTH = 60


def _box_mask(y0: int, y1: int, x0: int, x1: int) -> "torch.Tensor":
    mask = torch.zeros(HEIGHT, WIDTH, dtype=torch.bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _landmarks(labels, masks) -> dict:
    count = len(labels)
    return {
        "labels": torch.tensor(labels),
        "scores": torch.full((count,), 0.9),
        "bboxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]] * count),
        "masks": torch.stack(masks),
        "classes": CLASSES,
        "palette": PALETTE,
    }


@requires_torch
def should_exclude_field_from_landmark_overlays_by_default() -> None:
    from hmlib.segm.rink_landmarks import filter_landmarks

    landmarks = _landmarks([0, 5], [_box_mask(5, 15, 5, 15), _box_mask(0, HEIGHT, 0, WIDTH)])

    filtered = filter_landmarks(landmarks)

    assert filtered is not None
    assert [int(label) for label in filtered["labels"]] == [0]


@requires_torch
def should_keep_only_the_requested_landmark_classes() -> None:
    from hmlib.segm.rink_landmarks import filter_landmarks

    landmarks = _landmarks(
        [0, 3, 5], [_box_mask(5, 15, 5, 15), _box_mask(20, 30, 5, 15), _box_mask(0, 5, 0, 5)]
    )

    filtered = filter_landmarks(landmarks, include=["crease"])

    assert filtered is not None
    assert [int(label) for label in filtered["labels"]] == [3]


@requires_torch
def should_return_none_when_every_landmark_is_filtered_out() -> None:
    from hmlib.segm.rink_landmarks import filter_landmarks

    landmarks = _landmarks([5], [_box_mask(0, HEIGHT, 0, WIDTH)])

    assert filter_landmarks(landmarks) is None


@requires_torch
def should_convert_the_rgb_palette_to_bgr_for_the_overlay() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer

    landmarks = _landmarks([0], [_box_mask(10, 20, 10, 20)])

    layer = build_overlay_layer(HEIGHT, WIDTH, landmarks=landmarks, outline_thickness=0, bgr=True)
    rgb_layer = build_overlay_layer(
        HEIGHT, WIDTH, landmarks=landmarks, outline_thickness=0, bgr=False
    )

    # The layer is cropped to its coverage, so index relative to the box.
    y, x = 15 - layer.box[0], 15 - layer.box[2]
    # "Blue Line" is RGB (220, 20, 60); frames in this codebase are BGR.
    assert layer.color[:, y, x].tolist() == [60, 20, 220]
    assert rgb_layer.color[:, y, x].tolist() == [220, 20, 60]
    assert pytest.approx(float(layer.alpha[0, y, x])) == 0.35


@requires_torch
def should_draw_landmark_outlines_over_their_own_fill() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer

    landmarks = _landmarks([0], [_box_mask(10, 20, 10, 20)])

    layer = build_overlay_layer(
        HEIGHT, WIDTH, landmarks=landmarks, fill_alpha=0.35, outline_alpha=0.95
    )

    y0, _, x0, _ = layer.box
    # The border pixel carries the outline alpha, the interior the fill alpha.
    assert pytest.approx(float(layer.alpha[0, 10 - y0, 10 - x0])) == 0.95
    assert pytest.approx(float(layer.alpha[0, 15 - y0, 15 - x0])) == 0.35


@requires_torch
def should_crop_the_baked_layer_to_the_pixels_it_covers() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer

    layer = build_overlay_layer(
        HEIGHT, WIDTH, rink_mask=_box_mask(5, 35, 10, 50), rink_mask_alpha=0.5
    )

    # Blending the whole panorama for marks covering part of it is the
    # dominant per-frame cost, so the layer carries only its own box.
    assert layer.box == (5, 35, 10, 50)
    assert tuple(layer.color.shape) == (3, 30, 40)
    assert tuple(layer.alpha.shape) == (1, 30, 40)


@requires_torch
def should_not_build_a_layer_that_covers_nothing() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer

    landmarks = _landmarks([0], [_box_mask(10, 20, 10, 20)])

    # Asking for neither a fill nor an outline leaves an all-zero layer; it
    # must not be uploaded and blended over every frame for no visible effect.
    assert (
        build_overlay_layer(HEIGHT, WIDTH, landmarks=landmarks, fill_alpha=0.0, outline_thickness=0)
        is None
    )


@requires_torch
def should_leave_pixels_outside_the_overlay_box_untouched() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer, composite_overlay

    layer = build_overlay_layer(
        HEIGHT, WIDTH, rink_mask=_box_mask(10, 20, 10, 20), rink_mask_alpha=0.5
    )
    img = torch.full((HEIGHT, WIDTH, 3), 100, dtype=torch.uint8)

    out = composite_overlay(img, layer)

    assert out[0, 0].tolist() == [100, 100, 100]
    assert out[15, 15].tolist() != [100, 100, 100]
    # The caller's tensor is shared pipeline state; it must not be mutated.
    assert bool((img == 100).all())


@requires_torch
def should_skip_masks_that_do_not_match_the_overlay_plane() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer

    wrong_size = torch.ones(HEIGHT + 5, WIDTH, dtype=torch.bool)

    assert build_overlay_layer(HEIGHT, WIDTH, rink_mask=wrong_size) is None


@requires_torch
def should_return_none_when_there_is_nothing_to_draw() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer

    assert build_overlay_layer(HEIGHT, WIDTH) is None


@requires_torch
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((HEIGHT, WIDTH, 3), torch.uint8),
        ((3, HEIGHT, WIDTH), torch.uint8),
        ((2, HEIGHT, WIDTH, 3), torch.uint8),
        ((2, 3, HEIGHT, WIDTH), torch.float32),
    ],
)
def should_preserve_image_layout_and_dtype_when_compositing(shape, dtype) -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer, composite_overlay

    layer = build_overlay_layer(
        HEIGHT, WIDTH, rink_mask=_box_mask(5, 35, 5, 55), rink_mask_alpha=0.5
    )
    img = torch.full(shape, 100, dtype=dtype)

    out = composite_overlay(img, layer)

    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert bool((out != img).any())


@requires_torch
def should_blend_the_overlay_at_the_requested_alpha() -> None:
    from hmlib.segm.rink_landmarks import build_overlay_layer, composite_overlay

    layer = build_overlay_layer(
        HEIGHT,
        WIDTH,
        rink_mask=_box_mask(0, HEIGHT, 0, WIDTH),
        rink_mask_color=(0, 200, 0),
        rink_mask_alpha=0.5,
    )
    img = torch.zeros((HEIGHT, WIDTH, 3), dtype=torch.uint8)

    out = composite_overlay(img, layer)

    assert out[0, 0].tolist() == [0, 100, 0]


@requires_torch
def should_drop_landmark_instances_below_the_score_threshold() -> None:
    from types import SimpleNamespace

    from hmlib.segm import rink_landmarks

    instances = SimpleNamespace(
        scores=torch.tensor([0.9, 0.2]),
        labels=torch.tensor([0, 3]),
        bboxes=torch.tensor([[0.0, 0.0, 5.0, 5.0], [6.0, 6.0, 9.0, 9.0]]),
        masks=torch.stack([_box_mask(0, 5, 0, 5), _box_mask(6, 9, 6, 9)]),
    )
    model = SimpleNamespace(
        dataset_meta={"classes": CLASSES, "palette": PALETTE},
        pred_instances=instances,
    )

    def fake_inference(_model, _image):
        return SimpleNamespace(pred_instances=instances)

    import sys
    from types import ModuleType

    fake_apis = ModuleType("mmdet.apis")
    fake_apis.inference_detector = fake_inference
    sys.modules["mmdet.apis"] = fake_apis
    try:
        result = rink_landmarks.detect_rink_landmarks(
            torch.zeros((HEIGHT, WIDTH, 3), dtype=torch.uint8), model=model, score_thr=0.5
        )
    finally:
        del sys.modules["mmdet.apis"]

    assert result is not None
    assert [int(label) for label in result["labels"]] == [0]


@requires_torch
def should_summarize_detected_landmarks_by_class() -> None:
    from hmlib.segm.rink_landmarks import summarize_landmarks

    landmarks = _landmarks([0, 0, 3], [_box_mask(0, 2, 0, 2)] * 3)

    assert summarize_landmarks(landmarks) == "3 landmarks: Blue Line x2, Crease x1"
    assert summarize_landmarks(None) == "no rink landmarks detected"


@requires_torch
def should_composite_the_cached_overlay_onto_every_frame() -> None:
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin
    from hmlib.segm.rink_landmarks import build_overlay_layer

    plugin = RinkOverlayPlugin(enabled=True)
    plugin._layer = build_overlay_layer(
        HEIGHT, WIDTH, rink_mask=_box_mask(5, 35, 5, 55), rink_mask_alpha=0.5
    )
    plugin._layer_key = (HEIGHT, WIDTH, None)
    plugin._attempted_key = (HEIGHT, WIDTH, None)
    img = torch.full((1, HEIGHT, WIDTH, 3), 100, dtype=torch.uint8)

    out = plugin.forward({"img": img, "game_id": "game-1"})

    assert bool((out["img"] != img).any())
    assert out["img"].shape == img.shape
    # The layer is uploaded to the frame's device once and reused, not
    # re-sent from the host on every frame.
    assert plugin._device_layer is not None
    cached = plugin._device_layer
    plugin.forward({"img": img, "game_id": "game-1"})
    assert plugin._device_layer is cached


@pytest.fixture
def no_real_models(monkeypatch):
    """Keep the unit suite off the network.

    model.rink_landmarks_segm now defaults to a published release URL, so an
    unpatched call really would download 340 MB and run Mask2Former on the CPU.
    """

    def _absent(**kwargs):
        raise AssertionError("no game directory found for game id")

    monkeypatch.setattr("hmlib.segm.ice_rink.configure_ice_rink_mask", _absent)
    monkeypatch.setattr("hmlib.segm.rink_landmarks.configure_rink_landmarks", _absent)
    return monkeypatch


@requires_torch
def should_discard_a_stale_overlay_when_the_panorama_is_resized(no_real_models) -> None:
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin
    from hmlib.segm.rink_landmarks import build_overlay_layer

    plugin = RinkOverlayPlugin(enabled=True, rink_mask=False, landmarks=True)
    plugin._layer = build_overlay_layer(
        HEIGHT, WIDTH, rink_mask=_box_mask(5, 35, 5, 55), rink_mask_alpha=0.5
    )
    plugin._layer_key = (HEIGHT, WIDTH, None)
    plugin._attempted_key = (HEIGHT, WIDTH, None)
    resized = torch.full((1, HEIGHT * 2, WIDTH * 2, 3), 100, dtype=torch.uint8)

    # The rebuild at the new size fails, so the plugin must fall silent rather
    # than keep painting the old mask in the wrong place.
    assert plugin.forward({"img": resized, "game_id": "game-1"}) == {}
    assert plugin._layer is None


@requires_torch
def should_retry_the_overlay_once_the_panorama_settles_on_a_workable_size(monkeypatch) -> None:
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin

    calls: list[tuple[int, int]] = []

    def _only_small(**kwargs):
        shape = tuple(kwargs["expected_shape"])
        calls.append(shape)
        if shape != (HEIGHT, WIDTH):
            raise RuntimeError("mask unavailable at this size")
        return {"combined_mask": _box_mask(5, 35, 5, 55)}

    monkeypatch.setattr("hmlib.segm.ice_rink.configure_ice_rink_mask", _only_small)
    plugin = RinkOverlayPlugin(
        enabled=True, rink_mask=True, landmarks=False, save_debug_frame=False
    )

    big = torch.full((1, HEIGHT * 2, WIDTH * 2, 3), 100, dtype=torch.uint8)
    assert plugin.forward({"img": big, "game_id": "game-1"}) == {}
    # Latched for that size only, so the failing size is not retried...
    assert plugin.forward({"img": big, "game_id": "game-1"}) == {}
    assert len(calls) == 1

    # ...but a size the models can handle still gets its overlay.
    small = torch.full((1, HEIGHT, WIDTH, 3), 100, dtype=torch.uint8)
    out = plugin.forward({"img": small, "game_id": "game-1"})
    assert bool((out["img"] != small).any())


@requires_torch
def should_leave_the_frame_alone_when_the_overlay_is_switched_off() -> None:
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin

    img = torch.full((1, HEIGHT, WIDTH, 3), 100, dtype=torch.uint8)

    assert RinkOverlayPlugin(enabled=False).forward({"img": img}) == {}
    assert RinkOverlayPlugin(rink_mask=False, landmarks=False).forward({"img": img}) == {}
    assert RinkOverlayPlugin(enabled=True).forward({"game_id": "game-1"}) == {}


@requires_torch
def should_keep_stitching_when_the_overlay_models_are_unavailable(no_real_models) -> None:
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin

    # Both models raise. A decoration failure must never abort the stitch.
    plugin = RinkOverlayPlugin(enabled=True)
    img = torch.full((1, HEIGHT, WIDTH, 3), 100, dtype=torch.uint8)

    assert plugin.forward({"img": img, "game_id": "nope"}) == {}
    # And it must not retry the failing models on every subsequent frame.
    assert plugin._attempted_key == (HEIGHT, WIDTH, None)
    assert plugin.forward({"img": img, "game_id": "nope"}) == {}


@requires_torch
def should_keep_stitching_when_compositing_fails(monkeypatch) -> None:
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin
    from hmlib.segm.rink_landmarks import build_overlay_layer

    plugin = RinkOverlayPlugin(enabled=True, save_debug_frame=False)
    plugin._layer = build_overlay_layer(
        HEIGHT, WIDTH, rink_mask=_box_mask(5, 35, 5, 55), rink_mask_alpha=0.5
    )
    plugin._layer_key = (HEIGHT, WIDTH, None)
    plugin._attempted_key = (HEIGHT, WIDTH, None)

    def _oom(*args, **kwargs):
        raise torch.OutOfMemoryError("no room")

    monkeypatch.setattr("hmlib.segm.rink_landmarks.composite_overlay", _oom)
    img = torch.full((1, HEIGHT, WIDTH, 3), 100, dtype=torch.uint8)

    # A transient OOM on frame N costs the overlay, not the encode.
    assert plugin.forward({"img": img, "game_id": "game-1"}) == {}


@requires_torch
def should_not_persist_the_rink_profile_from_a_preview_overlay(monkeypatch) -> None:
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin

    seen: dict = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return {"combined_mask": _box_mask(5, 35, 5, 55)}

    monkeypatch.setattr("hmlib.segm.ice_rink.configure_ice_rink_mask", _capture)
    plugin = RinkOverlayPlugin(
        enabled=True, rink_mask=True, landmarks=False, save_debug_frame=False
    )
    img = torch.full((1, HEIGHT, WIDTH, 3), 100, dtype=torch.uint8)

    plugin.forward(
        {
            "img": img,
            "game_id": "game-1",
            "camera_input_geometry": {"stitched_geometry_revision": "rev-7"},
        }
    )

    # Drawing a preview must not rewrite rink_mask_<i>.png or drop
    # rink.ice_contours_geometry_revision, and must not reuse a mask baked for
    # a different warp that happens to share this panorama's size.
    assert seen["persist"] is False
    assert seen["geometry_revision"] == "rev-7"


def _stitch_graph_config(argv: list[str]) -> dict:
    """Build the stitch graph config exactly the way stitch_videos() does.

    stitch_videos() does *not* reuse args.game_config: it rebuilds from
    get_config + the stitching graph YAML and then applies only
    ARG_TO_CONFIG_MAP. Anything that relies on hm_opts.init() side effects is
    invisible here, which is why these tests go through the same path.
    """
    import sys

    from hmlib.cli.stitch import _arm_rink_overlay, make_parser
    from hmlib.config import (
        get_config,
        load_yaml_files_ordered,
        normalize_runtime_config,
        resolve_global_refs,
    )
    from hmlib.hm_opts import hm_opts

    saved = sys.argv
    sys.argv = ["stitch", *argv]
    try:
        parser = hm_opts.parser(parser=make_parser())
        args = parser.parse_args()
        args.explicit_arg_names = hm_opts.collect_explicit_arg_names(parser)
        args = hm_opts.init(args, parser=parser)
    finally:
        sys.argv = saved

    config = get_config(game_id=None, resolve_globals=False, ignore_private_config=True)
    aspen = load_yaml_files_ordered(["config/aspen/stitching.yaml"], base=config)
    normalize_runtime_config(aspen)
    hm_opts.apply_arg_config_overrides(
        aspen, args, parser=parser, explicit_arg_names=args.explicit_arg_names
    )
    hm_opts.apply_config_overrides(aspen, args.config_overrides)
    _arm_rink_overlay(aspen)
    resolve_global_refs(aspen)
    return aspen


@requires_torch
def should_place_the_rink_overlay_between_the_stitcher_and_the_camera_crop() -> None:
    plugins = _stitch_graph_config(["--ignore-private-config=1"])["aspen"]["plugins"]

    assert plugins["rink_overlay"]["depends"] == ["stitching"]
    # Annotating upstream of video_out_prep is what puts the overlay in the
    # preview, the camera UI and the encoded file at once.
    assert plugins["apply_camera"]["depends"] == ["rink_overlay"]


@requires_torch
@pytest.mark.parametrize(
    "flag, drawn, quiet",
    [
        ("--plot-ice-mask", "rink_mask", "landmarks"),
        ("--plot-rink-landmarks", "landmarks", "rink_mask"),
    ],
)
def should_arm_the_rink_overlay_from_the_stitch_cli(flag, drawn, quiet) -> None:
    node = _stitch_graph_config([flag, "--ignore-private-config=1"])["aspen"]["plugins"][
        "rink_overlay"
    ]

    assert node["enabled"] is True
    assert node["params"][drawn] is True
    assert node["params"][quiet] is False


@requires_torch
@pytest.mark.parametrize("key", ["plot.plot_ice_mask", "plot.plot_rink_landmarks"])
def should_arm_the_rink_overlay_from_a_config_file(key) -> None:
    from hmlib.cli.stitch import _arm_rink_overlay
    from hmlib.config import get_nested_value, set_nested_value

    # A game or private YAML can ask for an overlay without the flag ever
    # being typed, and ARG_TO_CONFIG_MAP only fires for explicit flags -- so
    # the master switch has to be derived from the final config, not mapped.
    config = _stitch_graph_config(["--ignore-private-config=1"])
    assert get_nested_value(config, "plot.plot_rink_overlay") is False

    set_nested_value(config, key, True)
    _arm_rink_overlay(config)

    assert get_nested_value(config, "plot.plot_rink_overlay") is True


@requires_torch
def should_leave_the_rink_overlay_node_disabled_by_default() -> None:
    node = _stitch_graph_config(["--ignore-private-config=1"])["aspen"]["plugins"]["rink_overlay"]

    # A disabled node is replaced by a no-op stub, so an ordinary stitch pays
    # nothing: no plugin construction, no model import, no per-frame work.
    assert node["enabled"] is False
    assert node["params"]["rink_mask"] is False
    assert node["params"]["landmarks"] is False


@requires_torch
def should_stub_out_the_rink_overlay_when_no_plot_flag_is_given() -> None:
    import hmlib.hm_transforms  # noqa: F401
    import hmlib.transforms  # noqa: F401
    from hmlib.aspen import AspenNet
    from hmlib.aspen.plugins.rink_overlay_plugin import RinkOverlayPlugin

    aspen = _stitch_graph_config(["--ignore-private-config=1"])["aspen"]
    net = AspenNet("default-off", aspen, shared={})
    module = net.node_map["rink_overlay"].module

    assert not isinstance(module, RinkOverlayPlugin)
    assert module.forward({"img": "untouched"}) == {}
    # apply_camera still resolves its dependency through the stub.
    assert ("rink_overlay", "apply_camera") in set(net.graph.edges())


@requires_torch
def should_pass_landmark_tuning_from_the_stitch_cli_to_the_graph() -> None:
    params = _stitch_graph_config(
        [
            "--plot-rink-landmarks",
            "--plot-rink-landmark-labels",
            "--rink-landmarks-score-thr=0.7",
            "--rink-landmarks-inference-scale=0.25",
            "--rink-landmarks-classes=Blue Line,Crease",
            "--ignore-private-config=1",
        ]
    )["aspen"]["plugins"]["rink_overlay"]["params"]

    assert params["label_text"] is True
    assert params["score_thr"] == 0.7
    assert params["inference_scale"] == 0.25
    assert params["classes"] == ["Blue Line", "Crease"]


@requires_torch
def should_route_the_landmark_checkpoint_override_into_the_graph() -> None:
    # get_model_config() re-reads the config from disk, so the checkpoint has
    # to arrive as a plugin param or the CLI override is silently dropped.
    params = _stitch_graph_config(
        [
            "--plot-rink-landmarks",
            "--rink-landmarks-checkpoint=/tmp/custom_landmarks.pth",
            "--ignore-private-config=1",
        ]
    )["aspen"]["plugins"]["rink_overlay"]["params"]

    assert params["checkpoint"] == "/tmp/custom_landmarks.pth"
    assert params["model_config"].endswith(
        "config/models/rink_landmarks/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
    )


@requires_torch
def should_default_the_landmark_checkpoint_to_the_published_release_asset() -> None:
    # A work_dirs path only resolves on the machine that trained the model, so
    # the default has to be the release URL that mmengine can fetch and cache.
    params = _stitch_graph_config(["--plot-rink-landmarks", "--ignore-private-config=1"])["aspen"][
        "plugins"
    ]["rink_overlay"]["params"]

    assert params["checkpoint"].startswith("https://")
    assert params["checkpoint"].endswith("/rink_landmarks_iter_97500.pth")
    assert "work_dirs" not in params["checkpoint"]


@requires_torch
def should_keep_the_stitch_graph_valid_under_cuda_graph_mode() -> None:
    import hmlib.hm_transforms  # noqa: F401
    import hmlib.transforms  # noqa: F401
    from hmlib.aspen import AspenNet

    aspen = _stitch_graph_config(["--plot-ice-mask", "--ignore-private-config=1"])["aspen"]

    # rink_overlay feeds apply_camera, so marking it a deferred CUDA-graph sink
    # would trip _validate_cuda_graph_deferred_nodes and abort the whole run.
    net = AspenNet("cuda-graph-check", aspen, shared={})
    net.set_cuda_graph_enabled(True)

    assert "rink_overlay" not in net.shared["aspen_cuda_graph_deferred_plugins"]
    assert "rink_overlay" not in net.shared["aspen_cuda_graph_supported_plugins"]
