from __future__ import annotations


def should_progress_bar_scale_completed_and_total_by_units_per_iter():
    # Avoid starting rich Live UI; we only want to assert the computed values
    # passed into RichProgress.update().
    from hmlib.utils.progress_bar import ProgressBar

    pb = ProgressBar(total=5, units_per_iter=4)
    # Force refresh() to call into _render_rich() immediately.
    pb._start_threshold = 0  # type: ignore[attr-defined]

    called = {}

    def _fake_update(*args, **kwargs):
        called.update(kwargs)
        return None

    class _FakeLive:
        def update(self, *args, **kwargs):
            return None

    pb._ensure_rich_started = lambda: None  # type: ignore[assignment]
    pb._build_layout = lambda: None  # type: ignore[assignment]
    pb._live = _FakeLive()
    pb._progress.update = _fake_update  # type: ignore[assignment]

    # Simulate 3 completed iterations out of 5.
    pb._counter = 3  # type: ignore[attr-defined]
    pb.refresh(final=False)

    assert called.get("completed") == 12
    assert called.get("total") == 20


def should_throttle_progress_bar_refreshes_to_once_per_second_when_fast(monkeypatch):
    from hmlib.utils import progress_bar as progress_bar_module
    from hmlib.utils.progress_bar import ProgressBar

    pb = ProgressBar(
        total=121,
        iterator=iter(range(121)),
        update_rate=20,
        min_refresh_interval_seconds=1.0,
    )
    pb._start_threshold = 0  # type: ignore[attr-defined]

    rendered: list[tuple[int, bool]] = []

    monkeypatch.setattr(progress_bar_module.time, "monotonic", lambda: pb._counter / 100.0)
    pb._render_rich = lambda final=False: rendered.append((pb._counter, final))  # type: ignore[assignment]

    for _ in pb:
        pass

    assert rendered == [(0, False), (100, False), (121, True)]


def should_keep_frame_based_refresh_interval_when_processing_is_slow(monkeypatch):
    from hmlib.utils import progress_bar as progress_bar_module
    from hmlib.utils.progress_bar import ProgressBar

    pb = ProgressBar(
        total=41,
        iterator=iter(range(41)),
        update_rate=20,
        min_refresh_interval_seconds=1.0,
    )
    pb._start_threshold = 0  # type: ignore[attr-defined]

    rendered: list[tuple[int, bool]] = []

    monkeypatch.setattr(progress_bar_module.time, "monotonic", lambda: float(pb._counter))
    pb._render_rich = lambda final=False: rendered.append((pb._counter, final))  # type: ignore[assignment]

    for _ in pb:
        pass

    assert rendered == [(0, False), (20, False), (40, False), (41, True)]


def should_delete_nested_key_and_prune_empty_dicts():
    from hmlib.stitching import configure_stitching

    cfg = {
        "rink": {
            "scoreboard": {
                "perspective_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "scoreboard_scale": 1.0,
            }
        }
    }
    assert configure_stitching._delete_nested_key(cfg, ["rink", "scoreboard"]) is True
    assert cfg == {}


def should_resolve_global_model_refs_in_aspen_configs():
    from hmlib.config import (
        get_config,
        get_nested_value,
        load_yaml_files_ordered,
        resolve_global_refs,
    )

    base = get_config(resolve_globals=False)
    merged = load_yaml_files_ordered(
        [
            "config/aspen/tracking_pose_actions.yaml",
        ],
        base=base,
    )
    resolve_global_refs(merged)

    assert get_nested_value(
        merged, "aspen.plugins.pose_factory.params.pose_config"
    ) == get_nested_value(merged, "model.pose_coco.config")
    assert get_nested_value(
        merged, "aspen.plugins.pose_factory.params.pose_checkpoint"
    ) == get_nested_value(merged, "model.pose_coco.checkpoint")
    assert get_nested_value(
        merged, "aspen.plugins.action_factory.params.action_config"
    ) == get_nested_value(merged, "model.action_posec3d.config")
    assert get_nested_value(
        merged, "aspen.plugins.action_factory.params.action_checkpoint"
    ) == get_nested_value(merged, "model.action_posec3d.checkpoint")


def should_clean_stitch_game_artifacts_delete_files_and_cached_config(monkeypatch, tmp_path):
    from hmlib.stitching import configure_stitching

    # Create a few representative artifacts.
    (tmp_path / "hm_project.pto").touch()
    (tmp_path / "mapping_0000.tif").touch()
    cam_dir = tmp_path / "cam1"
    cam_dir.mkdir()
    (cam_dir / "left.mp4").touch()
    (cam_dir / "left.png").touch()

    saved = {}

    def _fake_get_game_config_private(game_id: str):
        return {
            "stitching": {
                "stitch_frame_time": "00:00:07",
                "frame_offsets": {"left": 1.0, "right": 2.0},
            },
            "rink": {
                "scoreboard": {
                    "perspective_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "scoreboard_scale": 1.25,
                }
            },
        }

    def _fake_save_private_config(game_id: str, data, verbose: bool = True):
        saved["game_id"] = game_id
        saved["data"] = data

    monkeypatch.setattr(
        configure_stitching, "get_game_config_private", _fake_get_game_config_private
    )
    monkeypatch.setattr(configure_stitching, "save_private_config", _fake_save_private_config)

    removed = configure_stitching.clean_stitch_game_artifacts(
        game_id="test-game", game_dir=tmp_path
    )

    assert removed == 3
    assert not (tmp_path / "hm_project.pto").exists()
    assert not (tmp_path / "mapping_0000.tif").exists()
    assert not (cam_dir / "left.png").exists()
    assert saved.get("game_id") == "test-game"
    assert saved.get("data") == {
        "stitching": {"stitch_frame_time": "00:00:07"},
        "rink": {"scoreboard": {"scoreboard_scale": 1.25}},
    }


def should_normalize_legacy_game_stitching_keys_to_root_level():
    from hmlib.config import normalize_runtime_config

    cfg = {
        "stitching": {"stitch_frame_time": "00:00:07"},
        "game": {
            "stitching": {
                "stitch-frame-time": "00:00:03",
                "stitch-rotate-degrees": 5.0,
                "frame_offsets": {"left": 1.0, "right": 2.0},
            },
            "rgb_add": {"left": [1.0, 2.0, 3.0]},
        },
    }

    normalize_runtime_config(cfg)

    assert cfg["stitching"]["stitch_frame_time"] == "00:00:07"
    assert cfg["stitching"]["post_stitch_rotate_degrees"] == 5.0
    assert cfg["stitching"]["frame_offsets"] == {"left": 1.0, "right": 2.0}
    assert cfg["stitching"]["rgb_add"] == {"left": [1.0, 2.0, 3.0]}
    assert "stitching" not in cfg.get("game", {})
    assert "rgb_add" not in cfg.get("game", {})


def should_sync_stitch_frame_time_state_clean_and_persist_when_it_changes(monkeypatch, tmp_path):
    import copy

    from hmlib.stitching import configure_stitching

    saved = {}
    cleaned = {}
    private_cfg = {
        "stitching": {
            "stitch_frame_time": "00:00:03",
            "frame_offsets": {"left": 1.0, "right": 2.0},
        }
    }

    monkeypatch.setattr(
        configure_stitching,
        "get_game_config_private",
        lambda game_id: copy.deepcopy(private_cfg),
    )

    def _fake_clean(game_id: str, game_dir):
        cleaned["game_id"] = game_id
        cleaned["game_dir"] = str(game_dir)
        return 0

    def _fake_save_private_config(game_id: str, data, verbose: bool = True):
        saved["game_id"] = game_id
        saved["data"] = data

    monkeypatch.setattr(configure_stitching, "clean_stitch_game_artifacts", _fake_clean)
    monkeypatch.setattr(configure_stitching, "save_private_config", _fake_save_private_config)

    runtime_cfg = {}
    changed = configure_stitching.sync_stitch_frame_time_state(
        game_id="test-game",
        game_dir=tmp_path,
        stitch_frame_time="00:00:07",
        game_config=runtime_cfg,
    )

    assert changed is True
    assert cleaned == {"game_id": "test-game", "game_dir": str(tmp_path)}
    assert runtime_cfg == {"stitching": {"stitch_frame_time": "00:00:07"}}
    assert saved["game_id"] == "test-game"
    assert saved["data"]["stitching"]["stitch_frame_time"] == "00:00:07"


def should_clean_exit_before_configuring(monkeypatch, tmp_path):
    import types

    import hmlib.cli.stitch as stitch_cli

    called = {}

    def _fake_clean(game_id: str, game_dir):
        called["game_id"] = game_id
        called["game_dir"] = str(game_dir)
        return 0

    def _fail_configure_game_videos(*args, **kwargs):
        raise AssertionError("Expected --clean-only to exit before configuring")

    monkeypatch.setattr(stitch_cli, "clean_stitch_game_artifacts", _fake_clean)
    monkeypatch.setattr(stitch_cli, "configure_game_videos", _fail_configure_game_videos)

    args = types.SimpleNamespace(
        force=False,
        clean=True,
        game_id="test-game",
        video_dir=str(tmp_path),
    )
    stitch_cli._main(args)

    assert called.get("game_id") == "test-game"


def should_force_clean_before_configuring(monkeypatch, tmp_path):
    import types

    import hmlib.cli.stitch as stitch_cli

    order: list[str] = []

    def _fake_clean(game_id: str, game_dir):
        order.append("clean")
        return 0

    def _fake_configure_game_videos(*args, **kwargs):
        order.append("configure_game_videos")
        raise SystemExit(0)

    monkeypatch.setattr(stitch_cli, "clean_stitch_game_artifacts", _fake_clean)
    monkeypatch.setattr(stitch_cli, "configure_game_videos", _fake_configure_game_videos)

    args = types.SimpleNamespace(
        force=True,
        clean=False,
        game_id="test-game",
        video_dir=str(tmp_path),
        single_file=0,
        ice_rink_inference_scale=None,
    )
    try:
        stitch_cli._main(args)
    except SystemExit:
        pass

    assert order == ["clean", "configure_game_videos"]
