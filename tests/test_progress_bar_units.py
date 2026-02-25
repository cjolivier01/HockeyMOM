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
