from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any, Dict

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch is not available")

if _HAS_TORCH:
    import torch


class _DummyDataloader:
    """Minimal dataloader stub so run_mmtrack can build AspenNet once and exit."""

    def __init__(self, batch_size: int = 1, fps: float = 30.0):
        self.batch_size = batch_size
        self.fps = fps

    def __len__(self) -> int:
        # Zero batches so the main loop exits immediately after setup.
        return 0

    def __iter__(self):
        return iter(())


def should_propagate_camera_ui_into_aspen_shared(monkeypatch):
    from hmlib.tasks import tracking

    captured: Dict[str, Any] = {}
    sentinel_controller: object = object()

    # Stub AspenNet so we can inspect the shared dict passed from run_mmtrack.
    class DummyAspenNet(torch.nn.Module):
        def __init__(self, name: str, graph_cfg: Dict[str, Any], shared: Dict[str, Any] | None = None, **_: Any):  # type: ignore[override]
            super().__init__()
            captured["shared"] = dict(shared or {})

        def to(self, *args: Any, **kwargs: Any):  # pragma: no cover - trivial passthrough
            return self

        def forward(self, context: Dict[str, Any]):  # pragma: no cover - not exercised
            return context

        def finalize(self):  # pragma: no cover - not exercised
            pass

    monkeypatch.setattr(tracking, "AspenNet", DummyAspenNet)
    # Avoid filesystem lookups for precomputed CSVs.
    monkeypatch.setattr(tracking, "find_latest_dataframe_file", lambda *a, **k: None)

    dl = _DummyDataloader()

    cfg: Dict[str, Any] = {
        "aspen": {
            "plugins": {},
            "pipeline": {},
        },
        "initial_args": {
            "camera_ui": 1,
        },
        "camera_ui": 1,
        "game_config": {},
        "stitch_rotation_controller": sentinel_controller,
    }

    tracking.run_mmtrack(
        model=None,
        pose_inferencer=None,
        config=cfg,
        dataloader=dl,
        postprocessor=None,
        progress_bar=None,
        device=torch.device("cpu"),
        input_cache_size=1,
        fp16=False,
        no_cuda_streams=True,
        track_mean_mode=None,
        profiler=None,
    )

    shared = captured.get("shared")
    assert isinstance(shared, dict)
    # Ensure the CLI flag is threaded into Aspen shared context for PlayTrackerPlugin.
    assert shared.get("camera_ui") == 1
    # Stitch rotation controller should also be forwarded untouched.
    assert shared.get("stitch_rotation_controller") is sentinel_controller


def should_propagate_camera_ui_initialization_failure(monkeypatch):
    from hmlib.camera.play_tracker import PlayTracker

    tracker = PlayTracker.__new__(PlayTracker)
    tracker._ui_dialogs = {}
    tracker._hm_ui_process = None

    class PartialProcess:
        closed = False

        def close(self) -> None:
            self.closed = True

    partial_process = PartialProcess()

    def fail_to_create_dialog(*args: Any, **kwargs: Any) -> None:
        tracker._hm_ui_process = partial_process
        raise RuntimeError("hm-ui binary is unavailable")

    monkeypatch.setattr(tracker, "_create_ui_dialog", fail_to_create_dialog)

    with pytest.raises(RuntimeError, match="Failed to initialize camera UI controls") as exc_info:
        tracker._init_ui_controls()

    assert str(exc_info.value.__cause__) == "hm-ui binary is unavailable"
    assert partial_process.closed is True
    assert tracker._hm_ui_process is None


def should_round_trip_linked_and_independent_fixed_edge_rotation_controls():
    from hmlib.camera.play_tracker import PlayTracker

    tracker = PlayTracker.__new__(PlayTracker)
    tracker._game_config = {"rink": {"camera": {"fixed_edge_rotation_angle": 12.5}}}
    tracker._ui_dirty_paths = set()
    tracker._ui_window_name = "Tracker Controls"

    assert tracker._fixed_edge_rotation_slider_defaults() == (1, 125, 125)
    tracker._game_config["rink"]["camera"]["fixed_edge_rotation_angle"] = [12.5, 35.5]
    assert tracker._fixed_edge_rotation_slider_defaults() == (0, 125, 355)

    values = {
        "Link_Fixed_Edge_Rotation_Left_Right": 1,
        "Left_Fixed_Edge_Rotation_Angle_x10": 100,
        "Right_Fixed_Edge_Rotation_Angle_x10": 255,
    }
    tracker._fixed_edge_rotation_last_sliders = (100, 100)
    tracker._ui_slider_value = lambda _window, name: values[name]
    tracker._set_ui_slider_value = lambda _window, name, value, **_kwargs: values.__setitem__(
        name, value
    )

    assert tracker._apply_fixed_edge_rotation_controls() == 25.5
    assert values["Left_Fixed_Edge_Rotation_Angle_x10"] == 255
    assert tracker._game_config["rink"]["camera"]["fixed_edge_rotation_angle"] == 25.5

    values.update(
        {
            "Link_Fixed_Edge_Rotation_Left_Right": 0,
            "Left_Fixed_Edge_Rotation_Angle_x10": 125,
            "Right_Fixed_Edge_Rotation_Angle_x10": 355,
        }
    )
    assert tracker._apply_fixed_edge_rotation_controls() == [12.5, 35.5]
    assert tracker._game_config["rink"]["camera"]["fixed_edge_rotation_angle"] == [12.5, 35.5]


def should_replay_tracking_ui_action_snapshots_in_click_order():
    from hmlib.camera.play_tracker import PlayTracker

    class FakeProcess:
        def __init__(self, final_value, events) -> None:
            self.values = {"Tracker Controls": {"Fixed_Angle_x10": final_value}}
            self.events = events

        def control_values(self):
            return {window: dict(values) for window, values in self.values.items()}

        def consume_action_events(self, *, poll: bool):
            assert poll is False
            events, self.events = self.events, []
            return events

        def apply_control_values(self, values) -> None:
            self.values = {window: dict(items) for window, items in values.items()}

    def event(kind: str, value: int):
        return SimpleNamespace(
            kind=kind,
            values={"Tracker Controls": {"Fixed_Angle_x10": value}},
        )

    tracker = PlayTracker.__new__(PlayTracker)
    tracker._camera_ui_enabled = True
    tracker._ui_inited = True
    tracker._ui_controls_dirty = True
    tracker._system_game_config = {"fixed_angle": 0.5}
    tracker._open_game_config = {"fixed_angle": 5.0}
    tracker._render_ui_dialogs = lambda: None
    runtime = {"fixed_angle": 2.0}
    saved = []

    def apply_current() -> bool:
        runtime["fixed_angle"] = (
            tracker._hm_ui_process.values["Tracker Controls"]["Fixed_Angle_x10"] / 10.0
        )
        tracker._ui_controls_dirty = False
        return True

    tracker._apply_current_ui_control_values = apply_current
    tracker._restore_ui_managed_config = lambda source: runtime.update(source)
    tracker._save_ui_config = lambda: saved.append(runtime["fixed_angle"])

    # Save the pre-reset value, then leave the runtime at the reset value.
    tracker._hm_ui_process = FakeProcess(
        final_value=5,
        events=[event("save", 150), event("reset-system", 5)],
    )
    tracker._apply_ui_controls()
    assert saved == [15.0]
    assert runtime["fixed_angle"] == 0.5

    # A post-reset edit must be applied before its following Save.
    tracker._ui_controls_dirty = True
    tracker._hm_ui_process = FakeProcess(
        final_value=100,
        events=[event("reset-system", 5), event("save", 100)],
    )
    tracker._apply_ui_controls()
    assert saved[-1] == 10.0
    assert runtime["fixed_angle"] == 10.0

    # Consecutive resets retain their order and exact source values.
    tracker._ui_controls_dirty = True
    tracker._hm_ui_process = FakeProcess(
        final_value=50,
        events=[event("reset-system", 5), event("reset-open", 50)],
    )
    tracker._apply_ui_controls()
    assert runtime["fixed_angle"] == 5.0
