import importlib.util
import json
import time

import pytest

_HAS_IMAGE_DEPS = (
    importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None
)
pytestmark = pytest.mark.skipif(
    not _HAS_IMAGE_DEPS,
    reason="cv2/numpy are not available",
)

if _HAS_IMAGE_DEPS:
    import cv2
    import numpy as np

    from hmlib.camera.hm_ui_bridge import HmUiProcess


def should_update_controls_from_hm_ui_state(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Max_Speed_X_x10", 2000, 500)

    payload = {
        "version": 1,
        "updated_ms": int(time.time() * 1000),
        "windows": {"Tracker Controls": {"Max_Speed_X_x10": 725}},
        "last_action": None,
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")

    assert ui.get_value("Tracker Controls", "Max_Speed_X_x10") == 725


def should_write_programmatic_updates_to_state_and_spec(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Brightness_Multiplier_x100", 300, 100)
    ui.set_value("Tracker Controls", "Brightness_Multiplier_x100", 145, notify=False)

    state = json.loads(ui.state_path.read_text(encoding="utf-8"))
    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    assert state["windows"]["Tracker Controls"]["Brightness_Multiplier_x100"] == 145
    control = spec["windows"][0]["controls"][0]
    assert control["name"] == "Brightness_Multiplier_x100"
    assert control["value"] == 145
    assert control["default_value"] == 145


def should_not_change_defaults_for_runtime_notifications(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Brightness_Multiplier_x100", 300, 100)
    ui.set_value("Tracker Controls", "Brightness_Multiplier_x100", 145, notify=True)

    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    control = spec["windows"][0]["controls"][0]
    assert control["value"] == 145
    assert control["default_value"] == 100


def should_publish_grouping_and_system_defaults(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Overshoot_Stop_Delay_Frames", 60, 8)
    ui.add_slider("Tracker Controls", "Stitch_Rotate_Degrees", 180, 90)
    ui.set_system_defaults({"Tracker Controls": {"Overshoot_Stop_Delay_Frames": 3}})

    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    controls = {control["name"]: control for control in spec["windows"][0]["controls"]}
    assert controls["Overshoot_Stop_Delay_Frames"]["view"] == "Final"
    assert controls["Overshoot_Stop_Delay_Frames"]["group"] == "Breakaway Tracking"
    assert controls["Overshoot_Stop_Delay_Frames"]["default_value"] == 8
    assert controls["Overshoot_Stop_Delay_Frames"]["system_default_value"] == 3
    assert controls["Stitch_Rotate_Degrees"]["view"] == "Stitched"
    assert controls["Stitch_Rotate_Degrees"]["group"] == "Alignment"


def should_consume_hm_ui_actions_once(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Apply_To_Follower_Box", 1, 1)

    payload = {
        "version": 1,
        "updated_ms": int(time.time() * 1000),
        "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 1}},
        "last_action": {"seq": 1, "kind": "save"},
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")

    assert ui.consume_actions() == ["save"]
    assert ui.consume_actions() == []


def should_publish_preview_frame(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 200

    ui.publish_preview(frame, min_interval_seconds=0)

    assert ui.preview_path.exists()
    decoded = cv2.imread(str(ui.preview_path), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (24, 32)


def should_publish_named_preview_frames_independently(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None
    ui.add_window("Stitch Alignment")
    stitched = np.zeros((24, 32, 3), dtype=np.uint8)
    final = np.zeros((18, 30, 3), dtype=np.uint8)

    ui.publish_preview(stitched, name="Stitched", min_interval_seconds=0)
    ui.publish_preview(final, name="Final", min_interval_seconds=0)

    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    assert [preview["name"] for preview in spec["previews"]] == ["Stitched", "Final"]
    assert cv2.imread(str(ui.preview_paths["Stitched"])).shape[:2] == (24, 32)
    assert cv2.imread(str(ui.preview_paths["Final"])).shape[:2] == (18, 30)


def should_publish_latest_preview_frame_from_batch(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    frame = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    frame[0, :, :, 1] = 200
    frame[1, :, :, 2] = 220

    ui.publish_preview(frame, min_interval_seconds=0)

    decoded = cv2.imread(str(ui.preview_path), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (24, 32)
    assert decoded[:, :, 2].mean() > decoded[:, :, 1].mean()
