import json
import time

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
