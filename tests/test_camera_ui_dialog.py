from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from hmlib.camera.camera_ui import CameraControlDialog


def should_scale_control_layout_with_dialog_size():
    dialog = CameraControlDialog("Tracker Controls")
    dialog.add_slider("Stop_Direction_Change_Delay_Frames", 60, 10)
    dialog.add_slider("Cancel_Stop_On_Opposite_Direction", 1, 0)
    dialog.add_slider("Time_To_Dest_Speed_Limit_Frames", 120, 30)

    dialog.render((420, 240))
    small_layout = dialog._layouts["Stop_Direction_Change_Delay_Frames"]
    small_row_h = small_layout.row_rect[3] - small_layout.row_rect[1]
    small_track_w = small_layout.track_rect[2] - small_layout.track_rect[0]

    dialog.render((840, 480))
    large_layout = dialog._layouts["Stop_Direction_Change_Delay_Frames"]
    large_row_h = large_layout.row_rect[3] - large_layout.row_rect[1]
    large_track_w = large_layout.track_rect[2] - large_layout.track_rect[0]

    assert large_row_h > small_row_h
    assert large_track_w > small_track_w
    assert large_layout.track_rect[0] > large_layout.row_rect[0] + 120


def should_update_slider_from_scaled_mouse_position():
    seen_values = []
    dialog = CameraControlDialog("Tracker Controls", on_change=seen_values.append)
    dialog.add_slider("Max_Speed_X_x10", 2000, 0)
    dialog.render((800, 260))

    layout = dialog._layouts["Max_Speed_X_x10"]
    x1, y1, x2, y2 = layout.track_rect
    y = (y1 + y2) // 2
    dialog._on_mouse(cv2.EVENT_LBUTTONDOWN, x2, y, cv2.EVENT_FLAG_LBUTTON, None)

    assert dialog.get_value("Max_Speed_X_x10") == 2000
    assert seen_values[-1] == 2000


def should_render_all_controls_when_dialog_is_short():
    dialog = CameraControlDialog("Tracker Controls")
    names = [
        "Stop_Direction_Change_Delay_Frames",
        "Cancel_Stop_On_Opposite_Direction",
        "Stop_Cancel_Hysteresis_Frames",
        "Stop_Delay_Cooldown_Frames",
        "Overshoot_Stop_Delay_Frames",
        "Post_Nonstop_Stop_Delay_Frames",
        "Overshoot_Speed_Ratio_x100",
        "Time_To_Dest_Speed_Limit_Frames",
        "Apply_To_Fast_Box",
        "Apply_To_Follower_Box",
        "Stitch_Rotate_Degrees",
        "Max_Speed_X_x10",
        "Max_Speed_Y_x10",
        "Max_Accel_X_x10",
        "Max_Accel_Y_x10",
    ]
    for name in names:
        dialog.add_slider(name, 2000, 100)

    for size in ((900, 240), (420, 240), (260, 180)):
        dialog.render(size)

        assert set(dialog._layouts) == set(names)
        assert max(layout.row_rect[3] for layout in dialog._layouts.values()) <= size[1]
        assert all(
            layout.track_rect[2] > layout.track_rect[0] for layout in dialog._layouts.values()
        )


def should_clip_compact_labels_to_available_width():
    label = "Stop Direction Change Delay Frames"
    clipped = CameraControlDialog._clip_text_to_width(label, 40, 0.22)
    width, _height = CameraControlDialog._text_size(clipped, 0.22)

    assert clipped.endswith(".")
    assert width <= 40
