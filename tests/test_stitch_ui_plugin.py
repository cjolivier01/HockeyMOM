from __future__ import annotations

import copy

from hmlib.aspen.plugins import stitch_ui_plugin as stitch_ui_module
from hmlib.aspen.plugins.stitch_ui_plugin import StitchUiPlugin


def _color() -> dict:
    return {
        "white_balance": [1.0, 1.0, 1.0],
        "brightness": 1.0,
        "exposure_ev": 0.0,
        "contrast": 1.0,
        "gamma": 1.0,
    }


def _config(rotation: float) -> dict:
    return {
        "stitching": {
            "post_stitch_rotate_degrees": rotation,
            "left": {"color": _color()},
            "right": {"color": _color()},
        },
        "rink": {"camera": {"color": _color()}},
    }


class _FakeHmUiProcess:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.values = {}
        self.system_defaults = {}
        self.changed = False
        self.actions = []
        self.previews = []
        self.closed = False
        self.instances.append(self)

    def add_window(self, name: str) -> None:
        self.values.setdefault(name, {})

    def add_slider(self, window: str, name: str, _maximum: int, value: int) -> None:
        self.values[window][name] = value

    def set_system_defaults(self, defaults) -> None:
        self.system_defaults = copy.deepcopy(defaults)

    def get_value(self, window: str, name: str) -> int:
        return self.values[window][name]

    def poll(self) -> bool:
        changed, self.changed = self.changed, False
        return changed

    def consume_actions(self):
        actions, self.actions = self.actions, []
        return actions

    def publish_preview(self, img, *, name: str) -> None:
        self.previews.append((img, name))

    def close(self) -> None:
        self.closed = True


def should_apply_and_save_stitch_only_rust_controls(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    current_config = _config(rotation=5.0)
    system_config = _config(rotation=0.0)
    private_config = {"rink": {"camera": {"color": {"contrast": 1.0}}}}
    saved = {}

    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", _FakeHmUiProcess)
    monkeypatch.setattr(
        stitch_ui_module,
        "get_config",
        lambda **_kwargs: copy.deepcopy(system_config),
    )
    monkeypatch.setattr(
        stitch_ui_module,
        "get_game_config_private",
        lambda **_kwargs: copy.deepcopy(private_config),
    )
    monkeypatch.setattr(
        stitch_ui_module,
        "save_private_config",
        lambda _game_id, data, verbose=True: saved.update(copy.deepcopy(data)),
    )

    shared = {
        "camera_ui": 1,
        "game_id": "game-1",
        "game_config": current_config,
    }
    plugin = StitchUiPlugin()
    image = object()
    plugin.forward({"img": image, "shared": shared})

    process = _FakeHmUiProcess.instances[0]
    assert process.kwargs["preview_names"] == ("Stitched",)
    assert shared["hm_ui_process"] is process
    assert process.previews[-1] == (image, "Stitched")

    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 125
    process.changed = True
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert current_config["rink"]["camera"]["color"]["brightness"] == 1.25

    process.actions = ["save"]
    plugin.forward({"img": image, "shared": shared})

    assert saved["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert saved["rink"]["camera"]["color"] == {"brightness": 1.25}

    plugin.finalize()
    assert process.closed is True
    assert shared["hm_ui_process"] is None
