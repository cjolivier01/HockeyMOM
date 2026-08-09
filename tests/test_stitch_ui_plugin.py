from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="requires torch")

if torch is not None:
    from hmlib.aspen.plugins import stitch_ui_plugin as stitch_ui_module
    from hmlib.aspen.plugins.stitch_ui_plugin import StitchUiPlugin
else:
    stitch_ui_module = None  # type: ignore[assignment]
    StitchUiPlugin = None  # type: ignore[assignment,misc]


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
        self.open_defaults = {}
        self.system_defaults = {}
        self.changed = False
        self.last_poll_values_changed = False
        self.actions = []
        self.previews = []
        self.closed = False
        self.instances.append(self)

    def add_window(self, name: str) -> None:
        self.values.setdefault(name, {})
        self.open_defaults.setdefault(name, {})

    def add_slider(self, window: str, name: str, _maximum: int, value: int) -> None:
        self.values[window][name] = value
        self.open_defaults[window][name] = value

    def set_system_defaults(self, defaults) -> None:
        self.system_defaults = copy.deepcopy(defaults)

    def get_value(self, window: str, name: str, *, poll: bool = True) -> int:
        del poll
        return self.values[window][name]

    def poll(self) -> bool:
        changed, self.changed = self.changed, False
        self.last_poll_values_changed = changed
        return changed or bool(self.actions)

    def consume_action_events(self, *, poll: bool = True):
        del poll
        actions, self.actions = self.actions, []
        return actions

    def control_values(self):
        return copy.deepcopy(self.values)

    def apply_control_values(self, values) -> bool:
        changed = values != self.values
        for window, controls in values.items():
            self.values.setdefault(window, {}).update(copy.deepcopy(controls))
        return changed

    def queue_action(self, kind: str) -> None:
        self.actions.append(_FakeAction(kind=kind, values=self.control_values()))

    def queue_reset(self, *, system: bool) -> None:
        self.values = copy.deepcopy(self.system_defaults if system else self.open_defaults)
        self.changed = True
        self.queue_action("reset-system" if system else "reset-open")

    def publish_preview(self, img, *, name: str) -> None:
        self.previews.append((img, name))

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _FakeAction:
    kind: str
    values: dict


def should_apply_and_save_stitch_only_rust_controls(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    current_config = _config(rotation=5.0)
    system_config = _config(rotation=0.5)
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

    def save_private(_game_id, data, verbose=True):
        del verbose
        saved.clear()
        saved.update(copy.deepcopy(data))

    monkeypatch.setattr(stitch_ui_module, "save_private_config", save_private)

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

    process.queue_reset(system=True)
    process.queue_action("save")
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 0.5
    assert current_config["rink"]["camera"]["color"]["brightness"] == 1.0
    assert "post_stitch_rotate_degrees" not in saved.get("stitching", {})

    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 125
    process.changed = True
    plugin.forward({"img": image, "shared": shared})

    process.queue_action("save")
    plugin.forward({"img": image, "shared": shared})

    assert saved["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert saved["rink"]["camera"]["color"] == {"brightness": 1.25}

    # Save must use its own snapshot even if a later reset is in the same poll.
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 75
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 150
    process.queue_action("save")
    process.queue_reset(system=True)
    plugin.forward({"img": image, "shared": shared})

    assert saved["stitching"]["post_stitch_rotate_degrees"] == 15.0
    assert saved["rink"]["camera"]["color"] == {"brightness": 1.5}
    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 0.5

    # An edit after reset must be applied before a following Save snapshot.
    process.queue_reset(system=True)
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 125
    process.queue_action("save")
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert saved["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert saved["rink"]["camera"]["color"] == {"brightness": 1.25}

    # Multiple resets retain click order and restore the exact source value.
    process.queue_reset(system=True)
    process.queue_reset(system=False)
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 5.0

    plugin.finalize()
    assert process.closed is True
    assert shared["hm_ui_process"] is None


def should_propagate_stitch_ui_initialization_failure(monkeypatch):
    class FailingHmUiProcess(_FakeHmUiProcess):
        def add_window(self, name: str) -> None:
            del name
            raise OSError("cannot start hm-ui")

    FailingHmUiProcess.instances.clear()
    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", FailingHmUiProcess)
    plugin = StitchUiPlugin()

    with pytest.raises(RuntimeError, match="Failed to initialize stitch camera UI") as exc_info:
        plugin.forward(
            {
                "img": object(),
                "shared": {
                    "camera_ui": 1,
                    "game_config": _config(rotation=0.0),
                },
            }
        )

    assert str(exc_info.value.__cause__) == "cannot start hm-ui"
    assert FailingHmUiProcess.instances[0].closed is True
