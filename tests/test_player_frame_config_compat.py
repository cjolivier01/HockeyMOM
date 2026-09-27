"""Keep HStream's saved frame-selection fingerprint intact on Python saves."""

import yaml

from hmlib import config as hmlib_config
from hmlib.stitching.configure_stitching import _read_private_config_snapshot

_LEGACY_CONFIG = """\
stitching:
  calibration_frame_selection:
    fingerprint: abcdef0123456789
    context:
      output_rotation_degrees: 0.000000
      source_context: !!str null
other_number: 3.5
"""


def should_preserve_legacy_player_frame_context_through_private_config_save(tmp_path, monkeypatch):
    game = tmp_path / "sample"
    game.mkdir()
    config_path = game / "config.yaml"
    config_path.write_text(_LEGACY_CONFIG, encoding="utf-8")
    monkeypatch.setitem(
        hmlib_config.get_game_config_private.__globals__, "GAME_DIR_BASE", str(tmp_path)
    )

    snapshot, _ = _read_private_config_snapshot(config_path)
    assert snapshot["stitching"]["calibration_frame_selection"]["context"] == {
        "output_rotation_degrees": "0.000000",
        "source_context": "null",
    }

    loaded = hmlib_config.get_game_config_private("sample")
    assert loaded["stitching"]["calibration_frame_selection"]["context"] == {
        "output_rotation_degrees": "0.000000",
        "source_context": "null",
    }
    assert loaded["other_number"] == 3.5
    loaded["new_setting"] = True
    hmlib_config.save_private_config("sample", loaded, verbose=False)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["stitching"]["calibration_frame_selection"]["context"] == {
        "output_rotation_degrees": "0.000000",
        "source_context": "null",
    }
    assert saved["new_setting"] is True
    assert saved["other_number"] == 3.5
    snapshot, _ = _read_private_config_snapshot(config_path)
    assert (
        snapshot["stitching"]["calibration_frame_selection"]["context"]
        == saved["stitching"]["calibration_frame_selection"]["context"]
    )
