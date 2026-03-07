import pytest

from hmlib.hm_opts import hm_opts


def should_apply_config_override_when_key_exists() -> None:
    cfg = {"stitching": {"enabled": True}}
    hm_opts.apply_config_overrides(cfg, ["stitching.enabled=false"])
    assert cfg["stitching"]["enabled"] is False


def should_error_when_config_override_key_missing() -> None:
    cfg = {"stitching": {"enabled": True}}
    with pytest.raises(KeyError):
        hm_opts.apply_config_overrides(cfg, ["stitching.does_not_exist=true"])
