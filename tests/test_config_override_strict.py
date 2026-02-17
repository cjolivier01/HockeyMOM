import pytest

from hmlib.hm_opts import hm_opts


def should_apply_config_override_when_key_exists() -> None:
    cfg = {"aspen": {"stitching": {"enabled": True}}}
    hm_opts.apply_config_overrides(cfg, ["aspen.stitching.enabled=false"])
    assert cfg["aspen"]["stitching"]["enabled"] is False


def should_error_when_config_override_key_missing() -> None:
    cfg = {"aspen": {"stitching": {"enabled": True}}}
    with pytest.raises(KeyError):
        hm_opts.apply_config_overrides(cfg, ["aspen.stitching.does_not_exist=true"])
