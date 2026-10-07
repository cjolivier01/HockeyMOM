"""Keep every copy of the control-point matcher default equal to the baseline.

#191 switched the shared baseline to ``akaze-hamming`` but left the CLI, the
resolver fallback and a pipeline assertion on the old value, so the configured
default could not be typed on the command line and the suite was red.

These tests pin the *agreement* rather than any particular matcher, so a future
switch has to update every copy or fail here. They cover: the CLI choices, the
absence of a literal argparse default, the resolver fallback, the library-layer
``DEFAULT_CONTROL_POINT_MATCHER``, the alias map, and the helper signatures
whose non-None defaults would override a configured matcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "hmlib" / "config" / "baseline.yaml"


def _baseline_matcher() -> str:
    with BASELINE.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["stitching"]["control_point_matcher"]


def should_offer_every_known_matcher_on_the_command_line() -> None:
    from hmlib.cli.stitch import make_parser
    from hmlib.hm_opts import hm_opts
    from hmlib.stitching.control_points import CONTROL_POINT_MATCHERS

    parser = hm_opts.parser(parser=make_parser())
    action = next(item for item in parser._actions if item.dest == "control_point_matcher")

    # hm_opts cannot import CONTROL_POINT_MATCHERS (that module pulls in torch),
    # so the list is duplicated there. This is the guard on that duplication.
    assert set(action.choices) == set(CONTROL_POINT_MATCHERS)


def should_accept_the_configured_default_as_a_command_line_value() -> None:
    from hmlib.cli.stitch import make_parser
    from hmlib.hm_opts import hm_opts
    from hmlib.stitching.control_points import normalize_control_point_matcher

    parser = hm_opts.parser(parser=make_parser())
    args = parser.parse_args(["--game-id", "x", "--control-point-matcher", _baseline_matcher()])

    assert normalize_control_point_matcher(args.control_point_matcher) == _baseline_matcher()


def should_resolve_the_baseline_matcher_when_a_config_omits_the_key() -> None:
    from hmlib.stitching.settings import read_stitching_settings

    # A caller passing a bare config must land on the same matcher as one that
    # goes through the baseline, or two runs of the same game disagree about
    # calibration provenance.
    settings = read_stitching_settings({"stitching": {}})

    assert settings.control_point_matcher == _baseline_matcher()


@pytest.mark.parametrize("alias, expected", [("akaze", "akaze-hamming")])
def should_keep_the_matcher_aliases_resolvable(alias, expected) -> None:
    from hmlib.stitching.control_points import normalize_control_point_matcher

    assert normalize_control_point_matcher(alias) == expected


def should_keep_the_library_default_equal_to_the_baseline() -> None:
    from hmlib.stitching.control_points import DEFAULT_CONTROL_POINT_MATCHER

    # Helpers callable without a resolved config default to this constant. If
    # it drifts from the baseline, two entry points calibrate the same game
    # with different matchers and stamp different provenance.
    assert DEFAULT_CONTROL_POINT_MATCHER == _baseline_matcher()


def should_not_carry_a_literal_argparse_default_for_the_matcher() -> None:
    from hmlib.cli.stitch import make_parser
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(parser=make_parser())
    action = next(item for item in parser._actions if item.dest == "control_point_matcher")

    # finalize_parser nulls action.default and sources --help from the
    # baseline, stashing the literal in _yaml_config_original_default. That
    # stash is the only thing the literal still does, and what it does is let
    # sync_args_from_config null a typed value "equal to the default" -- so a
    # literal here silently discards that exact matcher. Assert on the stash,
    # not on action.default, which finalize makes None either way.
    assert (
        getattr(action, "_yaml_config_original_default", None) is None
    ), "the matcher's argparse default must stay None; baseline.yaml owns the value"


def should_not_let_a_helper_default_override_a_configured_matcher() -> None:
    import inspect

    from hmlib.stitching import configure_stitching

    # read_stitching_settings merges overrides with "if value is not None", so
    # a non-None keyword default here beats the game config instead of
    # deferring to it.
    for name in ("_configure_video_stitching_locked", "configure_video_stitching"):
        function = getattr(configure_stitching, name, None)
        if function is None:
            continue
        parameter = inspect.signature(function).parameters.get("control_point_matcher")
        if parameter is None:
            continue
        assert parameter.default is None, f"{name} must default control_point_matcher to None"
