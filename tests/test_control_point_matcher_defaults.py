"""The control-point matcher default is spelled in four places; keep them equal.

HockeyMONStream #237 switched the shared baseline to ``akaze-hamming`` but left
the CLI, the resolver fallback and a pipeline assertion on the old value, so
the configured default could not be typed on the command line and the suite was
red. These tests pin the agreement rather than any particular matcher, so a
future switch has to update every copy or fail here.
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
