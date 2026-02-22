import argparse


def should_show_scaled_imply_show_image_and_config() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--show-scaled", "0.5"])

    # Mimic hmtrack ordering: apply arg->config overrides before hm_opts.init.
    game_config = {"aspen": {"video_out": {"show_image": False, "show_scaled": None}}}
    hm_opts.apply_arg_config_overrides(game_config, args)
    args.game_config = game_config
    args = hm_opts.init(args, parser)

    assert args.show_image is True
    assert args.game_config["aspen"]["video_out"]["show_image"] is True


def should_map_stitch_frame_time_arg_into_config() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--stitch-frame-time", "00:00:07"])

    game_config = {"aspen": {"stitching": {"stitch_frame_time": "00:00:00"}}}
    hm_opts.apply_arg_config_overrides(game_config, args)

    assert game_config["aspen"]["stitching"]["stitch_frame_time"] == "00:00:07"


def should_read_stitch_frame_time_from_loaded_config_when_cli_missing() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args([])
    args.game_config = {"aspen": {"stitching": {"stitch_frame_time": "00:00:42"}}}

    args = hm_opts.init(args, parser)

    assert args.stitch_frame_time == "00:00:42"


def should_not_read_private_stitch_frame_time_when_ignore_private_config(monkeypatch) -> None:
    import hmlib.hm_opts as hm_opts_module
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--game-id", "test-game", "--ignore-private-config", "1"])
    args.game_config = {}

    monkeypatch.setattr(
        hm_opts_module,
        "get_game_config_private",
        lambda game_id: {"aspen": {"stitching": {"stitch_frame_time": "00:00:33"}}},
    )

    args = hm_opts.init(args, parser)

    assert args.stitch_frame_time is None
