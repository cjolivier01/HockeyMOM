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
