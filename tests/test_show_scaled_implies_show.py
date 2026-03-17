import argparse


def should_show_scaled_imply_show_image_and_config() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--show-scaled", "0.5"])

    # Mimic hmtrack ordering: apply arg->config overrides before hm_opts.init.
    game_config = {"video_out": {"show_image": False, "show_scaled": None}}
    hm_opts.apply_arg_config_overrides(game_config, args, parser=parser)
    args.game_config = game_config
    args = hm_opts.init(args, parser)

    assert args.show_image is True
    assert args.game_config["video_out"]["show_image"] is True


def should_map_stitch_frame_time_arg_into_config() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--stitch-frame-time", "00:00:07"])

    game_config = {"stitching": {"stitch_frame_time": "00:00:00"}}
    hm_opts.apply_arg_config_overrides(game_config, args, parser=parser)

    assert game_config["stitching"]["stitch_frame_time"] == "00:00:07"


def should_read_stitch_frame_time_from_loaded_config_when_cli_missing() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args([])
    args.game_config = {"stitching": {"stitch_frame_time": "00:00:42"}}

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
        lambda game_id: {"stitching": {"stitch_frame_time": "00:00:33"}},
    )

    args = hm_opts.init(args, parser)

    assert args.stitch_frame_time is None


def should_map_minimize_blend_zero_into_config_and_resolved_plugin_param() -> None:
    from hmlib.config import resolve_global_refs
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--minimize-blend", "0"])

    game_config = {
        "stitching": {"minimize_blend": True},
        "aspen": {
            "plugins": {
                "stitching": {
                    "class": "x.y.Z",
                    "depends": [],
                    "params": {"minimize_blend": "GLOBAL.stitching.minimize_blend"},
                }
            }
        },
    }
    hm_opts.apply_arg_config_overrides(game_config, args, parser=parser)
    resolve_global_refs(game_config)

    assert game_config["stitching"]["minimize_blend"] is False
    assert game_config["aspen"]["plugins"]["stitching"]["params"]["minimize_blend"] is False


def should_not_apply_parser_defaults_as_config_overrides() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args([])

    game_config = {"stitching": {"auto_adjust_exposure": True, "blend_mode": "multiblend"}}
    hm_opts.apply_arg_config_overrides(game_config, args, parser=parser)

    assert game_config["stitching"]["auto_adjust_exposure"] is True
    assert game_config["stitching"]["blend_mode"] == "multiblend"


def should_persist_explicit_yaml_backed_cli_args_to_private_config(monkeypatch) -> None:
    import hmlib.hm_opts as hm_opts_module
    from hmlib.hm_opts import hm_opts

    argv = [
        "--game-id",
        "test-game",
        "--output",
        "cli-output.mp4",
        "--show-scaled",
        "0.5",
        "--camera-window",
        "12",
    ]
    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    args.explicit_arg_names = hm_opts.collect_explicit_arg_names(parser, argv)

    args.game_config = {
        "video_out": {
            "output_video_path": "yaml-output.mp4",
            "show_image": False,
            "show_scaled": None,
        },
        "rink": {"camera": {"camera_window": 8}},
    }

    hm_opts.apply_arg_config_overrides(
        args.game_config,
        args,
        parser=parser,
        explicit_arg_names=args.explicit_arg_names,
    )
    args = hm_opts.init(args, parser)

    saved = {}
    monkeypatch.setattr(hm_opts_module, "get_game_config_private", lambda game_id: {})
    monkeypatch.setattr(
        hm_opts_module,
        "save_private_config",
        lambda game_id, data, verbose=True: saved.update(data),
    )

    changed = hm_opts.persist_private_config_overrides(
        args,
        parser=parser,
        config=args.game_config,
        explicit_arg_names=args.explicit_arg_names,
        verbose=False,
    )

    assert changed is True
    assert saved["video_out"]["output_video_path"] == "cli-output.mp4"
    assert saved["video_out"]["show_scaled"] == 0.5
    assert saved["video_out"]["show_image"] is True
    assert saved["rink"]["camera"]["camera_window"] == 12


def should_persist_config_override_to_private_config(monkeypatch) -> None:
    import hmlib.hm_opts as hm_opts_module
    from hmlib.hm_opts import hm_opts

    argv = [
        "--game-id",
        "test-game",
        "--config-override",
        "stitching.max_blend_levels=7",
    ]
    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    args.explicit_arg_names = hm_opts.collect_explicit_arg_names(parser, argv)
    args.game_config = {"stitching": {"max_blend_levels": 4}}

    args = hm_opts.init(args, parser)

    saved = {}
    monkeypatch.setattr(hm_opts_module, "get_game_config_private", lambda game_id: {})
    monkeypatch.setattr(
        hm_opts_module,
        "save_private_config",
        lambda game_id, data, verbose=True: saved.update(data),
    )

    changed = hm_opts.persist_private_config_overrides(
        args,
        parser=parser,
        config=args.game_config,
        explicit_arg_names=args.explicit_arg_names,
        verbose=False,
    )

    assert changed is True
    assert saved["stitching"]["max_blend_levels"] == 7


def should_not_write_private_config_when_ignored(monkeypatch) -> None:
    import hmlib.hm_opts as hm_opts_module
    from hmlib.hm_opts import hm_opts

    argv = [
        "--game-id",
        "test-game",
        "--ignore-private-config",
        "1",
        "--blend-mode",
        "gpu-hard-seam",
    ]
    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    args.explicit_arg_names = hm_opts.collect_explicit_arg_names(parser, argv)
    args.game_config = {"stitching": {"blend_mode": "laplacian"}}

    args = hm_opts.init(args, parser)

    calls = {"count": 0}
    monkeypatch.setattr(hm_opts_module, "get_game_config_private", lambda game_id: {})
    monkeypatch.setattr(
        hm_opts_module,
        "save_private_config",
        lambda game_id, data, verbose=True: calls.__setitem__("count", calls["count"] + 1),
    )

    changed = hm_opts.persist_private_config_overrides(
        args,
        parser=parser,
        config=args.game_config,
        explicit_arg_names=args.explicit_arg_names,
        verbose=False,
    )

    assert changed is False
    assert calls["count"] == 0
