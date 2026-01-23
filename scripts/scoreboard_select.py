from hmlib.hm_opts import hm_opts
from hmlib.scoreboard.selector import configure_scoreboard, get_max_screen_height

if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    configure_scoreboard(args.game_id, force=True, max_display_height=get_max_screen_height())
