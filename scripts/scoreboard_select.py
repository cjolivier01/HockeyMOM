from hmlib.hm_opts import hm_opts
from hmlib.scoreboard.selector import configure_scoreboard, get_screen_size

if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    configure_scoreboard(args.game_id, force=True, max_display_size=get_screen_size())
