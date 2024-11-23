from hmlib.hm_opts import hm_opts
from hmlib.scoreboard.selector import configure_scoreboard

if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    configure_scoreboard(args.game_id, force=True)
