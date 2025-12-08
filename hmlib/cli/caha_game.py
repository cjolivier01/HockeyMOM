import argparse
import json
import os
import sys
from typing import Optional

# Import API with fallback to path-local time2score modules.
try:  # pragma: no cover
    from hmlib.time2score import api  # type: ignore
except Exception:  # noqa: BLE001
    this_dir = os.path.dirname(__file__)
    t2s_path = os.path.normpath(os.path.join(this_dir, "..", "time2score"))
    if t2s_path not in sys.path:
        sys.path.insert(0, t2s_path)
    import api  # type: ignore  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fetch a CAHA game by id with teams and stats")
    p.add_argument("--game-id", type=int, required=True, help="Game id (e.g., 46300)")
    p.add_argument(
        "--season", type=int, default=0, help="Season id to prefer when syncing (0=auto)"
    )
    p.add_argument(
        "--no-sync-if-missing", action="store_true", help="Do not sync if game not found in DB"
    )
    p.add_argument(
        "--no-fetch-stats-if-missing",
        action="store_true",
        help="Do not scrape oss-scoresheet if stats missing",
    )
    p.add_argument("--out", type=str, default="-", help="Write JSON to file or '-' for stdout")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = p.parse_args(argv)

    season = None if args.season == 0 else args.season
    res = api.get_game_details(
        args.game_id,
        season=season,
        sync_if_missing=not args.no_sync_if_missing,
        fetch_stats_if_missing=not args.no_fetch_stats_if_missing,
    )

    if args.pretty:
        payload = json.dumps(res, indent=2, sort_keys=True)
    else:
        payload = json.dumps(res, separators=(",", ":"))

    if args.out == "-":
        print(payload)
    else:
        with open(args.out, "w") as f:
            f.write(payload)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
