#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import random
import string
import sys
from pathlib import Path


def load_db_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg.get("db", {})


def connect_pymysql(db_cfg: dict):
    import pymysql

    return pymysql.connect(
        host=db_cfg.get("host", "127.0.0.1"),
        port=int(db_cfg.get("port", 3306)),
        user=db_cfg.get("user", "hmapp"),
        password=db_cfg.get("pass", ""),
        database=db_cfg.get("name", "hm_app_db"),
        autocommit=False,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )


def ensure_user(conn, email: str, name: str, password_hash: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        now = dt.datetime.now().isoformat()
        cur.execute(
            "INSERT INTO users(email, password_hash, name, created_at) VALUES(%s,%s,%s,%s)",
            (email, password_hash, name, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def ensure_defaults(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM game_types")
        count = (cur.fetchone() or [0])[0]
        if int(count) == 0:
            for name in ("Preseason", "Regular Season", "Tournament", "Exhibition"):
                cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 1))
        conn.commit()


def create_team(conn, user_id: int, name: str, is_external: int = 0) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
            (user_id, name, is_external, dt.datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)


def create_player(conn, user_id: int, team_id: int, name: str, jersey: str, position: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO players(user_id, team_id, name, jersey_number, position, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
            (user_id, team_id, name, jersey, position, dt.datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)


def create_game(
    conn,
    user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_name: str,
    starts_at: dt.datetime,
    location: str,
    score: tuple | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM game_types WHERE name=%s", (game_type_name,))
        gt = cur.fetchone()
        gt_id = int(gt[0]) if gt else None
        cur.execute(
            "INSERT INTO hky_games(user_id, team1_id, team2_id, game_type_id, starts_at, location, team1_score, team2_score, is_final, created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                user_id,
                team1_id,
                team2_id,
                gt_id,
                starts_at.strftime("%Y-%m-%d %H:%M:%S"),
                location,
                score[0] if score else None,
                score[1] if score else None,
                1 if score else 0,
                dt.datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def add_random_stats(conn, game_id: int, team_id: int, user_id: int):
    import random as _r

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM players WHERE team_id=%s ORDER BY id", (team_id,))
        pids = [int(r[0]) for r in cur.fetchall()]
    with conn.cursor() as cur:
        for pid in pids:
            goals = _r.randint(0, 2)
            assists = _r.randint(0, 2)
            shots = goals + assists + _r.randint(0, 3)
            pim = _r.choice([0, 0, 2, 4])
            plus_minus = _r.randint(-1, 2)
            cur.execute(
                """
                INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists, shots, pim, plus_minus)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE goals=VALUES(goals), assists=VALUES(assists), shots=VALUES(shots), pim=VALUES(pim), plus_minus=VALUES(plus_minus)
                """,
                (user_id, team_id, game_id, pid, goals, assists, shots, pim, plus_minus),
            )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Seed demo data for HockeyMOM WebApp hockey features")
    ap.add_argument(
        "--config",
        default=os.environ.get("HM_DB_CONFIG")
        or str((Path(__file__).resolve().parent / "config.json")),
    )
    ap.add_argument("--email", default="demo@example.com")
    ap.add_argument("--name", default="Demo User")
    ap.add_argument(
        "--password-hash",
        default="pbkdf2:sha256:260000$demo$Yy6lWp5oSz5Ahh3yI9sRhW/9k5D5mZ0t8Xr6Z3YYc2U=",
    )
    ap.add_argument("--teams", nargs="*", default=["Thunderbirds 12U", "Falcons 12U"])
    args = ap.parse_args()

    db_cfg = load_db_cfg(args.config)
    try:
        conn = connect_pymysql(db_cfg)
    except Exception:
        print(
            "Failed to connect to DB. Ensure the webapp was installed and DB configured.",
            file=sys.stderr,
        )
        raise

    ensure_defaults(conn)
    user_id = ensure_user(conn, args.email, args.name, args.password_hash)

    # Teams
    team_ids = []
    for nm in args.teams:
        team_ids.append(create_team(conn, user_id, nm, is_external=0))
    ext_id = create_team(conn, user_id, "Ice Wolves", is_external=1)

    # Players
    def gen_name():
        return "Player " + "".join(random.choice(string.ascii_uppercase) for _ in range(3))

    positions = ["F", "D", "G"]
    num = 1
    for tid in team_ids:
        for i in range(10):
            create_player(conn, user_id, tid, gen_name(), str(num), random.choice(positions))
            num += 1

    # Games
    now = dt.datetime.now()
    g1 = create_game(
        conn,
        user_id,
        team_ids[0],
        team_ids[1],
        "Regular Season",
        now - dt.timedelta(days=1),
        "Main Rink",
        score=(3, 2),
    )
    create_game(
        conn,
        user_id,
        team_ids[0],
        ext_id,
        "Exhibition",
        now + dt.timedelta(days=2),
        "Community Ice",
        score=None,
    )

    # Stats for completed game
    add_random_stats(conn, g1, team_ids[0], user_id)
    add_random_stats(conn, g1, team_ids[1], user_id)

    print("Seeded demo data.")


if __name__ == "__main__":
    main()
