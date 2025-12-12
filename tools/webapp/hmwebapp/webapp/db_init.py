from __future__ import annotations

import json
import os
from pathlib import Path


def _get_db_config_path() -> str:
    base_dir = Path(__file__).resolve().parents[2]
    default_cfg = base_dir / "config.json"
    return os.environ.get("HM_DB_CONFIG", str(default_cfg))


def get_db():
    cfg_path = _get_db_config_path()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    dbcfg = cfg.get("db", {})

    import pymysql  # type: ignore[import]

    conn = pymysql.connect(
        host=dbcfg.get("host", "127.0.0.1"),
        port=int(dbcfg.get("port", 3306)),
        user=dbcfg.get("user", "hmapp"),
        password=dbcfg.get("pass", ""),
        database=dbcfg.get("name", "hm_app_db"),
        autocommit=False,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )
    return conn


def init_db() -> None:
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INT AUTO_INCREMENT PRIMARY KEY,
              email VARCHAR(255) UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              name VARCHAR(255),
              created_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS leagues (
              id INT AUTO_INCREMENT PRIMARY KEY,
              name VARCHAR(255) UNIQUE NOT NULL,
              owner_user_id INT NOT NULL,
              is_shared TINYINT(1) NOT NULL DEFAULT 0,
              source VARCHAR(64) NULL,
              external_key VARCHAR(255) NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(owner_user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_members (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              user_id INT NOT NULL,
              role VARCHAR(32) NOT NULL DEFAULT 'viewer',
              created_at DATETIME NOT NULL,
              UNIQUE KEY uniq_member (league_id, user_id),
              INDEX(league_id), INDEX(user_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              name VARCHAR(255) NOT NULL,
              dir_path TEXT NOT NULL,
              status VARCHAR(32) NOT NULL DEFAULT 'new',
              created_at DATETIME NOT NULL,
              INDEX(user_id),
              FOREIGN KEY(user_id) REFERENCES users(id)
                ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              game_id INT,
              dir_path TEXT NOT NULL,
              slurm_job_id VARCHAR(64),
              status VARCHAR(32) NOT NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              finished_at DATETIME NULL,
              user_email VARCHAR(255) NULL,
              INDEX(user_id), INDEX(game_id), INDEX(slurm_job_id), INDEX(status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resets (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              token VARCHAR(128) UNIQUE NOT NULL,
              expires_at DATETIME NOT NULL,
              used_at DATETIME NULL,
              created_at DATETIME NOT NULL,
              INDEX(user_id), INDEX(token), INDEX(expires_at),
              FOREIGN KEY(user_id) REFERENCES users(id)
                ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS teams (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              name VARCHAR(255) NOT NULL,
              logo_path TEXT NULL,
              is_external TINYINT(1) NOT NULL DEFAULT 0,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(user_id), INDEX(is_external),
              UNIQUE KEY uniq_team_user_name (user_id, name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        try:
            cur.execute("SHOW COLUMNS FROM users LIKE 'default_league_id'")
            exists = cur.fetchone()
            if not exists:
                cur.execute("ALTER TABLE users ADD COLUMN default_league_id INT NULL")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_users_default_league ON users(default_league_id)"
                )
        except Exception:
            try:
                cur.execute("ALTER TABLE users ADD COLUMN default_league_id INT NULL")
            except Exception:
                pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS players (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              team_id INT NOT NULL,
              name VARCHAR(255) NOT NULL,
              jersey_number VARCHAR(16) NULL,
              position VARCHAR(32) NULL,
              shoots VARCHAR(8) NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(user_id), INDEX(team_id), INDEX(name),
              FOREIGN KEY(team_id) REFERENCES teams(id)
                ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS game_types (
              id INT AUTO_INCREMENT PRIMARY KEY,
              name VARCHAR(64) UNIQUE NOT NULL,
              is_default TINYINT(1) NOT NULL DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hky_games (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              team1_id INT NOT NULL,
              team2_id INT NOT NULL,
              game_type_id INT NULL,
              starts_at DATETIME NULL,
              location VARCHAR(255) NULL,
              notes TEXT NULL,
              team1_score INT NULL,
              team2_score INT NULL,
              is_final TINYINT(1) NOT NULL DEFAULT 0,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(user_id), INDEX(team1_id), INDEX(team2_id), INDEX(game_type_id), INDEX(starts_at),
              FOREIGN KEY(team1_id) REFERENCES teams(id) ON DELETE RESTRICT ON UPDATE CASCADE,
              FOREIGN KEY(team2_id) REFERENCES teams(id) ON DELETE RESTRICT ON UPDATE CASCADE,
              FOREIGN KEY(game_type_id) REFERENCES game_types(id) ON DELETE SET NULL ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS player_stats (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              team_id INT NOT NULL,
              game_id INT NOT NULL,
              player_id INT NOT NULL,
              goals INT NULL,
              assists INT NULL,
              shots INT NULL,
              pim INT NULL,
              plus_minus INT NULL,
              hits INT NULL,
              blocks INT NULL,
              toi_seconds INT NULL,
              faceoff_wins INT NULL,
              faceoff_attempts INT NULL,
              goalie_saves INT NULL,
              goalie_ga INT NULL,
              goalie_sa INT NULL,
              UNIQUE KEY uniq_game_player (game_id, player_id),
              INDEX(user_id), INDEX(team_id), INDEX(game_id), INDEX(player_id),
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(player_id) REFERENCES players(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_teams (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              team_id INT NOT NULL,
              UNIQUE KEY uniq_league_team (league_id, team_id),
              INDEX(league_id), INDEX(team_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_games (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              game_id INT NOT NULL,
              UNIQUE KEY uniq_league_game (league_id, game_id),
              INDEX(league_id), INDEX(game_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
    db.commit()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM game_types")
        count = (cur.fetchone() or [0])[0]
        if int(count) == 0:
            for name in ("Preseason", "Regular Season", "Tournament", "Exhibition"):
                cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 1))
    db.commit()

