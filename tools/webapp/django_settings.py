from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


cfg_path = os.environ.get("HM_DB_CONFIG")
if cfg_path:
    cfg = _load_config(cfg_path)
else:
    default_cfg = BASE_DIR / "config.json"
    cfg = _load_config(default_cfg)

dbcfg = dict(cfg.get("db") or {})

db_engine = str(dbcfg.get("engine") or "").strip().lower()
use_sqlite = db_engine in {"sqlite", "sqlite3"}

if use_sqlite:
    db_name = str(dbcfg.get("name") or (INSTANCE_DIR / "hm_webapp.sqlite3"))
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": db_name}}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": str(dbcfg.get("name") or "hm_app_db"),
            "USER": str(dbcfg.get("user") or "hmapp"),
            "PASSWORD": str(dbcfg.get("pass") or ""),
            "HOST": str(dbcfg.get("host") or "127.0.0.1"),
            "PORT": str(dbcfg.get("port") or 3306),
            "OPTIONS": {"charset": "utf8mb4"},
        }
    }

SECRET_KEY = os.environ.get("HM_WEBAPP_SECRET", "hm-webapp-dev-secret")
DEBUG = False

if __name__.startswith("tools.webapp."):
    INSTALLED_APPS = [
        "tools.webapp.django_app.apps.HMWebappConfig",
    ]
else:
    INSTALLED_APPS = [
        "django_app.apps.HMWebappConfig",
    ]

DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

USE_TZ = False
TIME_ZONE = "UTC"
