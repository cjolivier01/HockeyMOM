from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
IS_REPO = __name__.startswith("tools.webapp.")


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
    # Django's MySQL backend imports `MySQLdb`. We intentionally use PyMySQL (no mysqlclient build deps),
    # but need to install the shim before Django configures the DB connection.
    try:  # pragma: no cover
        import pymysql  # type: ignore

        pymysql.install_as_MySQLdb()
        version_info = getattr(pymysql, "version_info", (0, 0, 0))
        if tuple(version_info[:3]) < (2, 2, 1):
            pymysql.version_info = (2, 2, 1, "final", 0)
            pymysql.__version__ = "2.2.1"
    except Exception:
        pass
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

def _load_or_create_secret_key() -> str:
    env = os.environ.get("HM_WEBAPP_SECRET")
    if env:
        return str(env)

    # Best-effort: allow config.json to pin the secret for multi-worker deployments.
    for k in ("app_secret", "secret_key", "webapp_secret"):
        v = cfg.get(k)
        if v:
            return str(v)

    # Fall back to a persistent secret under instance/ so all gunicorn workers share it.
    secret_path = INSTANCE_DIR / "app_secret.txt"
    if secret_path.exists():
        s = secret_path.read_text(encoding="utf-8").strip()
        if s:
            return s
    import secrets

    s = secrets.token_hex(32)
    secret_path.write_text(s + "\n", encoding="utf-8")
    try:
        os.chmod(secret_path, 0o600)
    except Exception:
        pass
    return s


SECRET_KEY = _load_or_create_secret_key()
DEBUG = os.environ.get("HM_WEBAPP_DEBUG", "").strip() == "1"
ALLOWED_HOSTS = ["*"]

if __name__.startswith("tools.webapp."):
    INSTALLED_APPS = [
        "django.contrib.sessions",
        "django.contrib.messages",
        "tools.webapp.django_app.apps.HMWebappConfig",
    ]
else:
    INSTALLED_APPS = [
        "django.contrib.sessions",
        "django.contrib.messages",
        "django_app.apps.HMWebappConfig",
    ]

DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

USE_TZ = False
TIME_ZONE = "UTC"

# ----------------------------
# Django "native" webapp bits
# ----------------------------

APPEND_SLASH = False

ROOT_URLCONF = "tools.webapp.urls" if IS_REPO else "urls"
WSGI_APPLICATION = "tools.webapp.wsgi.application" if IS_REPO else "wsgi.application"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    ("tools.webapp.django_app.middleware.LeagueSessionMiddleware" if IS_REPO else "django_app.middleware.LeagueSessionMiddleware"),
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [str(BASE_DIR / "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.template.context_processors.static",
                "django.contrib.messages.context_processors.messages",
                (
                    "tools.webapp.django_app.context_processors.user_leagues"
                    if IS_REPO
                    else "django_app.context_processors.user_leagues"
                ),
            ],
        },
    }
]

STATIC_URL = "/static/"
STATICFILES_DIRS = [str(BASE_DIR / "static")]

# Use DB-backed sessions so Django's test client and middleware behave consistently across requests.
# `django_orm.ensure_schema()` creates the `django_session` table on startup.
SESSION_ENGINE = "django.contrib.sessions.backends.db"
MESSAGE_STORAGE = "django.contrib.messages.storage.cookie.CookieStorage"

# Match legacy 500MB upload limit (nginx config uses the same default).
DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 500
