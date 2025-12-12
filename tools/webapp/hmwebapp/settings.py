import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


SECRET_KEY = os.environ.get("HM_WEBAPP_SECRET") or "hm-webapp-dev-secret-change-me"

DEBUG = _bool_env("HM_WEBAPP_DEBUG", default=False)

ALLOWED_HOSTS: list[str] = os.environ.get("HM_WEBAPP_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "hmwebapp.webapp",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "hmwebapp.webapp.middleware.LeagueSelectionMiddleware",
]

ROOT_URLCONF = "hmwebapp.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [ROOT_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "hmwebapp.webapp.context_processors.user_leagues",
            ],
        },
    },
]

WSGI_APPLICATION = "hmwebapp.wsgi.application"


def _load_db_from_config() -> dict | None:
    cfg_path = os.environ.get("HM_DB_CONFIG")
    if not cfg_path:
        return None
    path = Path(cfg_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    dbcfg = cfg.get("db", {})
    if not dbcfg:
        return None
    return {
        "ENGINE": "django.db.backends.mysql",
        "NAME": dbcfg.get("name", "hm_app_db"),
        "USER": dbcfg.get("user", "hmapp"),
        "PASSWORD": dbcfg.get("pass", ""),
        "HOST": dbcfg.get("host", "127.0.0.1"),
        "PORT": dbcfg.get("port", 3306),
        "OPTIONS": {"charset": "utf8mb4"},
    }


_db_from_cfg = _load_db_from_config()
if _db_from_cfg:
    DATABASES = {"default": _db_from_cfg}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LANGUAGE_CODE = "en-us"

TIME_ZONE = os.environ.get("HM_WEBAPP_TZ", "UTC")

USE_I18N = True

USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = ROOT_DIR / "staticfiles"
STATICFILES_DIRS = [ROOT_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

WATCH_ROOT = os.environ.get("HM_WATCH_ROOT", "/data/incoming")
HM_WEBAPP_EMAIL_FROM = os.environ.get("HM_FROM_EMAIL", "")
