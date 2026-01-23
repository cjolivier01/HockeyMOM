# `tools/webapp` Agent Guidelines

This subtree contains the HockeyMOM webapp plus its admin/import scripts. The webapp is now a **native Django app**
(Django views + DTL templates) and uses the **Django ORM** to access the database; do not re-introduce raw SQL access.

## High-level layout

- `tools/webapp/manage.py`: Django management entrypoint (devserver/admin tasks).
- `tools/webapp/hm_webapp/`: Django project package (`settings.py`, `urls.py`, `wsgi.py`).
- `tools/webapp/app.py`: Shared hockey/league webapp logic used by the Django runtime.
- `tools/webapp/django_app/views.py`: Django views (MVT) and REST endpoints.
- `tools/webapp/templates/`, `tools/webapp/static/`: UI templates/assets.
- `tools/webapp/hm_webapp/settings.py`: Minimal Django settings module (DB config + installed apps).
- `tools/webapp/django_settings.py`, `tools/webapp/urls.py`, `tools/webapp/wsgi.py`: Back-compat shims for older installs/tests.
- `tools/webapp/django_orm.py`: Django setup + schema/bootstrap helpers (no migrations).
- `tools/webapp/django_app/`: Django app containing `models.py` mapping the existing DB schema.
- Admin/import utilities:
  - `tools/webapp/scripts/import_time2score.py`: TimeToScore scraper/importer (direct DB or REST).
  - `tools/webapp/scripts/reset_league_data.py`, `tools/webapp/scripts/wipe_league.py`, `tools/webapp/scripts/delete_league.py`,
    `tools/webapp/scripts/dedupe_league_teams.py`, `tools/webapp/scripts/recalc_div_ratings.py`, etc.
- Deployment helpers:
  - `tools/webapp/ops/install_webapp.py`: system install (creates `/opt/hm-webapp/app` + venv + systemd + nginx).
  - `tools/webapp/ops/redeploy_local.sh`: copy code to `/opt/hm-webapp/app` + restart services.
  - `tools/webapp/ops/deploy_gcp.py`, `tools/webapp/ops/redeploy_gcp.py`: GCP helpers.

Related root-level helpers:
- `./import_webapp.sh`: redeploy + import + shift-spreadsheet upload (add `--rebuild` to reset league data first).
- `./gcp_import_webapp.sh`: similar flow for GCP.
  - Both scripts will best-effort create a default webapp user from local `git config` (`user.email`/`user.name`)
    with password `password` unless `--no-default-user` is specified (uses internal REST endpoint; may require token).
  - Both scripts support `--t2s-league` (import a subset of TimeToScore league ids) and `--rebuild` (reset league data first).
  - Shift spreadsheet file lists (e.g., `~/RVideos/game_list_long.yaml`) should use readable YAML mapping entries (no legacy one-line `|key=value` strings); see `tools/webapp/README.md`.
  - Event correction YAML for `scripts/parse_stats_inputs.py --corrections-yaml` should be structured YAML objects (no `|` separators); see `tools/webapp/README.md`.

## Architecture: Django + Django ORM (no raw SQL)

### Key rule: avoid raw SQL
- Do not add `cursor.execute(...)`, `pymysql.connect(...)`, or string SQL back into `tools/webapp`.
- Use Django ORM/querysets for reads/writes and `transaction.atomic()` for multi-step updates.
- If you need an aggregate/query that is awkward in ORM, prefer queryset annotations/subqueries first; only fall back to
  database-specific SQL as a last resort (and keep it isolated/minimal).

### How Django is embedded
- `tools/webapp/django_orm.setup_django()` configures Django in-process and is safe to call multiple times.
- `tools/webapp/app.py` still contains shared business logic and lazily loads ORM modules via `_orm_modules()` (cached),
  so importing it does not require a working DB in tests.
- Deployment uses Django WSGI via `wsgi:application` (see `tools/webapp/hm_webapp/wsgi.py`).
- Schema management is done programmatically:
  - `django_orm.ensure_schema()` creates missing tables and best-effort adds missing columns for older installs.
  - There is no Django migration workflow here; keep schema changes small and additive unless you also handle upgrades.

### MySQL/MariaDB backend notes
- Django’s MySQL backend expects a `MySQLdb` module and validates a minimum `mysqlclient` version.
- The webapp uses `pymysql.install_as_MySQLdb()` and may shim version metadata in `tools/webapp/django_orm.py` so Django
  will accept older PyMySQL installs.
- If the deployed service fails with `ImproperlyConfigured: mysqlclient ... required`, ensure:
  - the updated `django_orm.py` is deployed to `/opt/hm-webapp/app`, and
  - `hm-webapp` has been restarted (`sudo systemctl restart hm-webapp`).

## Configuration

### DB config (`config.json`)
Most scripts accept `--config` (default: `/opt/hm-webapp/app/config.json`). The DB stanza is used by
`tools/webapp/hm_webapp/settings.py`:

```json
{
  "db": {
    "engine": "mysql",
    "name": "hm_app_db",
    "user": "hmapp",
    "pass": "…",
    "host": "127.0.0.1",
    "port": 3306
  },
  "import_token": "optional"
}
```

SQLite is supported for tests/dev (`engine: sqlite3`, with `name` pointing at a `.sqlite3` path).

### Environment variables
- `HM_DB_CONFIG`: path to the active `config.json` (used by Django settings and many scripts).
- `HM_WEBAPP_SECRET`: Django secret (dev/test value is fine for local use).
- `HM_WEBAPP_IMPORT_TOKEN`: if set, internal REST import/reset endpoints require this token.
- `HM_WEBAPP_SKIP_DB_INIT`: tests set this to avoid eager DB init at import time.
- `HM_WATCH_ROOT`: upload/watch directory (affects Jobs/DirWatcher integration).

## Local deploy + import workflow

Recommended end-to-end local workflow:
- `./import_webapp.sh` (redeploy, import TimeToScore via REST, upload shift spreadsheets; add `--rebuild` to reset league data first).

Quick “code-only” update:
- `tools/webapp/ops/redeploy_local.sh` (copies code to `/opt/hm-webapp/app` and restarts `hm-webapp` + `nginx`).

Smoke UI validation (nginx front by default):
- `bash tools/webapp/ops/smoke_ui.sh` (registers a user, creates a team/player/game, imports a tiny CSV, verifies record).

## Testing

### Where tests live
- Webapp tests are in `tests/test_webapp_*.py`.
- Run: `pytest -q tests/test_webapp_*.py`.

### Golden regression fixtures (do not update without permission)
- The repo includes a snapshot-based regression dataset for the Jr Sharks 2013 2025–2026 season:
  - Tests: `tests/test_webapp_regression_jrsharks2013_2025_2026.py`
  - Fixtures/expected output: `tests/data/webapp_regression/jrsharks2013_2025_2026/`
- Policy: if the regression snapshots fail, fix the webapp code to restore the prior UI/stat output. Do **not**
  change the fixture or expected snapshot data unless the user explicitly authorizes updating the golden data.
- When the golden test fails: capture/present the diff (the test prints a unified diff), explain what changed and why,
  and ask the user before changing any golden files so recent intentional UI changes aren’t accidentally overwritten.

### Django settings are process-global
Django cannot be “reconfigured” once initialized in a Python process. To avoid flakes:
- Prefer using the shared fixtures in `tests/conftest.py` (`webapp_db`, `webapp_db_reset`) which configure Django to use a
  sqlite DB.
- Tests that call ORM-backed helpers should take `webapp_db` (or `webapp_orm_modules`) as a fixture dependency so Django
  is initialized correctly before importing/using webapp modules.

## Common troubleshooting

- `500 Internal Server Error` in browser:
  - Check logs: `sudo journalctl -u hm-webapp -n 200 --no-pager`.
  - If the error references Django configuration, confirm `/opt/hm-webapp/venv` has Django installed.
- Service health:
  - `sudo systemctl status hm-webapp nginx mariadb --no-pager`
  - `curl -I http://127.0.0.1:8008/teams` (gunicorn) and `curl -I http://127.0.0.1/teams` (nginx proxy)
- If an old install is missing Django:
  - `/opt/hm-webapp/venv/bin/python -m pip install django`

## Code style and safety

- Keep changes scoped: webapp code changes should stay within `tools/webapp/` + related tests/docs.
- Prefer small, explicit ORM queries (`select_related`, `values`, `values_list`) to avoid N+1 performance issues.
- Indentation: use spaces (not tabs) in all source files (Python/C/C++/JS/HTML/CSS/etc). Tabs are only allowed where they have special meaning (e.g., Makefiles).
- When touching import/reset scripts, be careful with deletion order and FK constraints; use `transaction.atomic()` for
  multi-table operations and keep plans idempotent when possible.
- Before finalizing changes, run `python -m black` and `ruff check` on any modified/new Python files and fix issues until clean.
- Do not silently ignore failures by default: avoid `except Exception: pass` / bare `except:` unless there is a clear, documented best-effort reason and the failure is surfaced (log/error return) with context.
- For CLI argument access, do not use `getattr(args, "flag", ...)` to tolerate missing attributes; define all expected args in the parser (with defaults) and access via `args.flag` (use subparsers or separate namespaces if modes differ).
