# Repository Guidelines

## Project Structure & Module Organization
- `hmlib/`: Primary Python library and CLI entry points (e.g., `hmlib.cli.hmtrack`). Packaged via Bazel wheel rules.
- `src/`: Ancillary modules (`core/`, `users/`) used by higher-level code.
- `tests/`: Bazel `py_test` targets and simple runtime checks.
- `tools/`, `scripts/`: Bazel helpers, formatting utilities, and dev scripts.
- `assets/`, `external/`, `openmm/`, `cmake/`: Assets and native/C++ build integration.

## Build, Test, and Development Commands
- Build all: `bazelisk build //...` (or `bazel build //...`).
- Run tests: `bazelisk test //...`.
- Coverage report: `./coverage.sh` (generates HTML in `reports/coverage/`).
- Apply formatters: `./format.sh` (Black/isort via Bazel aspect).
- Run a Bazel target: `./run.sh //path:target --args`.
- Package wheel: `./run.sh //hmlib:bdist_wheel`.
- Run tracker locally (example): `EXP_NAME=dev VIDEO=video.mp4 ./hm_run.sh`.
 - Exclude wheels: `bazelisk build --config=no-python-wheels //...` (or `./bld --no-python-wheels`).

### Handy local debugging command

- Run TensorRT-enabled tracking on a short clip (5s) for `chicago-3`:
  - `PYTHONPATH=$(pwd) python hmlib/cli/hmtrack.py --game-id=chicago-3 --async-post-processing=0 --async-video-out=0 --show-scaled=0.5 --camera-ui=1 --detector-trt-enable --detector-static-detections --detector-static-max-detections=800 --plot-tracking -t=5`

## Webapp Deployment Notes
- Install via sudo: `python3 tools/webapp/install_webapp.py --watch-root /data/incoming --server-name _ --port 8008` (creates `/opt/hm-webapp/venv`, installs gunicorn/django/pymysql, sets up `hm-webapp.service` and nginx proxy).
- DB setup: installer creates `hm_app_db` and user; ensure MariaDB is running/healthy before install. If prior DB state is corrupt, drop `hm_app_db` then rerun the installer.
- Use the app venv for helpers: `/opt/hm-webapp/venv/bin/python tools/webapp/seed_demo.py --config /opt/hm-webapp/app/config.json --email demo@example.com --name "Demo User"` (same for reset/import scripts).
- Watch root ownership: installer chowns the watch directory to the app user so uploads work.

### Webapp (Django) Implementation Notes
- Code lives under `tools/webapp/`:
  - Django project: `tools/webapp/hmwebapp` (settings, urls, WSGI).
  - App: `hmwebapp.webapp` (models are `managed = False` ORM mappings onto the existing MySQL schema).
  - WSGI entrypoint for gunicorn: `tools/webapp/app.py` (`app`/`application` is the Django WSGI callable).
- System install layout (created by `install_webapp.py`):
  - App root: `/opt/hm-webapp/app` (contains `app.py`, `manage.py`, `hmwebapp/`, `templates/`, `static/`, `config.json`).
  - Venv: `/opt/hm-webapp/venv` (Django, gunicorn, mysqlclient).
  - Service: `hm-webapp.service` (uses `DJANGO_SETTINGS_MODULE=hmwebapp.settings` and `ExecStart=… python -m gunicorn -b 127.0.0.1:8008 app:app`).
  - Config: `/opt/hm-webapp/app/config.json` (read via `HM_DB_CONFIG`).
- Environment variables:
  - `DJANGO_SETTINGS_MODULE=hmwebapp.settings` (set in dev via `tools/webapp/run_dev.sh` and in systemd).
  - `HM_DB_CONFIG=/opt/hm-webapp/app/config.json` to point Django at the MySQL DB; otherwise it falls back to SQLite for local dev.
  - `HM_WATCH_ROOT=/data/incoming` (or custom) controls where game directories and `_READY` sentinels are created.
  - `HM_WEBAPP_SKIP_DB_INIT=1` disables the best-effort `db_init.init_db()` call from `WebAppConfig.ready` (important for tests that should not touch the DB).
- DB and migrations:
  - Do **not** add Django migrations that try to alter the existing `users/games/teams/...` tables; they are managed by `tools/webapp/hmwebapp/webapp/db_init.py` and external scripts.
  - Django’s own apps (`auth`, `admin`, `sessions`, `contenttypes`) **do** use migrations; run `python manage.py migrate` only after `HM_DB_CONFIG` is correctly pointing at the target DB.
  - Schema initializer: `python manage.py init_hm_db` calls `db_init.init_db()` and is the canonical way to create/upgrade the webapp tables.
- Testing and smoke checks:
  - Unit-style tests for the webapp live in `tests/test_webapp_*.py` and are designed to run with:
    - `PYTHONPATH=tools/webapp python -m pytest tests/test_webapp_*.py`
    - They expect `HM_WEBAPP_SKIP_DB_INIT=1` for URL/config-only tests, and may rely on Django but not on a live DB.
  - HTTP health view: `/healthz` (implemented in `hmwebapp.webapp.views.healthcheck`) should stay simple (`SELECT 1` against the default DB, returning `{"status": "...", "db": bool}`) so it’s safe for liveness probes.
  - End-to-end UI smoke: `tools/webapp/smoke_ui.sh [BASE_URL]` (defaults to `http://127.0.0.1`) drives registration, team creation, schedule, and stats using curl with CSRF-aware headers. Keep this script aligned with template/URL changes.
- CSRF and templates:
  - All HTML forms under `tools/webapp/templates/` must include `{% csrf_token %}` for POSTs.
  - When porting Jinja templates, use Django’s `{% empty %}` clause instead of `for … {% else %}` patterns.
  - The curl-based smoke tests rely on a `csrftoken` cookie and send `X-CSRFToken` headers; if you rename the cookie or change CSRF settings, update `tools/webapp/smoke_ui.sh` accordingly.
- DirWatcher and filesystem interaction:
  - Game creation and job submission write under `WATCH_ROOT`/`HM_WATCH_ROOT` and use `_READY` files plus `.dirwatch_meta.json` to signal DirWatcher; avoid changing these conventions without updating DirWatcher.
  - Views that read or delete files (uploads, game deletion) guard against path traversal by checking `Path.resolve().startswith(WATCH_ROOT)`. Preserve this pattern when modifying file IO.


## Coding Style & Naming Conventions
- Python: Black (line length 120) and isort; run `./format.sh` before committing.
- Typing: mypy is configured; prefer typed public APIs in `hmlib/`.
- Naming: modules and functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- C/C++: follow `.clang-format`; standard set to C++17 in CMake.

## Testing Guidelines
- Framework: pytest conventions are supported; prefer test functions named `should_*` (see `pyproject.toml`).
- Location: add tests under `tests/` or as Bazel `py_test` targets alongside code.
- Run: `bazelisk test //...` for CI-equivalent runs; use `./coverage.sh` to inspect branch coverage locally.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., `fix(hmlib): handle empty frames`). Prefix with scope when helpful (`hmlib/`, `tools/`, `build/`).
- PRs: include a clear description, linked issues, what/why, and test evidence (logs, sample outputs). Update docs when behavior changes.
- Assets: do not commit large datasets or model weights; use `datasets/` and `pretrained/` symlinks.

## Security & Configuration Tips
- Secrets: never commit credentials; prefer environment variables.
- Large files: keep outside the repo (symlinks `datasets/`, `pretrained/`).
- Reproducibility: run via Bazel for consistent tooling; avoid ad‑hoc local installs unless developing isolated modules.

## AspenNet Architecture
- Graph runner built from YAML `aspen.trunks` mapping (`class`, `depends`, `params`, optional `enabled`); missing deps or cycles raise; disabled trunks become no-op stubs to preserve graph shape; graph is exported to `aspennet.dot` on init.
- Execution modes set under `aspen.pipeline`/`threaded_trunks`: sequential topological order by default, or threaded pipeline with one worker per trunk connected by bounded `Queue(queue_size)`; optional per-trunk CUDA streams (`cuda_streams`) wrap each trunk and synchronize before handoff; grad/no-grad follows the `training` flag.
- Context flow: `forward` threads a shared mutable `context` (injects `shared` and `trunks` namespaces); trunks can declare `input_keys`/`output_keys` and, when `minimal_context` is true, only requested keys plus `shared` are passed; outputs update context, `DeleteKey` removes entries, and each trunk's outputs are stored under `context["trunks"][name]`.
- Device selection for stream usage is inferred from `context`/`shared` (`device`, `cuda_stream`, tensor devices) with CUDA current-device fallback; profiling is plumbed through `shared["profiler"]` using trunk `profile_scope`.
- Shutdown: `finalize()` is invoked on trunks if present; DAG is available via `to_networkx`/`to_dot` helpers and `display_graphviz`.
