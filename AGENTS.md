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
- Webapp-specific agent notes live in `tools/webapp/AGENTS.md` (Django, deployment, import flow, tests).
- Install via sudo: `python3 tools/webapp/ops/install_webapp.py --watch-root /data/incoming --server-name _ --port 8008` (creates `/opt/hm-webapp/venv`, installs gunicorn/pymysql/django, sets up `hm-webapp.service` and nginx proxy).
- Fast local redeploy: `tools/webapp/ops/redeploy_local.sh` (copies code to `/opt/hm-webapp/app`, restarts services, verifies endpoints).
- End-to-end local import: `./import_webapp.sh` (redeploy + TimeToScore import + shift spreadsheet upload; add `--rebuild` to reset league data first).
- DB setup: installer creates `hm_app_db` and user; ensure MariaDB is running/healthy before install. If prior DB state is corrupt, drop `hm_app_db` then rerun the installer.
- Use the app venv for helpers: `/opt/hm-webapp/venv/bin/python tools/webapp/scripts/seed_demo.py --config /opt/hm-webapp/app/config.json --email demo@example.com --name "Demo User"` (same for reset/import scripts).
- Watch root ownership: installer chowns the watch directory to the app user so uploads work.

## Coding Style & Naming Conventions
- Python: Black (see `pyproject.toml`) and isort; run `./format.sh` before committing.
- Before finalizing changes, run `python -m black` and `ruff check` on any modified/new Python files and fix issues until clean.
- Typing: mypy is configured; prefer typed public APIs in `hmlib/`.
- Naming: modules and functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- C/C++: follow `.clang-format`; standard set to C++17 in CMake.

## Error Handling & CLI Args
- Do not silently ignore failures by default: avoid `except Exception: pass` / bare `except:` / catching-and-returning-success unless there is a clear, documented best-effort reason and the failure is surfaced (log/error return) with context.
- For CLI argument access, do not use `getattr(args, "flag", ...)` to tolerate missing attributes; define all expected args in the parser (with defaults) and access via `args.flag` (use subparsers or separate namespaces if modes differ).

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
