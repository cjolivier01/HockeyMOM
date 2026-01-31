# `scripts/` Agent Guidelines

This file applies to the `scripts/` subtree.

## Formatting and lint

- Before finalizing changes, run `python -m black` and `ruff check` on any modified/new Python files and fix issues until clean.

## Error handling & CLI args

- Do not silently ignore failures by default: avoid `except Exception: pass` / bare `except:` unless there is a clear, documented best-effort reason and the failure is surfaced (log/error return) with context.
- For CLI argument access, do not use `getattr(args, "flag", ...)` to tolerate missing attributes; define all expected args in the parser (with defaults) and access via `args.flag` (use subparsers or separate namespaces if modes differ).
