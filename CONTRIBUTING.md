# Contributing

Thanks for your interest in improving this project! Start here for the essentials; consult `AGENTS.md` for full repository guidelines.

## Getting Started
- Prereqs: Python 3.10, Bazel/Bazelisk installed. Optional: `lcov` for coverage HTML.
- Clone submodules (if needed): `git submodule update --init --recursive`.

## Quick Commands
- Build all: `bazelisk build //...`
- Run tests: `bazelisk test //...`
- Coverage report: `./coverage.sh` (outputs to `reports/coverage/`).
- Auto-format: `./format.sh`
- Package wheel: `./run.sh //hmlib:bdist_wheel`
- Example run: `EXP_NAME=dev VIDEO=video.mp4 ./hm_run.sh`

## Workflow
- Create a feature branch from `main`.
- Make focused commits with clear, imperative subjects (e.g., `fix(hmlib): handle empty frames`).
- Keep changes small and scoped; update docs and tests alongside code.

## Before You Open a PR
- Run `./format.sh` and ensure formatting is clean.
- Run `bazelisk test //...` and fix any failures.
- If behavior changes, update `AGENTS.md` or relevant READMEs.
- Provide a descriptive PR: what/why, linked issues, and test evidence (logs, screenshots, or sample outputs).

## More Info
- Detailed structure, commands, style, and testing conventions live in `AGENTS.md`.
