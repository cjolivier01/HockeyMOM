#!/usr/bin/env bash

# The following are additional stamps injected into bazel builds.

branch="nogit"
sha="nogit"
serial="0"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr '/' '-' || true)"
  sha="$(git rev-parse --short HEAD 2>/dev/null || true)"
  serial="$(git show -s --format=%ct HEAD 2>/dev/null || true)"
fi

echo "STABLE_GIT_BRANCH ${branch}"
echo "GIT_SHA ${sha}"
echo "GIT_SERIAL_NUMBER ${serial}"
