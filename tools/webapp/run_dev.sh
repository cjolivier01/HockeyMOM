#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

export DJANGO_SETTINGS_MODULE=hmwebapp.settings

python3 manage.py migrate --noinput || true
python3 manage.py runserver 127.0.0.1:8008

