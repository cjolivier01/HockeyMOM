#!/usr/bin/env bash
set -euo pipefail

# Default to the git-controlled season file-list when present; MP4s are expected under
# `$HOME/Videos/<game_label>/` (or next to each game's `stats/` directory).
TEAM_SEASON="${TEAM_SEASON:-jrsharks2013-2025-2026}"
DEFAULT_FILE_LIST_1="/mnt/ripper-data/Videos/${TEAM_SEASON}/game_list_long.yaml"
DEFAULT_FILE_LIST_2="$HOME/Videos/${TEAM_SEASON}/game_list_long.yaml"

FILE_LIST="${FILE_LIST:-}"
if [ -z "$FILE_LIST" ]; then
  if [ -f "$DEFAULT_FILE_LIST_1" ]; then
    FILE_LIST="$DEFAULT_FILE_LIST_1"
  elif [ -f "$DEFAULT_FILE_LIST_2" ]; then
    FILE_LIST="$DEFAULT_FILE_LIST_2"
  else
    echo "Error: could not find game list YAML." >&2
    echo "Set FILE_LIST=/path/to/game_list_long.yaml" >&2
    exit 2
  fi
fi

./p scripts/parse_stats_inputs.py \
  --season-highlight-types Goal \
  --season-highlight-window-seconds=15 \
  --clip-transition-seconds=2 \
  --file-list "$FILE_LIST" \
  "$@"

