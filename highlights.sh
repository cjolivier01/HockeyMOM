#!/bin/bash
./p scripts/parse_stats_inputs.py \
  --season-highlight-types Goal \
  --season-highlight-window-seconds=10 \
  --clip-transition-seconds=2 \
  --file-list $HOME/Videos/jrsharks2013-2025-2026/game_list_long.yaml \
  $@


