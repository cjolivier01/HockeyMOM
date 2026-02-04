#!/bin/bash
#
# Exaple yaml script to run an oscillation sweep experiment for HMTrack.
#
# experiment:
#   name: oscillation_sweep_2026_02_04
#   clip:
#     start_time: "00:04:00"
#     duration: "00:00:30"
#   overlay:
#     prefix: "variant: "
#     color: [255, 255, 255]
#     position: [40, 60]
#     max_lines: 6
#   output:
#     mode: tile
#     path: output_workdirs/tv-12-1-r2/experiments/oscillation_sweep_2026_02_04/oscillation_sweep_tile.mkv
#     tile:
#       rows: 2
#       cols: 2
#       padding: 8
#       background: [0, 0, 0]
#   variants:
#     - name: base
#       config: {}
#     - name: hyst20
#       config:
#         rink:
#           camera:
#             resizing_stop_cancel_hysteresis_frames: 20
#     - name: stop_thresh_025
#       config:
#         rink:
#           camera:
#             resizing_time_to_dest_stop_speed_threshold: 0.25
#     - name: cooldown20
#       config:
#         rink:
#           camera:
#             resizing_stop_delay_cooldown_frames: 20
#

PYTHONPATH=$(pwd) python -m hmlib.cli.hmtrack --game-id=tv-12-1-r2 --experiment-config /tmp/hmtrack_osc_experiment.yaml
