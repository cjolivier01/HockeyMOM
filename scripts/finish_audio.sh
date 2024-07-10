#!/bin/bash
set -e

# This function parses command line arguments
parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --game-id)
        GAME_ID="$2"
        shift 2
        ;;
      --audio)
        AUDIO_FILE="$2"
        shift 2
        ;;
      *)
        echo "Error: Unknown option $1"
        echo "Usage: $0 --game-id <game_id>"
        exit 1
        ;;
    esac
  done
}

GAME_ID=""

# Function to check if the game ID was set
check_game_id() {
  if [ -z "${GAME_ID}" ]; then
    echo "Error: Game ID is required."
    echo "Usage: $0 --game-id <game_id> [--audio <audio_file>]"
    exit 1
  fi
}

# Parse command line arguments
parse_args "$@"

# Validate that GAME_ID is provided
check_game_id

INPUT_VIDEO_WITH_AUDIO=""

if [ ! -z "${AUDIO_FILE}" ]; then
  INPUT_VIDEO_WITH_AUDIO="${AUDIO_FILE}"
  if [ ! -f "${INPUT_VIDEO_WITH_AUDIO}" ]; then
    INPUT_VIDEO_WITH_AUDIO="${HOME}/Videos/${GAME_ID}/${AUDIO_FILE}"
    if [ ! -f "${INPUT_VIDEO_WITH_AUDIO}" ]; then
      echo "Could not find audio file ${AUDIO_FILE} or ${INPUT_VIDEO_WITH_AUDIO}"
      exit 1
    fi
  fi
fi

if [ -z "${INPUT_VIDEO_WITH_AUDIO}" ]; then
  INPUT_VIDEO_WITH_AUDIO="${HOME}/Videos/${GAME_ID}/stitched_output-with-audio.mkv"
  if [ ! -f "${INPUT_VIDEO_WITH_AUDIO}" ]; then
    INPUT_VIDEO_WITH_AUDIO="${HOME}/Videos/${GAME_ID}/stitched_output-with-audio.mp4"
  fi
fi

INPUT_VIDEO_NO_AUDIO="./output_workdirs/${GAME_ID}/tracking_output.mkv"
OUTPUT_VIDEO_WITH_AUDIO="${HOME}/Videos/${GAME_ID}/tracking_output-with-audio.mkv"
ffmpeg \
  -i "${INPUT_VIDEO_WITH_AUDIO}" \
  -i "${INPUT_VIDEO_NO_AUDIO}" \
  -c:v copy -c:a copy \
  -map 1:v:0 \
  -map 0:a:0 \
  -shortest \
  "${OUTPUT_VIDEO_WITH_AUDIO}"
