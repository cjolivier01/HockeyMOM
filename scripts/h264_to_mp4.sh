#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./h265_to_mp4_fastcopy.sh in.h265 out.mp4
#   ./h265_to_mp4_fastcopy.sh in.h265 out.mp4 --audio other.mkv
#   ./h265_to_mp4_fastcopy.sh in.h265 out.mp4 --audio other.mp4 --audio-offset 0.250
#   ./h265_to_mp4_fastcopy.sh in.h265 out.mp4 --audio other.mkv --audio-stream 0

IN="$1"
OUT="$2"
shift 2

AUDIO_FILE=""
AUDIO_STREAM="0"
AUDIO_OFFSET="0"
FPS="30000/1001"
AAC_BITRATE="192k"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio) AUDIO_FILE="$2"; shift 2;;
    --audio-stream) AUDIO_STREAM="$2"; shift 2;;
    --audio-offset) AUDIO_OFFSET="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg not found"
  exit 1
fi

# ------------------------------------------------------------
# Build video input (raw h265 needs fps + pts)
# ------------------------------------------------------------
VIDEO_INPUT=(-fflags +genpts -r "$FPS" -i "$IN")

MP4_FLAGS=(-tag:v hvc1 -movflags +faststart)

# ------------------------------------------------------------
# No audio
# ------------------------------------------------------------
if [[ -z "$AUDIO_FILE" ]]; then
  ffmpeg -hide_banner -y \
    "${VIDEO_INPUT[@]}" \
    -map 0:v:0 \
    -c:v copy \
    "${MP4_FLAGS[@]}" \
    -an \
    "$OUT"
  exit 0
fi

# ------------------------------------------------------------
# Detect audio codec
# ------------------------------------------------------------
AUDIO_CODEC=$(ffprobe -v error \
  -select_streams "a:${AUDIO_STREAM}" \
  -show_entries stream=codec_name \
  -of default=nk=1:nw=1 \
  "$AUDIO_FILE" || true)

if [[ -z "$AUDIO_CODEC" ]]; then
  echo "Could not detect audio codec"
  exit 1
fi

echo "Detected audio codec: $AUDIO_CODEC"

# ------------------------------------------------------------
# Choose audio mode
# ------------------------------------------------------------
if [[ "$AUDIO_CODEC" == "aac" ]]; then
  echo "AAC detected → stream copy audio"
  AUDIO_CODEC_ARGS=(-c:a copy)
else
  echo "Non-AAC detected → re-encode audio to AAC"
  AUDIO_CODEC_ARGS=(-c:a aac -b:a "$AAC_BITRATE" -ac 2 -ar 48000)
fi

# Apply offset if needed
AUDIO_INPUT=()
if [[ "$AUDIO_OFFSET" != "0" && "$AUDIO_OFFSET" != "0.0" ]]; then
  AUDIO_INPUT+=(-itsoffset "$AUDIO_OFFSET")
fi
AUDIO_INPUT+=(-i "$AUDIO_FILE")

# ------------------------------------------------------------
# Final mux
# ------------------------------------------------------------
ffmpeg -hide_banner -y \
  "${VIDEO_INPUT[@]}" \
  "${AUDIO_INPUT[@]}" \
  -map 0:v:0 \
  -map "1:a:${AUDIO_STREAM}" \
  -c:v copy \
  "${AUDIO_CODEC_ARGS[@]}" \
  "${MP4_FLAGS[@]}" \
  -shortest \
  "$OUT"
