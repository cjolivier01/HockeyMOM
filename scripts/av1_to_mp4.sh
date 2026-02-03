#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<EOF
Usage:
  $0 <input.av1|input.ivf> <output.mp4> [options]

Options:
  --audio <file>         Audio source (mp4, mkv, etc.)
  --audio-stream <N>     Audio stream index in audio file (default: 0)
  --audio-offset <sec>   Audio offset in seconds (e.g. 0.250 or -0.100)
  --fps <rate>           Input video frame rate (default: 30000/1001)

Examples:
  $0 in.av1 out.mp4
  $0 in.ivf out.mp4 --audio audio.mkv
  $0 in.av1 out.mp4 --audio audio.mp4 --audio-offset 0.25
  $0 in.ivf out.mp4 --audio audio.mkv --audio-stream 1
EOF
}

# ------------------------------------------------------------
# Require at least input + output
# ------------------------------------------------------------
if [[ $# -lt 2 ]]; then
  print_usage
  exit 1
fi

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
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

if ! command -v ffmpeg >/dev/null; then
  echo "Error: ffmpeg not found in PATH"
  exit 1
fi

# ------------------------------------------------------------
# Video input (raw AV1/IVF → needs fps + pts if missing)
# ------------------------------------------------------------
VIDEO_INPUT=(-fflags +genpts -r "$FPS" -i "$IN")
MP4_FLAGS=(-tag:v av01 -movflags +faststart -bsf:v av1_frame_merge)

# ------------------------------------------------------------
# Video only
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
  echo "Error: could not detect audio stream ${AUDIO_STREAM} in $AUDIO_FILE"
  exit 1
fi

echo "Detected audio codec: $AUDIO_CODEC"

# ------------------------------------------------------------
# Choose audio handling
# ------------------------------------------------------------
if [[ "$AUDIO_CODEC" == "aac" ]]; then
  echo "Audio is AAC → stream copy"
  AUDIO_CODEC_ARGS=(-c:a copy)
else
  echo "Audio is $AUDIO_CODEC → re-encode to AAC"
  AUDIO_CODEC_ARGS=(-c:a aac -b:a "$AAC_BITRATE" -ac 2 -ar 48000)
fi

# Apply optional offset
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
