#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

GAME_ID="${1:-}"
CKPT="${2:-}"
SECONDS="${3:-5}"

if [[ -z "${GAME_ID}" || -z "${CKPT}" ]]; then
  echo "Usage: $0 <game-id> <camera_gpt_checkpoint.pt> [seconds]"
  exit 2
fi

OUT_ROOT="$(mktemp -d "/tmp/hm_camgpt_bench_${GAME_ID}.XXXXXX")"
RULE_DIR="${OUT_ROOT}/rule/${GAME_ID}"
GPT_DIR="${OUT_ROOT}/gpt/${GAME_ID}"
mkdir -p "${RULE_DIR}" "${GPT_DIR}"

echo "Output root: ${OUT_ROOT}"

python hmlib/cli/hmtrack.py --game-id "${GAME_ID}" -t "${SECONDS}" --no-audio --no-save-video --deploy-dir "${RULE_DIR}"
python hmlib/cli/hmtrack.py --game-id "${GAME_ID}" -t "${SECONDS}" --no-audio --no-save-video --deploy-dir "${GPT_DIR}" \
  --camera-controller gpt --camera-model "${CKPT}"

VIDEO=""
for candidate in \
  "${HOME}/Videos/${GAME_ID}/${GAME_ID}-stitched_output-with-audio.mp4" \
  "${HOME}/Videos/${GAME_ID}/stitched_output-with-audio.mp4" \
  "${HOME}/Videos/${GAME_ID}/${GAME_ID}-stitched_output.mp4" \
  "${HOME}/Videos/${GAME_ID}/stitched_output.mp4"; do
  if [[ -f "${candidate}" ]]; then
    VIDEO="${candidate}"
    break
  fi
done
if [[ -z "${VIDEO}" ]]; then
  shopt -s nullglob
  candidates=(
    "${HOME}/Videos/${GAME_ID}"/*stitched_output-with-audio*.mp4
    "${HOME}/Videos/${GAME_ID}"/*stitched_output*.mp4
  )
  shopt -u nullglob
  if [[ ${#candidates[@]} -gt 0 ]]; then
    VIDEO="${candidates[0]}"
  fi
fi

COMPARE_ARGS=(--a-dir "${RULE_DIR}" --b-dir "${GPT_DIR}")
if [[ -n "${VIDEO}" ]]; then
  COMPARE_ARGS+=(--video "${VIDEO}" --out-video "${OUT_ROOT}/compare_${GAME_ID}.mp4" --draw-fast)
fi

python -m hmlib.cli.camera_compare "${COMPARE_ARGS[@]}"
