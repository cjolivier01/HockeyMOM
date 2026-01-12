#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

GAME_ID="${1:-}"
SECONDS="${2:-5}"
STEPS="${3:-400}"

if [[ -z "${GAME_ID}" ]]; then
  echo "Usage: $0 <game-id> [seconds] [train_steps]"
  exit 2
fi

OUT_ROOT="$(mktemp -d "/tmp/hm_camgpt_e2e_${GAME_ID}.XXXXXX")"
RULE_DIR="${OUT_ROOT}/rule/${GAME_ID}"
GPT_DIR="${OUT_ROOT}/gpt/${GAME_ID}"
mkdir -p "${RULE_DIR}" "${GPT_DIR}"

echo "Output root: ${OUT_ROOT}"

python hmlib/cli/hmtrack.py --game-id "${GAME_ID}" -t "${SECONDS}" --no-audio --no-save-video --deploy-dir "${RULE_DIR}"

CKPT="${OUT_ROOT}/camera_gpt_${GAME_ID}.pt"
python -m hmlib.cli.camgpt_train \
  --videos-root "${OUT_ROOT}/rule" \
  --game-id "${GAME_ID}" \
  --seq-len 32 \
  --batch-size 32 \
  --steps "${STEPS}" \
  --val-split 0.0 \
  --no-pose \
  --feature-mode base_prev_y \
  --scheduled-sampling \
  --ss-prob-end 0.5 \
  --ss-warmup-steps 200 \
  --out "${CKPT}"

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

COMPARE_ARGS=(--a-dir "${RULE_DIR}" --b-dir "${GPT_DIR}" --draw-fast)
if [[ -n "${VIDEO}" ]]; then
  COMPARE_ARGS+=(--video "${VIDEO}" --out-video "${OUT_ROOT}/overlay_${GAME_ID}.mp4")
fi

python -m hmlib.cli.camera_compare "${COMPARE_ARGS[@]}"

echo "Checkpoint: ${CKPT}"
