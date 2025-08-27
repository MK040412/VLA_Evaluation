#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(dirname "$0")"
cd "$REPO_ROOT"
OUT_DIR="$REPO_ROOT/quantized_checkpoints_6bit"
RESULT_ROOT="results/prioritized_6bit_$(date +%Y%m%d_%H%M)"
mkdir -p "$RESULT_ROOT/videos" "$RESULT_ROOT/logs"

declare -A ENV2EMB=(
  ["PegInsertionSide-v1"]="lang_embeds/text_embed_PegInsertionSide-v1.pt"
  ["PlugCharger-v1"]="lang_embeds/text_embed_PlugCharger-v1.pt"
  ["PushCube-v1"]="lang_embeds/text_embed_PushCube-v1.pt"
  ["StackCube-v1"]="lang_embeds/text_embed_StackCube-v1.pt"
  ["PickCube-v1"]="lang_embeds/text_embed_PickCube-v1.pt"
)
ENVS=("PegInsertionSide-v1" "PlugCharger-v1" "PushCube-v1" "StackCube-v1" "PickCube-v1")
NUM=25
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

find "$OUT_DIR" -type f -name "*_6bit.pt" | while read CKPT; do
  bname=$(basename "$CKPT" .pt)
  for ENV in "${ENVS[@]}"; do
    LANG="${ENV2EMB[$ENV]}"
    if [[ ! -f "$LANG" ]]; then echo "[SKIP] no lang embed for $ENV -> $LANG"; continue; fi
    VID_DIR="$RESULT_ROOT/videos/${ENV}/${bname}"
    LOG="$RESULT_ROOT/logs/${ENV}__${bname}.log"
    mkdir -p "$VID_DIR" "$(dirname "$LOG")"
    echo "[RUN] $bname on $ENV"
    for QUANT_MODE in "none" "8bit" "4bit"; do
      QUANT_ARGS=""
      if [[ "$QUANT_MODE" != "none" ]]; then
        QUANT_ARGS="--quant $QUANT_MODE --quant-scope all --quant-compute-dtype fp16"
      fi
      LOG="$RESULT_ROOT/logs/${ENV}__${bname}__${QUANT_MODE}.log"
      VID_DIR="$RESULT_ROOT/videos/${ENV}/${bname}/${QUANT_MODE}"
      mkdir -p "$VID_DIR" "$(dirname "$LOG")"
      echo "[RUN] $bname on $ENV with quantization $QUANT_MODE"
      python -m scripts.eval_qunat_rdt_maniskill \
        $QUANT_ARGS \
      --pretrained_path "$CKPT" \
      --env-id "$ENV" \
      --obs-mode rgb \
      --num-traj "$NUM" \
      --render-mode rgb_array \
      --pretrained_text_encoder_name_or_path precomputed \
      --lang_embeddings_path "$LANG" \
      --dtype fp16 \
      --max-steps 400 \
      --action-downsample 1 \
      --save-video-dir "$VID_DIR" \
      --random_seed "$(echo "ibase=16; $(echo "$ENV-$bname" | sha256sum | head -c 10)" | bc)" 2>&1 | tee "$LOG"
  done
done

--random_seed "$(echo "ibase=16; $(echo "$ENV-$bname-$QUANT_MODE" | sha256sum | head -c 10)" | bc)" 2>&1 | tee "$LOG"
    done
done

python tools/summarize_eval_logs.py --logs "$RESULT_ROOT/logs" --out "$RESULT_ROOT/summary.csv"
echo "[OK] summary -> $RESULT_ROOT/summary.csv"