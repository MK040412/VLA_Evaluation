#!/usr/bin/env bash
set -euo pipefail

# === 결과 루트 ===
OUT_BASE="results/ablation_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_BASE/videos" "$OUT_BASE/logs"

# === 공통 설정 ===
NUM=25
MAX_STEPS=400
ADOWN=1
FPS=20

# 평가 순서: 먼저 fp32(원본), 이후 양자화 가중치들
RUN_TAGS=("fp32" "w8" "w6" "w5" "w4" "w3" "w2")
# 각 tag -> 체크포인트 경로 매핑 템플릿(printf로 채움)
#  - fp32: 원본
#  - wX  : ablate.sh에서 만든 파일 네이밍과 일치
declare -A TAG2CKPT
TAG2CKPT["fp32"]="pretrained_models/rdt/mp_rank_00_model_states.pt"
TAG2CKPT["w8"]="pretrained_models/rdt/ablate_w8_uniform_pc_sym.pt"
TAG2CKPT["w6"]="pretrained_models/rdt/ablate_w6_uniform_pc_sym.pt"
TAG2CKPT["w5"]="pretrained_models/rdt/ablate_w5_uniform_pc_sym.pt"
TAG2CKPT["w4"]="pretrained_models/rdt/ablate_w4_uniform_pc_sym.pt"
TAG2CKPT["w3"]="pretrained_models/rdt/ablate_w3_uniform_pc_sym.pt"
TAG2CKPT["w2"]="pretrained_models/rdt/ablate_w2_uniform_pc_sym.pt"

# === 환경 ↔ 사전계산 임베딩 ===
declare -A ENV2EMB=(
  ["PegInsertionSide-v1"]="lang_embeds/text_embed_PegInsertionSide-v1.pt"
  ["PlugCharger-v1"]="lang_embeds/text_embed_PlugCharger-v1.pt"
  ["PushCube-v1"]="lang_embeds/text_embed_PushCube-v1.pt"
  ["StackCube-v1"]="lang_embeds/text_embed_StackCube-v1.pt"
  ["PickCube-v1"]="lang_embeds/text_embed_PickCube-v1.pt"  # 마지막에 수행
)

# 실행 순서(마지막에 PickCube-v1)
ENV_ORDER=("PegInsertionSide-v1" "PlugCharger-v1" "PushCube-v1" "StackCube-v1" "PickCube-v1")

# (선택) PYTHONPATH 보정
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

echo "[INFO] Results root: $OUT_BASE"
echo "[INFO] PYTHONPATH  : $PYTHONPATH"
echo

for ENV in "${ENV_ORDER[@]}"; do
  LANG="${ENV2EMB[$ENV]:-}"
  if [[ -z "$LANG" || ! -f "$LANG" ]]; then
    echo "[SKIP] $ENV : lang embedding 없음 -> $LANG"
    continue
  fi

  for TAG in "${RUN_TAGS[@]}"; do
    CKPT="${TAG2CKPT[$TAG]}"
    if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
      echo "[SKIP] $ENV | $TAG : 체크포인트 없음 -> $CKPT"
      continue
    fi

    VID_DIR="$OUT_BASE/videos/${ENV}/${TAG}"
    LOG_PATH="$OUT_BASE/logs/${ENV}_${TAG}.log"
    mkdir -p "$VID_DIR" "$(dirname "$LOG_PATH")"

    echo "=== Run ${ENV} | ${TAG} ==="
    echo "[CKPT] $CKPT"
    echo "[LANG] $LANG"
    echo "[VID ] $VID_DIR"
    echo "[LOG ] $LOG_PATH"

    python -m scripts.eval_qunat_rdt_maniskill \
      --pretrained_path "$CKPT" \
      --env-id "$ENV" \
      --obs-mode rgb \
      --num-traj "$NUM" \
      --render-mode rgb_array \
      --pretrained_text_encoder_name_or_path precomputed \
      --lang_embeddings_path "$LANG" \
      --dtype fp16 \
      --max-steps "$MAX_STEPS" \
      --action-downsample "$ADOWN" \
      --save-video-dir "$VID_DIR" \
      --video-fps "$FPS" 2>&1 | tee "$LOG_PATH"

    echo
  done
done

# 로그 요약 CSV
python tools/summarize_eval_logs.py \
  --logs "$OUT_BASE/logs" \
  --out "$OUT_BASE/summary.csv"

echo "[OK] Summary CSV -> $OUT_BASE/summary.csv"
echo "Videos saved under -> $OUT_BASE/videos/"
