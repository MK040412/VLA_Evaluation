#!/bin/bash

# Create a timestamped directory for results
RESULT_ROOT="results/ablation_$(date +%Y%m%d_%H%M)"
mkdir -p "$RESULT_ROOT/videos" "$RESULT_ROOT/logs"

# Remove previously generated quantized checkpoints to ensure regeneration
rm -rf ./quantized_checkpoints_6bit

# Base command for evaluation
BASE_EVAL_CMD="python -m scripts.eval_qunat_rdt_maniskill --pretrained_path /home/perelman/RoboticsDiffusionTransformer/pretrained_models/rdt/mp_rank_00_model_states.pt --num-traj 25 --only-count-success --video-fps 20"

TASKS=(
    "PickCube-v1"
    "PegInsertionSide-v1"
    "PlugCharger-v1"
    "StackCube-v1"
    "PushCube-v1"
)

for TASK in "${TASKS[@]}"; do
    echo "================================================================="
    echo "Running evaluations for task: $TASK"
    echo "================================================================="
    
    # Baseline: No quantization
    echo "Running Baseline: No Quantization"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__baseline.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/baseline"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant none --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 4-bit quantization ablation
    QUANT_MODE="4bit"
    COMPUTE_DTYPE="fp16"

    # Priority Candidates (12 as per user's request)
    # 1. embeddings_and_adaptors
    echo "Running 4-bit Quantization: embeddings_and_adaptors"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__embeddings_and_adaptors.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/embeddings_and_adaptors"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "embed" "adaptor" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 2. attn_qkv_block_X (per-block qkv) - Example: block 0
    echo "Running 4-bit Quantization: attn_qkv_block_0"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__attn_qkv_block_0.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/attn_qkv_block_0"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.blocks.0.attn.qkv" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 3. attn_proj_block_X (output projection) - Example: block 0
    echo "Running 4-bit Quantization: attn_proj_block_0"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__attn_proj_block_0.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/attn_proj_block_0"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.blocks.0.attn.proj" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 4. ffn_fc1_block_X - Example: block 0
    echo "Running 4-bit Quantization: ffn_fc1_block_0"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__ffn_fc1_block_0.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/ffn_fc1_block_0"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.blocks.0.ffn.fc1" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 5. ffn_fc2_block_X - Example: block 0
    echo "Running 4-bit Quantization: ffn_fc2_block_0"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__ffn_fc2_block_0.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/ffn_fc2_block_0"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.blocks.0.ffn.fc2" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 6. cross_q_all (all cross attention query projections)
    echo "Running 4-bit Quantization: cross_q_all"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__cross_q_all.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/cross_q_all"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "cross_attn.q" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 7. cross_kv_all (all cross attention key/value projections)
    echo "Running 4-bit Quantization: cross_kv_all"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__cross_kv_all.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/cross_kv_all"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "cross_attn.kv" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 8. stage_early (first few transformer blocks, e.g., 0-3)
    echo "Running 4-bit Quantization: stage_early (blocks 0-3)"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__stage_early.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/stage_early"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.blocks.[0-3]." --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 9. stage_mid (middle transformer blocks, e.g., 4-7)
    echo "Running 4-bit Quantization: stage_mid (blocks 4-7)"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__stage_mid.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/stage_mid"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.blocks.[4-7]." --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 10. stage_late (last few transformer blocks, e.g., 8-11)
    echo "Running 4-bit Quantization: stage_late (blocks 8-11)"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__stage_late.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/stage_late"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.blocks.([8-9]|1[0-1])." --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 11. norms_all (all normalization layers)
    echo "Running 4-bit Quantization: norms_all"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__norms_all.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/norms_all"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "norm" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"

    # 12. model.head (final output layer)
    echo "Running 4-bit Quantization: model.head"
    LOG_FILE="$RESULT_ROOT/logs/${TASK}__model_head.log"
    VIDEO_DIR="$RESULT_ROOT/videos/${TASK}/model_head"
    mkdir -p "$VIDEO_DIR"
    $BASE_EVAL_CMD --env-id $TASK --quant $QUANT_MODE --quant-compute-dtype $COMPUTE_DTYPE --quant-modules "model.head" --save-video-dir "$VIDEO_DIR" &> "$LOG_FILE"
done

python -m tools.summarize_eval_logs --log-dir "$RESULT_ROOT/logs" > "$RESULT_ROOT/summary.csv"
echo "Evaluation summary saved to $RESULT_ROOT/summary.csv"