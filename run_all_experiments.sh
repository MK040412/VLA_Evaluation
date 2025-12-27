#!/bin/bash
# RDT-2 Multi-Approach Experiments
# Tests all possible approaches with 10 trials each

set -e

# Configuration
ENV_ID="${1:-PickCube-v1}"
NUM_TRAJ="${2:-10}"
OUTPUT_BASE="results/rdt2_multi_approach"
LOG_FILE="${OUTPUT_BASE}/experiment_log_$(date +%Y%m%d_%H%M%S).txt"

# Ensure RDT2 is in PYTHONPATH
export PYTHONPATH=$(pwd)/RDT2:$(pwd)/src/evaluation:$PYTHONPATH

# Create results directory
mkdir -p "$OUTPUT_BASE"

echo "RDT-2 Multi-Approach Experiment" | tee "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "Environment: $ENV_ID" | tee -a "$LOG_FILE"
echo "Trials per approach: $NUM_TRAJ" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Define approaches
# High priority (likely to work)
PRIORITY_APPROACHES=("vae_raw" "fix_scale" "direct")
# Standard approaches
STANDARD_APPROACHES=("delta" "raw" "rdt2_style" "vae_normalize")

# Define scales for manual testing
POS_SCALES=("0.1" "1.0" "10.0" "100.0")
ROT_SCALES=("0.01" "0.1" "1.0")

run_experiment() {
    local approach=$1
    local pos_scale=$2
    local rot_scale=$3
    local output_dir="${OUTPUT_BASE}/${approach}_ps${pos_scale}_rs${rot_scale}"

    echo ""
    echo "=========================================="
    echo "Approach: $approach | pos_scale: $pos_scale | rot_scale: $rot_scale"
    echo "=========================================="

    mkdir -p "$output_dir"

    python -m src.evaluation.eval_rdt2 \
        --env-id "$ENV_ID" \
        --approach "$approach" \
        --pos-scale "$pos_scale" \
        --rot-scale "$rot_scale" \
        --num-traj "$NUM_TRAJ" \
        --save-video \
        --output-dir "$output_dir" \
        2>&1 | tee "$output_dir/output.log"

    # Extract success rate
    success_rate=$(grep "Success rate:" "$output_dir/output.log" | tail -1 || echo "N/A")
    echo "$approach | pos=$pos_scale | rot=$rot_scale | $success_rate" | tee -a "$LOG_FILE"
}

# Run priority approaches first with various scales
echo "" | tee -a "$LOG_FILE"
echo "=== PRIORITY APPROACHES ===" | tee -a "$LOG_FILE"

for approach in "${PRIORITY_APPROACHES[@]}"; do
    for ps in "${POS_SCALES[@]}"; do
        for rs in "${ROT_SCALES[@]}"; do
            run_experiment "$approach" "$ps" "$rs"
        done
    done
done

# Run standard approaches with default scale
echo "" | tee -a "$LOG_FILE"
echo "=== STANDARD APPROACHES ===" | tee -a "$LOG_FILE"

for approach in "${STANDARD_APPROACHES[@]}"; do
    # Test with a few scale combinations
    run_experiment "$approach" "1.0" "1.0"
    run_experiment "$approach" "10.0" "0.1"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo "Log file: $LOG_FILE"
echo ""
echo "Summary:"
echo "--------"
cat "$LOG_FILE"
