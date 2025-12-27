#!/bin/bash
# RDT-2 Multi-Approach Experiments
# This script runs RDT-2 evaluation with different conversion approaches and scales

set -e

# Configuration
ENVS=("PickCube-v1")  # Add more: "PushCube-v1" "StackCube-v1"
APPROACHES=("delta" "raw" "rdt2_style" "direct")
SCALES=("0.5" "1.0" "2.0" "5.0" "10.0")
NUM_TRAJ=5  # Reduce for faster testing
MAX_STEPS=200
OUTPUT_BASE="results/rdt2_experiments"

# Ensure RDT2 is in PYTHONPATH
export PYTHONPATH=$(pwd)/RDT2:$(pwd)/src/evaluation:$PYTHONPATH

# Create results directory
mkdir -p "$OUTPUT_BASE"

# Log file
LOG_FILE="$OUTPUT_BASE/experiment_log_$(date +%Y%m%d_%H%M%S).txt"
echo "RDT-2 Multi-Approach Experiment Log" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

run_experiment() {
    local env=$1
    local approach=$2
    local pos_scale=$3
    local rot_scale=$4
    local output_dir="${OUTPUT_BASE}/${env}/${approach}_ps${pos_scale}_rs${rot_scale}"

    echo ""
    echo "==========================================="
    echo "Experiment: $env / $approach / pos_scale=$pos_scale / rot_scale=$rot_scale"
    echo "==========================================="

    mkdir -p "$output_dir"

    # Run the evaluation
    python -m src.evaluation.eval_rdt2 \
        --env-id "$env" \
        --approach "$approach" \
        --pos-scale "$pos_scale" \
        --rot-scale "$rot_scale" \
        --num-traj "$NUM_TRAJ" \
        --max-steps "$MAX_STEPS" \
        --save-video \
        --output-dir "$output_dir" \
        2>&1 | tee "$output_dir/output.log"

    # Extract success rate from log
    success_rate=$(grep "Success rate:" "$output_dir/output.log" | tail -1 || echo "N/A")
    echo "$env | $approach | pos=$pos_scale | rot=$rot_scale | $success_rate" >> "$LOG_FILE"
}

# Main experiment loop
echo "Starting RDT-2 experiments..."
echo "Environments: ${ENVS[*]}"
echo "Approaches: ${APPROACHES[*]}"
echo "Scales: ${SCALES[*]}"
echo ""

for env in "${ENVS[@]}"; do
    echo ""
    echo "============================================="
    echo "Environment: $env"
    echo "============================================="

    for approach in "${APPROACHES[@]}"; do
        if [ "$approach" == "raw" ]; then
            # Raw approach doesn't use scales
            run_experiment "$env" "$approach" "1.0" "1.0"
        else
            # Test different scales for delta, rdt2_style, direct
            for scale in "${SCALES[@]}"; do
                run_experiment "$env" "$approach" "$scale" "$scale"
            done
        fi
    done
done

echo ""
echo "==========================================="
echo "All experiments completed!"
echo "==========================================="
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo "Log file: $LOG_FILE"
echo ""
echo "Summary:"
cat "$LOG_FILE"
