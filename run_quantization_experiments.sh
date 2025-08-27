#!/bin/bash
#
# This script runs the second step of the quantization experiment:
# creating quantized checkpoints for each specified group.

# --- CONFIGURATION -- -
# 1. PLEASE UPDATE THIS: Set the path to your original, non-quantized checkpoint.
ORIGINAL_CHECKPOINT="/home/perelman/RoboticsDiffusionTransformer/pretrained_models/rdt/mp_rank_00_model_states.pt"

# 2. Set the output directory for the generated 6-bit quantized checkpoints.
OUTPUT_DIR="quantized_checkpoints_6bit"

# 3. List of priority groups for the single-group 6-bit quantization experiment.
# These names should match the groups defined in tools/priority_groups.json
GROUPS=(
    "embeddings_and_adaptors"
    "attn_qkv_block_0"
    "attn_qkv_block_9"
    "attn_qkv_block_18"
    "attn_proj_block_9"
    "ffn_fc1_block_9"
    "cross_q_all"
    "stage_early"
    "norms_all"
)

# 4. Combination experiments
COMBINATIONS=(
    "attn_qkv_block_9,attn_proj_block_9"
    "ffn_fc1_block_9,ffn_fc2_block_9"
    "cross_q_all,cross_kv_all"
    "embeddings_and_adaptors,stage_early"
)

# --- END OF CONFIGURATION ---


# --- SCRIPT EXECUTION ---
set -e # Exit immediately if a command exits with a non-zero status.

# Check if the original checkpoint file exists
if [ ! -f "$ORIGINAL_CHECKPOINT" ]; then
    echo "Error: Original checkpoint not found at '$ORIGINAL_CHECKPOINT'"
    echo "Please update the ORIGINAL_CHECKPOINT variable in this script before running."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory for quantized checkpoints: $OUTPUT_DIR"
echo ""

# Loop through the groups and run the quantization script for each
for GROUP in "${GROUPS[@]}"; do
    echo "--------------------------------------------------"
    echo "Quantizing group: '$GROUP' to 6 bits"
    echo "--------------------------------------------------"

    OUTPUT_CHECKPOINT_PATH="${OUTPUT_DIR}/ckpt_quant_${GROUP}_6bit.pt"

    python tools/quantize_rdt_weights_adv.py \
        --in "$ORIGINAL_CHECKPOINT" \
        --out "$OUTPUT_CHECKPOINT_PATH" \
        --group "$GROUP" \
        --bits 6

    if [ $? -eq 0 ]; then
        echo "Successfully created quantized checkpoint: $OUTPUT_CHECKPOINT_PATH"
    else
        echo "Error: Failed to quantize group '$GROUP'. Aborting."
        exit 1
    fi
    echo ""
done

# Loop through the combinations and run the quantization script for each
for COMBO in "${COMBINATIONS[@]}"; do
    echo "--------------------------------------------------"
    echo "Quantizing combination: '$COMBO' to 6 bits"
    echo "--------------------------------------------------"

    # Convert comma-separated string to array of groups
    IFS=',' read -r -a GROUP_ARRAY <<< "$COMBO"

    # Build the group arguments for the python script
    GROUP_ARGS=()
    for GROUP in "${GROUP_ARRAY[@]}"; do
        GROUP_ARGS+=("--group" "$GROUP")
    done

    # Create a filename-friendly name for the combination
    COMBO_NAME=$(echo "$COMBO" | tr ',' '_')

    OUTPUT_CHECKPOINT_PATH="${OUTPUT_DIR}/ckpt_quant_combo_${COMBO_NAME}_6bit.pt"

    python tools/quantize_rdt_weights_adv.py \
        --in "$ORIGINAL_CHECKPOINT" \
        --out "$OUTPUT_CHECKPOINT_PATH" \
        "${GROUP_ARGS[@]}" \
        --bits 6

    if [ $? -eq 0 ]; then
        echo "Successfully created quantized checkpoint: $OUTPUT_CHECKPOINT_PATH"
    else
        echo "Error: Failed to quantize combination '$COMBO'. Aborting."
        exit 1
    fi
    echo ""
done

echo "--------------------------------------------------"
echo "All quantization tasks completed successfully."
echo "You can find the generated checkpoints in the '$OUTPUT_DIR' directory."
echo "Next step is to generate manifests and run the evaluation."
echo "--------------------------------------------------"