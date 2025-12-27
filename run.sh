#!/bin/bash
set -e

# Evaluation script with video recording support

usage() {
    echo "Usage: ./run.sh <env_id> <model> [options]"
    echo ""
    echo "Arguments:"
    echo "  env_id        Environment ID (PegInsertionSide-v1|PickCube-v1|PlugCharger-v1|PushCube-v1|StackCube-v1)"
    echo "  model         Model to evaluate (rdt|rdt2|dp|openvla|octo|all)"
    echo ""
    echo "Options:"
    echo "  --clean       Clean existing pretrained models before downloading"
    echo "  --save-video  Enable video recording with RecordEpisode"
    echo "  --live-view   Show live view during evaluation"
    echo "  --output-dir  Video output directory (default: videos)"
    echo ""
    echo "RDT-2 specific options:"
    echo "  --robot       Robot to use [default: panda_wristcam]"
    echo "                Options: panda|panda_wristcam|xarm6_robotiq_wristcam"
    echo "                panda_wristcam has hand_camera for wrist view (recommended)"
    echo "  --approach    Action conversion approach [default: delta]"
    echo "                Options: delta|raw|rdt2_style|direct|vae_raw|fix_scale|vae_normalize"
    echo "                  delta       - Compute deltas between consecutive predictions"
    echo "                  raw         - Use output directly (clipped to [-1,1])"
    echo "                  direct      - Scale position/rotation directly"
    echo "                  vae_raw     - Skip normalizer, use raw VAE output"
    echo "                  fix_scale   - Fix normalizer scale if too small"
    echo "                  vae_normalize - Normalize VAE output to [-1,1]"
    echo "  --pos-scale   Position scale factor [default: 1.0]"
    echo "  --rot-scale   Rotation scale factor [default: 1.0]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh PickCube-v1 rdt --save-video"
    echo "  ./run.sh PickCube-v1 rdt2 --save-video --approach delta --pos-scale 5.0"
    echo "  ./run.sh PickCube-v1 rdt2 --approach vae_raw --pos-scale 1.0 --save-video"
    echo "  ./run.sh PickCube-v1 rdt2 --approach fix_scale --save-video"
    echo "  ./run.sh PickCube-v1 all --clean"
    exit 1
}

if [ "$#" -lt 2 ]; then
    usage
fi

ENV_ID=$1
MODEL=$2
shift 2

# Parse optional arguments
CLEAN_MODELS=false
SAVE_VIDEO=""
LIVE_VIEW=""
OUTPUT_DIR="videos"
# RDT-2 specific
APPROACH="delta"
POS_SCALE="1.0"
ROT_SCALE="1.0"
ROBOT="panda_wristcam"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_MODELS=true
            shift
            ;;
        --save-video)
            SAVE_VIDEO="--save-video"
            shift
            ;;
        --live-view)
            LIVE_VIEW="--live-view"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --approach)
            APPROACH="$2"
            shift 2
            ;;
        --pos-scale)
            POS_SCALE="$2"
            shift 2
            ;;
        --rot-scale)
            ROT_SCALE="$2"
            shift 2
            ;;
        --robot)
            ROBOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ "$CLEAN_MODELS" = "true" ]; then
    if [ -d "pretrained_models" ]; then
        echo "[INFO] Cleaning existing pretrained models..."
        rm -rf pretrained_models/*
    fi
fi

# Create output directory for videos
if [ -n "$SAVE_VIDEO" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "[INFO] Videos will be saved to: $OUTPUT_DIR"
fi

# Download lang_embeds (needed for RDT-1B)
download_lang_embeds() {
    if [ ! -d "pretrained_models/lang_embeds" ]; then
        echo "[INFO] Downloading language embeddings..."
        python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models lang_embeds/
    fi
}

# Download RDT-1B model
download_rdt() {
    download_lang_embeds
    if [ ! -d "pretrained_models/rdt" ]; then
        echo "[INFO] rdt pretrained model not found. Downloading..."
        python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt/
    fi
}

# Download RDT-2 model (from HuggingFace Hub)
download_rdt2() {
    echo "[INFO] RDT-2 will be downloaded automatically from HuggingFace Hub on first run."
    echo "[INFO] Model: robotics-diffusion-transformer/RDT2-VQ"
    echo "[INFO] VAE: robotics-diffusion-transformer/RVQActionTokenizer"

    # Download normalizer if not exists
    if [ ! -f "pretrained_models/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt" ]; then
        mkdir -p pretrained_models/rdt2
        echo "[INFO] Downloading RDT-2 normalizer..."
        wget -q -O pretrained_models/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt \
            "http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt" || \
            echo "[WARN] Could not download normalizer. RDT-2 may not work correctly."
    fi
}

# Download Diffusion Policy model
download_dp() {
    if [ ! -d "pretrained_models/diffusion_policy" ]; then
        echo "[INFO] diffusion_policy pretrained model not found. Downloading..."
        python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models diffusion_policy/
    fi
}

# Download OpenVLA model
download_openvla() {
    if [ ! -d "pretrained_models/openvla" ]; then
        echo "[INFO] openvla pretrained model not found. Downloading..."
        python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models "openvla-7b*/"
        mv pretrained_models/openvla-7b* pretrained_models/openvla
    fi
}

# Download Octo model
download_octo() {
    if [ ! -d "pretrained_models/octo" ]; then
        echo "[INFO] octo pretrained model not found. Downloading..."
        python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models octo/
    fi
}

# RDT2 must be in PYTHONPATH for internal imports (data.umi, models.normalizer, etc.)
export PYTHONPATH=$(pwd)/RDT2:$(pwd)/src/evaluation:$PYTHONPATH

LANG_EMBEDS_PATH=pretrained_models/lang_embeds/text_embed_${ENV_ID}.pt

# Evaluate based on model selection
case $MODEL in
    rdt)
        download_rdt
        echo "[INFO] Evaluating RDT-1B on $ENV_ID..."
        python -m src.evaluation.eval_rdt \
            --env-id $ENV_ID \
            --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt \
            --lang_embeddings_path $LANG_EMBEDS_PATH \
            $SAVE_VIDEO \
            $LIVE_VIEW \
            --output-dir "$OUTPUT_DIR"
        ;;
    rdt2)
        download_rdt2
        # Create approach-specific output directory
        RDT2_OUTPUT_DIR="${OUTPUT_DIR}/rdt2_${ROBOT}_${APPROACH}_ps${POS_SCALE}_rs${ROT_SCALE}"
        echo "[INFO] Evaluating RDT-2 on $ENV_ID..."
        echo "[INFO] Robot: $ROBOT"
        echo "[INFO] Approach: $APPROACH, pos_scale: $POS_SCALE, rot_scale: $ROT_SCALE"
        echo "[INFO] Output directory: $RDT2_OUTPUT_DIR"
        python -m src.evaluation.eval_rdt2 \
            --env-id $ENV_ID \
            --robot "$ROBOT" \
            --pretrained_path robotics-diffusion-transformer/RDT2-VQ \
            --vae_path robotics-diffusion-transformer/RVQActionTokenizer \
            --normalizer_path pretrained_models/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt \
            --approach "$APPROACH" \
            --pos-scale "$POS_SCALE" \
            --rot-scale "$ROT_SCALE" \
            $SAVE_VIDEO \
            $LIVE_VIEW \
            --output-dir "$RDT2_OUTPUT_DIR"
        ;;
    dp)
        download_dp
        echo "[INFO] Evaluating Diffusion Policy on $ENV_ID..."
        python -m src.evaluation.eval_dp \
            --env-id $ENV_ID \
            --pretrained_path pretrained_models/diffusion_policy/700.ckpt \
            --vis \
            $SAVE_VIDEO \
            --output-dir "$OUTPUT_DIR"
        ;;
    openvla)
        download_openvla
        echo "[INFO] Evaluating OpenVLA on $ENV_ID..."
        python -m src.evaluation.eval_openvla \
            --env-id $ENV_ID \
            --pretrained_path pretrained_models/openvla \
            $SAVE_VIDEO \
            --output-dir "$OUTPUT_DIR"
        ;;
    octo)
        download_octo
        export PYTHONPATH=$(pwd)/RDT2:$(pwd)/src/evaluation/octo:$PYTHONPATH
        echo "[INFO] Evaluating Octo on $ENV_ID..."
        python -m src.evaluation.eval_octo \
            --env-id $ENV_ID \
            --pretrained_path pretrained_models/octo/experiment_20241208_112612 \
            $SAVE_VIDEO \
            --output-dir "$OUTPUT_DIR"
        ;;
    all)
        echo "[INFO] Evaluating all models on $ENV_ID..."

        # RDT-1B
        download_rdt
        echo "[INFO] Running RDT-1B..."
        python -m src.evaluation.eval_rdt \
            --env-id $ENV_ID \
            --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt \
            --lang_embeddings_path $LANG_EMBEDS_PATH \
            $SAVE_VIDEO \
            $LIVE_VIEW \
            --output-dir "$OUTPUT_DIR"

        # Diffusion Policy
        download_dp
        echo "[INFO] Running Diffusion Policy..."
        python -m src.evaluation.eval_dp \
            --env-id $ENV_ID \
            --pretrained_path pretrained_models/diffusion_policy/700.ckpt \
            --vis \
            $SAVE_VIDEO \
            --output-dir "$OUTPUT_DIR"

        echo "[INFO] All evaluations complete!"
        ;;
    *)
        echo "Unknown model: $MODEL"
        usage
        ;;
esac

echo "[INFO] Evaluation complete for $MODEL on $ENV_ID"
