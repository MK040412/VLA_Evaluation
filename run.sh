#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh <env_id> <clean_models>"
    echo "Example: ./run.sh <PegInsertionSide-v1|PickCube-v1|PlugCharger-v1|PushCube-v1|StackCube-v1> <true|false>"
    exit 1
fi

ENV_ID=$1
CLEAN_MODELS=$2

if [ "$CLEAN_MODELS" = "true" ]; then
    if [ -d "pretrained_models" ]; then
        echo "[INFO] Cleaning existing pretrained models..."
        rm -rf pretrained_models/*
    fi
fi

# Download lang_embeds
python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models lang_embeds/

# Download the pretrained model from Hugging Face if it does not exist
if [ ! -d "pretrained_models/openvla" ]; then
    echo "[INFO] openvla pretrained model not found. Downloading..."
    python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models "openvla-7b*/"
    mv pretrained_models/openvla-7b* pretrained_models/openvla
fi
if [ ! -d "pretrained_models/rdt" ]; then
    echo "[INFO] rdt pretrained model not found. Downloading..."
    python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt/
fi
if [ ! -d "pretrained_models/dissusion_policy" ]; then
    echo "[INFO] diffusion_policy pretrained model not found. Downloading..."
    python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models diffusion_policy/
fi
if [ ! -d "pretrained_models/octo" ]; then
    echo "[INFO] octo pretrained model not found. Downloading..."
    python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models octo/
fi

LANG_EMBEDS_PATH=pretrained_models/lang_embeds/text_embed_${ENV_ID}.pt

export PYTHONPATH=$(pwd)/src/evaluation



# Evaluate the model on ManiSkill tasks with live view
python -m src.evaluation.eval_rdt --env-id $ENV_ID --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt --lang_embeddings_path $LANG_EMBEDS_PATH --live-view
python -m src.evaluation.eval_dp --env-id $ENV_ID --pretrained_path pretrained_models/diffusion_policy/700.ckpt

# in progress
python -m src.evaluation.eval_openvla --env-id $ENV_ID --pretrained_path pretrained_models/openvla

export PYTHONPATH=$(pwd)/src/evaluation/octo
python -m src.evaluation.eval_octo --env-id $ENV_ID --pretrained_path pretrained_models/octo/experiment_20241208_112612