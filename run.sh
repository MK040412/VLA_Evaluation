#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh <env_id>"
    echo "Example: ./run.sh <PegInsertionSide-v1|PickCube-v1|PlugCharger-v1|PushCube-v1|StackCube-v1>"
    exit 1
fi

ENV_ID=$1

# Download the pretrained model from Hugging Face
python -m src.utility.download_hf_model robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt/ octo/ diffusion_policy/ lang_embeds/ "openvla-7b*/"

LANG_EMBEDS_PATH=pretrained_models/lang_embeds/text_embed_${ENV_ID}.pt

export PYTHONPATH=$(pwd)/src/evaluation

# Evaluate the model on ManiSkill tasks with live view
python -m src.evaluation.eval_rdt --env-id $ENV_ID --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt --lang_embeddings_path $LANG_EMBEDS_PATH --live-view
python -m src.evaluation.eval_dp --env-id $ENV_ID --pretrained_path pretrained_models/diffusion_policy/700.ckpt


# in progress
# python -m src.evaluation.eval_openvla --env-id $ENV_ID --pretrained_path pretrained_models/openvla-7b+example_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug+example_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug
# export PYTHONPATH=$(pwd)/src/evaluation/octo
# python -m src.evaluation.eval_octo --env-id $ENV_ID --pretrained_path pretrained_models/octo/experiment_20241208_112612