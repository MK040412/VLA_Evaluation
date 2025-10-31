#!/bin/bash
set -e

ROOT_DIR=$(dirname $(realpath $0))
SCRIPT_DIR=${ROOT_DIR}/script

ENV_NAME="$(basename "$(pwd)")"
ENV_DIR=env/"$ENV_NAME"

sudo apt install -y curl libgtk2.0-dev pkg-config

if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment at $ENV_DIR"
    uv venv "$ENV_DIR"
else
    echo "Virtual environment already exists at $ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
uv pip install -r requirements.txt
uv pip install --no-cache-dir --verbose --no-build-isolation flash-attn

git submodule update --init --recursive
git config commit.template commit_msg_template.txt

#uv pip install lerobot==0.4.0 
#uv pip uninstall opencv-python opencv-python-headless opencv-contrib-python
#uv pip install opencv-python==4.12.0.88

echo ""
echo -e "\033[1;32m========================================\033[0m"
echo -e "\033[1;92mSetup complete!\033[0m"
echo -e "\033[1;92mTo activate the environment, run the following command:\033[0m"
echo -e "\033[1;32m> source $ENV_DIR/bin/activate\033[0m"
echo -e "\033[1;32m========================================\033[0m"
