#!/bin/bash
set -e

echo "=========================================="
echo "  RunPod Setup Script for VLA_Evaluation"
echo "=========================================="

cd /workspace/VLA_Evaluation

# 1. uv 설치 (없으면)
echo "[1/7] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "  -> uv installed"
else
    echo "  -> uv already installed"
fi

# 2. HuggingFace 캐시를 Volume으로 설정 (디스크 부족 방지)
echo "[2/7] Setting HuggingFace cache to volume storage..."
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME

if ! grep -q "HF_HOME=/workspace/.cache/huggingface" ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    echo "  -> Added HF_HOME to ~/.bashrc"
fi

# 3. Container disk 정리 (기존 캐시 삭제)
echo "[3/7] Cleaning container disk cache..."
if [ -d ~/.cache/huggingface ]; then
    rm -rf ~/.cache/huggingface
    echo "  -> Removed old cache from container disk"
fi

# 4. RDT2 저장소 클론
echo "[4/7] Cloning RDT2 repository..."
if [ -d "RDT2" ]; then
    echo "  -> RDT2 already exists, pulling latest..."
    cd RDT2 && git pull && cd ..
else
    git clone https://github.com/thu-ml/RDT2
    echo "  -> RDT2 cloned successfully"
fi

# 5. Python 의존성 설치 (uv 사용)
echo "[5/7] Installing Python dependencies with uv..."
uv pip install --system "transformers>=4.40" accelerate qwen-vl-utils
uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo "  -> Dependencies installed"

# flash-attn 설치 시도 (실패해도 계속 진행 - sdpa fallback 사용)
echo "  -> Attempting to install flash-attn (optional)..."
uv pip install --system flash-attn --no-build-isolation 2>/dev/null || echo "  -> flash-attn skipped (will use sdpa fallback)"

# 6. PYTHONPATH 설정 (vqvae는 RDT2 루트에 있음)
echo "[6/7] Setting PYTHONPATH..."
export PYTHONPATH=/workspace/VLA_Evaluation/RDT2:/workspace/VLA_Evaluation/RDT2/vqvae:/workspace/VLA_Evaluation/RDT2/models:$PYTHONPATH

if ! grep -q "PYTHONPATH=.*RDT2" ~/.bashrc 2>/dev/null; then
    echo 'export PYTHONPATH=/workspace/VLA_Evaluation/RDT2:/workspace/VLA_Evaluation/RDT2/vqvae:/workspace/VLA_Evaluation/RDT2/models:$PYTHONPATH' >> ~/.bashrc
    echo "  -> Added PYTHONPATH to ~/.bashrc"
fi

# 7. vqvae 심볼릭 링크 생성
echo "[7/7] Creating symlinks for vqvae and models modules..."

if [ ! -L "/workspace/VLA_Evaluation/vqvae" ] && [ ! -d "/workspace/VLA_Evaluation/vqvae" ]; then
    ln -sf /workspace/VLA_Evaluation/RDT2/vqvae /workspace/VLA_Evaluation/vqvae 2>/dev/null || true
    echo "  -> Created vqvae symlink"
fi

if [ ! -L "/workspace/VLA_Evaluation/models" ] && [ ! -d "/workspace/VLA_Evaluation/models" ]; then
    ln -sf /workspace/VLA_Evaluation/RDT2/models /workspace/VLA_Evaluation/models 2>/dev/null || true
    echo "  -> Created models symlink"
fi

if [ ! -L "/workspace/VLA_Evaluation/src/evaluation/vqvae" ] && [ ! -d "/workspace/VLA_Evaluation/src/evaluation/vqvae" ]; then
    ln -sf /workspace/VLA_Evaluation/RDT2/vqvae /workspace/VLA_Evaluation/src/evaluation/vqvae 2>/dev/null || true
    echo "  -> Created src/evaluation/vqvae symlink"
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Now run:"
echo "  source ~/.bashrc"
echo "  ./run.sh PickCube-v1 rdt2 --save-video"
echo ""
