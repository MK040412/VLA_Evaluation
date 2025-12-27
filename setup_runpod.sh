#!/bin/bash
set -e

echo "=========================================="
echo "  RunPod Setup Script for VLA_Evaluation"
echo "=========================================="

cd /workspace/VLA_Evaluation

# 1. uv 설치 (없으면)
echo "[1/8] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "  -> uv installed"
else
    echo "  -> uv already installed"
fi

# 2. HuggingFace 캐시를 Volume으로 설정 (디스크 부족 방지)
echo "[2/8] Setting HuggingFace cache to volume storage..."
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME

if ! grep -q "HF_HOME=/workspace/.cache/huggingface" ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    echo "  -> Added HF_HOME to ~/.bashrc"
fi

# 3. Container disk 정리 (기존 캐시 삭제)
echo "[3/8] Cleaning container disk cache..."
if [ -d ~/.cache/huggingface ]; then
    rm -rf ~/.cache/huggingface
    echo "  -> Removed old cache from container disk"
fi

# 4. RDT2 저장소 클론
echo "[4/8] Cloning RDT2 repository..."
if [ -d "RDT2" ]; then
    echo "  -> RDT2 already exists, pulling latest..."
    cd RDT2 && git pull && cd ..
else
    git clone https://github.com/thu-ml/RDT2
    echo "  -> RDT2 cloned successfully"
fi

# 5. Python 의존성 설치 (uv 사용)
echo "[5/8] Installing Python dependencies with uv..."
uv pip install --system "transformers>=4.40" accelerate qwen-vl-utils 2>/dev/null || pip install "transformers>=4.40" accelerate qwen-vl-utils
echo "  -> Dependencies installed"

# flash-attn 설치 시도 (실패해도 계속 진행 - sdpa fallback 사용)
echo "  -> Attempting to install flash-attn (optional)..."
uv pip install --system flash-attn --no-build-isolation 2>/dev/null || echo "  -> flash-attn skipped (will use sdpa fallback)"

# 6. vqvae __init__.py 생성 (RDT2에 없음)
echo "[6/8] Creating __init__.py files for vqvae module..."
if [ ! -f "/workspace/VLA_Evaluation/RDT2/vqvae/__init__.py" ]; then
    echo "from .models.multivqvae import MultiVQVAE" > /workspace/VLA_Evaluation/RDT2/vqvae/__init__.py
    echo "  -> Created vqvae/__init__.py"
fi
if [ ! -f "/workspace/VLA_Evaluation/RDT2/vqvae/models/__init__.py" ]; then
    echo "from .multivqvae import MultiVQVAE" > /workspace/VLA_Evaluation/RDT2/vqvae/models/__init__.py
    echo "  -> Created vqvae/models/__init__.py"
fi

# 7. 심볼릭 링크 생성 (src/evaluation에서 RDT2 모듈 접근)
echo "[7/8] Creating symlinks for RDT2 modules..."

# 기존 링크/파일 제거
rm -f /workspace/VLA_Evaluation/src/evaluation/vqvae 2>/dev/null || true
rm -f /workspace/VLA_Evaluation/src/evaluation/models 2>/dev/null || true
rm -f /workspace/VLA_Evaluation/src/evaluation/utils.py 2>/dev/null || true

# 새 심볼릭 링크 생성
ln -sf /workspace/VLA_Evaluation/RDT2/vqvae /workspace/VLA_Evaluation/src/evaluation/vqvae
ln -sf /workspace/VLA_Evaluation/RDT2/models /workspace/VLA_Evaluation/src/evaluation/models
ln -sf /workspace/VLA_Evaluation/RDT2/utils.py /workspace/VLA_Evaluation/src/evaluation/utils.py
echo "  -> Created symlinks: vqvae, models, utils.py"

# 8. PYTHONPATH 설정
echo "[8/8] Setting PYTHONPATH..."
export PYTHONPATH=/workspace/VLA_Evaluation/src/evaluation:/workspace/VLA_Evaluation/RDT2:$PYTHONPATH

if ! grep -q "PYTHONPATH=.*RDT2" ~/.bashrc 2>/dev/null; then
    echo 'export PYTHONPATH=/workspace/VLA_Evaluation/src/evaluation:/workspace/VLA_Evaluation/RDT2:$PYTHONPATH' >> ~/.bashrc
    echo "  -> Added PYTHONPATH to ~/.bashrc"
fi

# Import 테스트
echo ""
echo "Testing imports..."
python -c "
from vqvae import MultiVQVAE
from models.normalizer import LinearNormalizer
from utils import batch_predict_action
print('All imports successful!')
" 2>/dev/null && echo "  -> Import test passed" || echo "  -> Import test failed (run 'source ~/.bashrc' first)"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Now run:"
echo "  source ~/.bashrc"
echo "  ./run.sh PickCube-v1 rdt2 --save-video"
echo ""
