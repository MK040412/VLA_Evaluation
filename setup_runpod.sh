#!/bin/bash
set -e

echo "=========================================="
echo "  RunPod Setup Script for VLA_Evaluation"
echo "=========================================="

cd /workspace/VLA_Evaluation

# 1. uv 설치 (없으면)
echo "[1/9] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "  -> uv installed"
else
    echo "  -> uv already installed"
fi

# 2. HuggingFace 캐시를 Volume으로 설정 (디스크 부족 방지)
echo "[2/9] Setting HuggingFace cache to volume storage..."
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME

if ! grep -q "HF_HOME=/workspace/.cache/huggingface" ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    echo "  -> Added HF_HOME to ~/.bashrc"
fi

# 3. Container disk 정리 (기존 캐시 삭제)
echo "[3/9] Cleaning container disk cache..."
if [ -d ~/.cache/huggingface ]; then
    rm -rf ~/.cache/huggingface
    echo "  -> Removed old cache from container disk"
fi

# 4. RDT2 저장소 클론
echo "[4/9] Cloning RDT2 repository..."
if [ -d "RDT2" ]; then
    echo "  -> RDT2 already exists, pulling latest..."
    cd RDT2 && git pull && cd ..
else
    git clone https://github.com/thu-ml/RDT2
    echo "  -> RDT2 cloned successfully"
fi

# 5. uv venv 생성 (이름: rdt2)
echo "[5/9] Creating virtual environment 'rdt2' with uv..."
if [ ! -d "rdt2" ]; then
    uv venv rdt2 --python 3.9
    echo "  -> Created venv 'rdt2'"
else
    echo "  -> venv 'rdt2' already exists"
fi

# 6. venv 활성화 및 의존성 설치
echo "[6/9] Installing Python dependencies in venv..."
source rdt2/bin/activate

# 기본 의존성 설치
uv pip install "transformers>=4.40" accelerate qwen-vl-utils torch torchvision
uv pip install mani_skill gymnasium numpy pillow tqdm opencv-python
echo "  -> Dependencies installed"

# flash-attn 설치 시도 (실패해도 계속 진행 - sdpa fallback 사용)
echo "  -> Attempting to install flash-attn (optional)..."
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  -> flash-attn skipped (will use sdpa fallback)"

# 7. vqvae __init__.py 생성 (RDT2에 없음)
echo "[7/9] Creating __init__.py files for vqvae module..."
if [ ! -f "/workspace/VLA_Evaluation/RDT2/vqvae/__init__.py" ]; then
    echo "from .models.multivqvae import MultiVQVAE" > /workspace/VLA_Evaluation/RDT2/vqvae/__init__.py
    echo "  -> Created vqvae/__init__.py"
fi
if [ ! -f "/workspace/VLA_Evaluation/RDT2/vqvae/models/__init__.py" ]; then
    echo "from .multivqvae import MultiVQVAE" > /workspace/VLA_Evaluation/RDT2/vqvae/models/__init__.py
    echo "  -> Created vqvae/models/__init__.py"
fi

# 8. 심볼릭 링크 생성 (src/evaluation에서 RDT2 모듈 접근)
echo "[8/9] Creating symlinks for RDT2 modules..."

# 기존 링크/파일 제거
rm -f /workspace/VLA_Evaluation/src/evaluation/vqvae 2>/dev/null || true
rm -f /workspace/VLA_Evaluation/src/evaluation/models 2>/dev/null || true
rm -f /workspace/VLA_Evaluation/src/evaluation/utils.py 2>/dev/null || true

# 새 심볼릭 링크 생성
ln -sf /workspace/VLA_Evaluation/RDT2/vqvae /workspace/VLA_Evaluation/src/evaluation/vqvae
ln -sf /workspace/VLA_Evaluation/RDT2/models /workspace/VLA_Evaluation/src/evaluation/models
ln -sf /workspace/VLA_Evaluation/RDT2/utils.py /workspace/VLA_Evaluation/src/evaluation/utils.py
echo "  -> Created symlinks: vqvae, models, utils.py"

# 9. 활성화 스크립트 생성
echo "[9/9] Creating activation helper..."
cat > /workspace/VLA_Evaluation/activate_rdt2.sh << 'ACTIVATE_EOF'
#!/bin/bash
cd /workspace/VLA_Evaluation
source rdt2/bin/activate
export PYTHONPATH=/workspace/VLA_Evaluation/RDT2:/workspace/VLA_Evaluation/src/evaluation:$PYTHONPATH
export HF_HOME=/workspace/.cache/huggingface
echo "RDT2 environment activated!"
echo "Run: ./run.sh PickCube-v1 rdt2 --save-video"
ACTIVATE_EOF
chmod +x /workspace/VLA_Evaluation/activate_rdt2.sh

# Import 테스트
echo ""
echo "Testing imports..."
export PYTHONPATH=/workspace/VLA_Evaluation/RDT2:/workspace/VLA_Evaluation/src/evaluation:$PYTHONPATH
python -c "
from vqvae import MultiVQVAE
from models.normalizer import LinearNormalizer
from utils import batch_predict_action
print('All imports successful!')
" && echo "  -> Import test passed" || echo "  -> Import test failed"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source activate_rdt2.sh"
echo ""
echo "Then run:"
echo "  ./run.sh PickCube-v1 rdt2 --save-video"
echo ""
