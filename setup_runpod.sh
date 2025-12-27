#!/bin/bash
set -e

echo "=========================================="
echo "  RunPod Setup Script for VLA_Evaluation"
echo "=========================================="

# 1. HuggingFace 캐시를 Volume으로 설정 (디스크 부족 방지)
echo "[1/5] Setting HuggingFace cache to volume storage..."
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME

# .bashrc에 영구 적용
if ! grep -q "HF_HOME=/workspace/.cache/huggingface" ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    echo "  -> Added HF_HOME to ~/.bashrc"
fi

# 2. Container disk 정리 (기존 캐시 삭제)
echo "[2/5] Cleaning container disk cache..."
if [ -d ~/.cache/huggingface ]; then
    rm -rf ~/.cache/huggingface
    echo "  -> Removed old cache from container disk"
fi

# 3. RDT2 저장소 클론
echo "[3/5] Cloning RDT2 repository..."
cd /workspace/VLA_Evaluation

if [ -d "RDT2" ]; then
    echo "  -> RDT2 already exists, pulling latest..."
    cd RDT2 && git pull && cd ..
else
    git clone https://github.com/thu-ml/RDT2
    echo "  -> RDT2 cloned successfully"
fi

# 4. PYTHONPATH 설정 (vqvae는 RDT2 루트에 있음)
echo "[4/5] Setting PYTHONPATH..."
export PYTHONPATH=/workspace/VLA_Evaluation/RDT2:/workspace/VLA_Evaluation/RDT2/vqvae:/workspace/VLA_Evaluation/RDT2/models:$PYTHONPATH

# .bashrc에 영구 적용
if ! grep -q "PYTHONPATH=.*RDT2" ~/.bashrc 2>/dev/null; then
    echo 'export PYTHONPATH=/workspace/VLA_Evaluation/RDT2:/workspace/VLA_Evaluation/RDT2/vqvae:/workspace/VLA_Evaluation/RDT2/models:$PYTHONPATH' >> ~/.bashrc
    echo "  -> Added PYTHONPATH to ~/.bashrc"
fi

# 5. vqvae 심볼릭 링크 생성 (vqvae는 RDT2 루트에 있음!)
echo "[5/5] Creating symlinks for vqvae and models modules..."

# vqvae 링크 (RDT2/vqvae -> VLA_Evaluation/vqvae)
if [ ! -L "/workspace/VLA_Evaluation/vqvae" ] && [ ! -d "/workspace/VLA_Evaluation/vqvae" ]; then
    ln -sf /workspace/VLA_Evaluation/RDT2/vqvae /workspace/VLA_Evaluation/vqvae 2>/dev/null || true
    echo "  -> Created vqvae symlink"
fi

# models 디렉토리 링크 (RDT2/models -> VLA_Evaluation/models)
if [ ! -L "/workspace/VLA_Evaluation/models" ] && [ ! -d "/workspace/VLA_Evaluation/models" ]; then
    ln -sf /workspace/VLA_Evaluation/RDT2/models /workspace/VLA_Evaluation/models 2>/dev/null || true
    echo "  -> Created models symlink"
fi

# src/evaluation에도 vqvae 링크 (import 경로 호환성)
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
echo "Or in one line:"
echo "  source ~/.bashrc && ./run.sh PickCube-v1 rdt2 --save-video"
echo ""
