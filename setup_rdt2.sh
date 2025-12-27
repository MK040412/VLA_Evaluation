#!/bin/bash
set -e

# Setup script for RDT-2 dependencies on RunPod

echo "[INFO] Setting up RDT-2 dependencies..."

# Install required packages
pip install transformers>=4.40.0 accelerate

# Install flash-attention (optional but recommended)
echo "[INFO] Installing flash-attention (may take a while)..."
pip install flash-attn --no-build-isolation || echo "[WARN] flash-attn installation failed. Will use sdpa fallback."

# Clone RDT2 repository for vqvae and utils modules
if [ ! -d "RDT2" ]; then
    echo "[INFO] Cloning RDT2 repository..."
    git clone https://github.com/thu-ml/RDT2.git
fi

# Add RDT2 to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/RDT2"
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)/RDT2\"" >> ~/.bashrc

# Download normalizer
mkdir -p pretrained_models/rdt2
if [ ! -f "pretrained_models/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt" ]; then
    echo "[INFO] Downloading RDT-2 normalizer..."
    wget -q -O pretrained_models/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt \
        "http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt"
fi

echo "[INFO] RDT-2 setup complete!"
echo ""
echo "To use RDT-2, run:"
echo "  ./run.sh PickCube-v1 rdt2 --save-video"
