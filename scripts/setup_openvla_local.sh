#!/usr/bin/env bash
# Add OpenVLA-OFT to existing libero-sim conda env.
# Run AFTER setup_libero_local.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$PROJECT_ROOT/external"
OFT_DIR="$EXTERNAL_DIR/openvla-oft"
ENV_NAME="libero-sim"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 1. Clone (lightweight ~2MB)
if [ ! -d "$OFT_DIR" ]; then
  git clone --depth 1 https://github.com/moojink/openvla-oft.git "$OFT_DIR"
fi

# 2. pip install OFT (this drags torch 2.2.0, transformers 4.40.1, tensorflow, etc.)
pip install -e "$OFT_DIR"

# 3. transformers fork (required for OFT eval/inference)
pip install git+https://github.com/moojink/transformers-openvla-oft.git

# NOTE: flash-attn is CUDA-only — skipped on macOS. Eager attention used instead.

echo ""
echo "Done. Try:"
echo "  conda activate $ENV_NAME"
echo "  python local/openvla_inference.py --device mps"
echo ""
echo "WARNING: First inference downloads ~15GB to ~/.cache/huggingface/"
