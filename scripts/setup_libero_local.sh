#!/usr/bin/env bash
# Local LIBERO sim setup (macOS / Linux)
# Sets up a conda env `libero-sim` with mujoco + robosuite + LIBERO
# for opening a GLFW viewer to inspect tasks. OpenVLA inference NOT included
# (use Colab notebook 00 for that).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$PROJECT_ROOT/external"
LIBERO_DIR="$EXTERNAL_DIR/LIBERO"
ENV_NAME="libero-sim"

# 1. Conda
if ! command -v conda >/dev/null; then
  echo "conda not found. Install miniforge first."; exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2. Env
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "Creating conda env: $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

# 3. LIBERO clone
mkdir -p "$EXTERNAL_DIR"
if [ ! -d "$LIBERO_DIR" ]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
# Add empty __init__.py so find_packages() picks up libero.libero etc.
touch "$LIBERO_DIR/libero/__init__.py"

# 4. Install
pip install --quiet -e "$LIBERO_DIR"
pip install --quiet \
  mujoco "robosuite==1.4.0" bddl \
  numpy opencv-python "gym==0.25.2" \
  hydra-core easydict matplotlib cloudpickle einops future \
  imageio imageio-ffmpeg termcolor h5py thop

# 5. LIBERO config (skip interactive prompt)
mkdir -p "$HOME/.libero"
LIB_BASE="$LIBERO_DIR/libero/libero"
cat > "$HOME/.libero/config.yaml" <<EOF
benchmark_root: $LIB_BASE
bddl_files: $LIB_BASE/bddl_files
init_states: $LIB_BASE/init_files
datasets: $LIB_BASE/../datasets
assets: $LIB_BASE/assets
EOF

echo ""
echo "Done. Try:"
echo "  conda activate $ENV_NAME"
echo "  python local/libero_view.py --list                    # see tasks"
echo "  python local/libero_view.py --suite libero_spatial --task 0"
