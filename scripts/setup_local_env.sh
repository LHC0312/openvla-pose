#!/usr/bin/env bash
# Local visualization/analysis env (Apple Silicon, no CUDA)
# OpenVLA training은 절대 여기서 안 합니다 — Colab 전용.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/local/.venv"

# 1. Python 3.10+ 확인 (miniforge에 있을 가능성 큼)
PY=""
for c in python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    PY="$c"
    break
  fi
done
[ -z "$PY" ] && { echo "Python 3.10+ not found"; exit 1; }
echo "Using Python: $($PY --version) at $(which $PY)"

# 2. venv 생성
if [ ! -d "$VENV_DIR" ]; then
  $PY -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# 3. 가벼운 패키지만 (torch는 MPS용 CPU/MPS wheel)
pip install \
  torch torchvision \
  numpy scipy matplotlib plotly \
  jupyter ipykernel \
  scikit-learn umap-learn \
  pillow opencv-python \
  pyyaml tqdm rich

# 4. e3nn / torch-geometric (CPU)
pip install e3nn
pip install torch_geometric

# 5. (선택) MediaPipe — 로컬에서 빠른 포즈 테스트용
pip install mediapipe || echo "mediapipe 설치 실패 — Colab에서 진행"

# 6. Jupyter 커널 등록
python -m ipykernel install --user --name openvla-pose-local --display-name "OpenVLA-Pose (local)"

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
