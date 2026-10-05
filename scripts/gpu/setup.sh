#!/usr/bin/env bash
# One-shot setup of a rented NVIDIA GPU machine for VAR Box (4D meshes + VideoMAE training).
#
#   git clone -b feature/mesh4d https://github.com/shem2019/varbox.git ~/work/varbox
#   cd ~/work/varbox && bash scripts/gpu/setup.sh
#
# Re-running is safe: finished steps are skipped. Afterwards: source ~/work/env.sh
# Needs a Hugging Face token with access to facebook/sam-3d-body-dinov3 (gated): run
# `hf auth login` once, or export HF_TOKEN before this script.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-$(dirname "$REPO")}"
VENV="$WORK/venv"
SAM3D_DIR="$WORK/sam-3d-body"
MODELS="$REPO/models"
log() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null; then SUDO="sudo"; fi

log "System packages"
if ! command -v ffmpeg >/dev/null || ! ldconfig -p | grep -q libEGL; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq ffmpeg git libgl1 libglib2.0-0 libegl1 libgles2 build-essential ninja-build >/dev/null
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "Python 3.11 environment (uv)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PIP="uv pip install --quiet"

log "PyTorch for this driver"
CUDA_MAJOR_MINOR="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
case "$CUDA_MAJOR_MINOR" in
  12.[0-5]) TORCH_INDEX=cu124 ;;
  12.[6-7]) TORCH_INDEX=cu126 ;;
  *)        TORCH_INDEX=cu128 ;;
esac
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  $PIP torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

log "VAR Box and its server-side dependencies"
$PIP numpy scipy "opencv-python-headless" lap ultralytics pyyaml \
  "transformers>=4.57,<5" "accelerate>=1.10,<2" "safetensors>=0.6,<1" \
  "hydra-core>=1.3,<2" "iopath>=0.1.10,<1" "pillow>=10,<13" huggingface_hub kaggle pytest
$PIP -e "$REPO" --no-deps

log "SAM 2.1 (masks)"
if ! python -c "import sam2" 2>/dev/null; then
  SAM2_BUILD_ALLOW_ERRORS=1 $PIP --no-build-isolation \
    "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
fi
mkdir -p "$MODELS/sam2.1"
for name in sam2.1_hiera_large sam2.1_hiera_tiny; do
  [ -s "$MODELS/sam2.1/$name.pt" ] || curl -sSL -o "$MODELS/sam2.1/$name.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$name.pt"
done

log "SAM 3D Body (meshes)"
[ -d "$SAM3D_DIR/.git" ] || git clone -q https://github.com/facebookresearch/sam-3d-body.git "$SAM3D_DIR"
$PIP pytorch-lightning pyrender yacs scikit-image einops timm dill pandas rich hydra-submitit-launcher \
  hydra-colorlog pyrootutils webdataset chump "networkx==3.2.1" roma joblib jsonlines xtcocotools loguru \
  optree fvcore pycocotools braceexpand omegaconf trimesh cython
if ! python -c "import detectron2" 2>/dev/null; then
  $PIP --no-build-isolation --no-deps "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9"
fi
python -c "import moge" 2>/dev/null || $PIP "git+https://github.com/microsoft/MoGe.git"
# MoGe pulls the newest huggingface_hub; transformers 4.x (VideoMAE) needs < 1.0.
$PIP "huggingface_hub>=0.34,<1.0"

log "Model downloads (gated: facebook/sam-3d-body-dinov3)"
if ! python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("facebook/sam-3d-body-dinov3",):
    print("downloading", repo, "->", snapshot_download(repo_id=repo))
PY
then
  echo "!! SAM 3D Body download failed. Check that Meta approved your access request at"
  echo "!! https://huggingface.co/facebook/sam-3d-body-dinov3 and run: hf auth login"
  echo "!! Then re-run this script; finished steps are skipped."
  exit 3
fi
python -c "from huggingface_hub import snapshot_download as s; s('Ruicheng/moge-2-vitl-normal')" >/dev/null
python -c "from huggingface_hub import snapshot_download as s; s('MCG-NJU/videomae-base-finetuned-kinetics', local_dir='$MODELS/videomae-base-finetuned-kinetics')" >/dev/null
(cd "$REPO" && python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')" >/dev/null)

cat > "$WORK/env.sh" <<EOF
source "$VENV/bin/activate"
export SAM3D_BODY_DIR="$SAM3D_DIR"
export VARBOX_SAM2_CHECKPOINT="$MODELS/sam2.1/sam2.1_hiera_large.pt"
export PYOPENGL_PLATFORM=egl
export HF_HUB_DISABLE_TELEMETRY=1
cd "$REPO"
EOF

log "Smoke checks"
# shellcheck disable=SC1091
source "$WORK/env.sh"
python -m pytest -q tests/unit/test_mesh4d.py
python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["SAM3D_BODY_DIR"])
import sam2, sam_3d_body  # noqa: F401
from tools.build_fov_estimator import FOVEstimator  # noqa: F401
print("imports ok: sam2, sam_3d_body, MoGe FOV estimator")
PY
log "Ready. Next: source $WORK/env.sh"
