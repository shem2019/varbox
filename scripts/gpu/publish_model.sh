#!/usr/bin/env bash
# Upload the trained strike classifier to a PRIVATE Hugging Face repo so any new GPU machine gets it
# from setup.sh. Private on purpose: the Olympic dataset and the Kinetics VideoMAE base are
# non-commercial. Needs a Hugging Face token with write access (hf auth login).
#   bash scripts/gpu/publish_model.sh [model_dir] [repo_id]
set -euo pipefail
MODEL="${1:-models/varbox-videomae-cuda/best}"
REPO_ID="${2:-${VARBOX_VIDEOMAE_REPO:-shemking/varbox-videomae-strike}}"
[ -f "$MODEL/model.safetensors" ] || { echo "no model at $MODEL"; exit 1; }
hf repo create "$REPO_ID" --repo-type model --private --exist-ok 2>/dev/null \
  || python -c "from huggingface_hub import create_repo; create_repo('$REPO_ID', private=True, exist_ok=True)"
hf upload "$REPO_ID" "$MODEL" . --repo-type model --commit-message "VAR Box strike classifier $(date +%F)"
echo "uploaded $MODEL -> https://huggingface.co/$REPO_ID (private)"
