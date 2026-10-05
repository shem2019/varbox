#!/usr/bin/env bash
# Download the Olympic Boxing Punch Classification Video Dataset (~3.1 GB) onto the GPU machine.
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> Create New Token).
set -euo pipefail
DEST="${1:-$HOME/work/data/olympic}"
mkdir -p "$DEST"
if [ -n "$(find "$DEST" -name '*.mp4' -print -quit)" ]; then
  echo "dataset already present in $DEST"; exit 0
fi
kaggle datasets download -d piotrstefaskiue/olympic-boxing-punch-classification-video-dataset -p "$DEST" --unzip
# The adapter expects task_kam*_gh*/ folders; report what arrived.
find "$DEST" -maxdepth 3 -type d -name 'task_kam*' | head -5
echo "videos: $(find "$DEST" -name '*.mp4' | wc -l)"
