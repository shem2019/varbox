#!/usr/bin/env bash
# Full VideoMAE strike-classifier training on CUDA, then evaluation on the held-out test split.
#   bash scripts/gpu/train_videomae.sh [dataset_dir]
set -euo pipefail
DATA="${1:-$HOME/work/data/olympic}"
DATA_DIR="$(dirname "$(find "$DATA" -type d -name 'task_kam*' -print -quit)")"
OUT=models/varbox-videomae-cuda
python -m boxing_analytics.training.train_videomae \
  --dataset-dir "$DATA_DIR" \
  --audit-output-dir artifacts/olympic_dataset_audit_gpu \
  --checkpoint models/videomae-base-finetuned-kinetics \
  --output-dir "$OUT" \
  --label-schema five_class \
  --device cuda --batch-size "${BATCH:-8}" --gradient-accumulation 2 \
  --epochs "${EPOCHS:-15}" --learning-rate 5e-5 --patience 4
python -m boxing_analytics.training.evaluate_videomae \
  --manifest artifacts/olympic_dataset_audit_gpu/generated_manifest.jsonl \
  --model-dir "$OUT/best" --split test --device cuda \
  --output artifacts/videomae_cuda_test_metrics.json
