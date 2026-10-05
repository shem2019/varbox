#!/usr/bin/env bash
# 4D analysis of one Olympic bout segment seen by both GoPros (kam2 + kam4, same chapter number).
#   bash scripts/gpu/run_olympic_4d.sh [chapter] [start_s] [duration_s]
set -euo pipefail
CH="${1:-08}"; START="${2:-60}"; DUR="${3:-30}"
DATA="${DATA:-$HOME/work/data/olympic}"
A="$(find "$DATA" -path "*task_kam2_gh${CH}*" -name '*.mp4' | head -1)"
B="$(find "$DATA" -path "*task_kam4_gh${CH}*" -name '*.mp4' | head -1)"
[ -n "$A" ] || { echo "no kam2 video for chapter $CH under $DATA"; exit 1; }
ARGS=(--run-dir "runs/olympic_${CH}_${START}s" --video-a "$A" --start-s "$START" --duration-s "$DUR"
      --roles "${ROLES:-red,blue,referee}")
[ -n "$B" ] && ARGS+=(--video-b "$B")
python -m boxing_analytics.mesh4d.cli run "${ARGS[@]}" "${@:4}"
