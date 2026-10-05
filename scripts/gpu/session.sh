#!/usr/bin/env bash
# One command for a recorded sparring session, on a fresh GPU machine or an existing one.
#
#   git clone -b feature/mesh4d https://github.com/shem2019/varbox.git ~/work/varbox
#   cd ~/work/varbox
#   HF_TOKEN=hf_xxx bash scripts/gpu/session.sh "<google-drive-folder-link or local folder>"
#
# Video names follow the shoot checklist: r1_A.mov, r1_B.mov, r2_A.mov, ... (A and B = the two
# phones). A folder with exactly two videos is treated as one round from two phones. Each round is
# synced from the clap (planks), tracked, meshed, fused, scored, rescored with the trained strike
# classifier, and packaged under ~/work/download/<session>/.
#
# Options (environment): FPS=30 (60 for full detail, about twice as slow), SESSION=name.
# Everything is resumable: re-run the same command after any interruption.
set -euo pipefail

SRC="${1:?usage: session.sh <google-drive-folder-link | local folder>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(dirname "$REPO")"
FPS="${FPS:-30}"
SESSION="${SESSION:-session_$(date +%Y%m%d)}"
RAW="$WORK/sessions/$SESSION/raw"
PREP="$WORK/sessions/$SESSION/prepared"
OUT="$WORK/download/$SESSION"
log() { printf '\n\033[1;35m## %s\033[0m\n' "$*"; }

if [ ! -f "$WORK/env.sh" ] || [ ! -x "$WORK/venv/bin/python" ] || [ ! -d "$WORK/sam-3d-body" ]; then
  log "First run on this machine: installing everything"
  bash "$REPO/scripts/gpu/setup.sh"
fi
# shellcheck disable=SC1091
source "$WORK/env.sh"
mkdir -p "$RAW" "$PREP" "$OUT"

log "Fetching videos"
if [[ "$SRC" == http* ]]; then
  [ -n "$(ls -A "$RAW" 2>/dev/null)" ] || gdown --folder "$SRC" -O "$RAW" --remaining-ok
else
  cp -n "$SRC"/* "$RAW"/ 2>/dev/null || true
fi
mapfile -t VIDEOS < <(find "$RAW" -maxdepth 2 -type f \( -iname '*.mov' -o -iname '*.mp4' -o -iname '*.m4v' \) | sort)
[ "${#VIDEOS[@]}" -gt 0 ] || { echo "no videos found in $SRC"; exit 1; }
printf '  %s\n' "${VIDEOS[@]}"

log "Converting to constant ${FPS} fps (phones record variable frame rate)"
for v in "${VIDEOS[@]}"; do
  out="$PREP/$(basename "${v%.*}").mp4"
  [ -s "$out" ] && continue
  ffmpeg -v error -y -i "$v" -vf "fps=$FPS,scale='min(1920,iw)':-2" -c:v libx264 -preset veryfast -crf 18 \
    -c:a aac -ar 48000 -ac 1 -movflags +faststart "$out.tmp.mp4" && mv "$out.tmp.mp4" "$out"
done

# Pair videos into rounds: rN_A / rN_B, or the only two files.
declare -A ROUND_A ROUND_B
for v in "$PREP"/*.mp4; do
  name="$(basename "$v" .mp4)"
  if [[ "${name,,}" =~ ^(r[0-9]+|round[0-9]+|calib)[_-]?([ab])$ ]]; then
    key="${BASH_REMATCH[1]}"; cam="${BASH_REMATCH[2]}"
    [ "$key" = "calib" ] && continue
    if [ "$cam" = "a" ]; then ROUND_A[$key]="$v"; else ROUND_B[$key]="$v"; fi
  fi
done
if [ "${#ROUND_A[@]}" -eq 0 ]; then
  files=("$PREP"/*.mp4)
  if [ "${#files[@]}" -eq 2 ]; then ROUND_A[r1]="${files[0]}"; ROUND_B[r1]="${files[1]}";
  else for i in "${!files[@]}"; do ROUND_A[clip$((i + 1))]="${files[$i]}"; done; fi
fi

process_round() {
  local key=$1
  local run_dir="runs/$SESSION/$key" dest r
  args=(--run-dir "$run_dir" --video-a "${ROUND_A[$key]}" --start-s 0 --duration-s 0 --roles red,blue --seed-scan-s 20)
  [ -n "${ROUND_B[$key]:-}" ] && args+=(--video-b "${ROUND_B[$key]}")
  log "Round $key: $(basename "${ROUND_A[$key]}")${ROUND_B[$key]:+ + $(basename "${ROUND_B[$key]}")}"
  python -m boxing_analytics.mesh4d.cli run "${args[@]}"
  if [ -f models/varbox-videomae-cuda/best/model.safetensors ]; then
    python -m boxing_analytics.mesh4d.cli combine --run-dir "$run_dir"
  fi

  dest="$OUT/$key"; mkdir -p "$dest"
  r="$run_dir/render"
  if [ -f "$r/overlay_A_combined.mp4" ]; then
    ffmpeg -v error -y -i "$r/overlay_A.mp4" -i "$r/overlay_A_combined.mp4" -filter_complex \
      "[0:v]scale=960:-2,drawtext=text=BEFORE  mesh only:x=20:y=h-44:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6[a];\
       [1:v]scale=960:-2,drawtext=text=AFTER  mesh + VideoMAE:x=20:y=h-44:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6[b];\
       [a][b]hstack" -c:v libx264 -crf 24 -pix_fmt yuv420p -movflags +faststart "$dest/before_after.mp4"
  fi
  for f in overlay_A.mp4 overlay_B.mp4 overlay_A_combined.mp4 isolated_red.mp4 isolated_blue.mp4; do
    [ -f "$r/$f" ] && cp "$r/$f" "$dest/"
  done
  for d in viewer viewer_combined; do
    [ -d "$run_dir/$d" ] && (cd "$run_dir" && tar czf "$dest/$d.tgz" "$d")
  done
  for f in report/summary.json report/events.json report_combined/summary.json report_combined/events.json sync.json fusion.json; do
    [ -f "$run_dir/$f" ] && cp "$run_dir/$f" "$dest/$(echo "$f" | tr / _)"
  done
  cp "$run_dir"/view_*/seed_preview.jpg "$dest/" 2>/dev/null || true
  if [ -n "${VARBOX_WORKER_TOKEN:-}" ]; then
    python -m boxing_analytics.mesh4d.worker publish "$run_dir" --title "$SESSION, round ${key#r}" \
      || echo "!! publishing $key to the dashboard failed; results stay in $dest"
  fi
}

# Rounds run side by side, as many as the GPU memory and CPU cores allow.
SLOTS="${SLOTS:-$(python - <<'PY'
from boxing_analytics.mesh4d.worker import auto_slots, system_status
print(auto_slots(system_status()))
PY
)}"
log "Processing $(printf '%s\n' "${!ROUND_A[@]}" | wc -l) round(s), $SLOTS at a time"
for key in $(printf '%s\n' "${!ROUND_A[@]}" | sort -V); do
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n || true; done
  process_round "$key" > "$WORK/sessions/$SESSION/$key.log" 2>&1 &
  echo "  round $key started (log: $WORK/sessions/$SESSION/$key.log)"
done
wait
log "Done. Results in $OUT"
du -sh "$OUT"/*
echo "Download to the Mac:  scp -r -i <key.pem> $(whoami)@<this-machine-ip>:$OUT ~/Downloads/"
