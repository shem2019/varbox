# Boxing Analytics Assistant

Decision-support system for boxing judging analytics. This project is **not** a referee replacement.

## Mandatory Positioning
- Assistive judging and analytics decision support only.
- Final decisions require human judges and licensed referees.
- Outputs are unverified without calibration and validation.

## Current State
This repository is under active refactor from legacy prototype modules into a testable package layout.
Phase 2 includes timestamp-based round segmentation in the processing pipeline and configurable bout parameters.
Phase 3 replaces primary red/blue prior identity assignment with motion + appearance re-identification and logs role swaps.
Phase 4 introduces a multi-stage strike classifier and time-based event deduplication with evidence clips.
Phase 5 adds scoring-assistant gating, manual referee-event confirmation, and audit log export.
Phase 6 adds camera calibration tooling, calibration-profile loading in UI/runtime metadata, and a reproducible evaluation harness with a toy regression dataset.
Phase 7 starts expert-review hardening with manual correction replay, scoring recomputation, and hash-chained audit export.
Phase 8 adds timeline filtering in review UI and in-app corrected scoring preview from saved analysis metadata.
Phase 9 adds save/load edit sessions plus correction queue remove/clear controls for repeatable expert review workflows.
Phase 10 adds visible frame/time progress tracing during processing in GUI logs and progress bar.
Phase 11 adds operator cancellation/interruption so long runs can be stopped safely with partial artifacts preserved.

## Development Setup
```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Apple Silicon Runtime
- Use Python 3.10 or newer.
- Prefer `VARBOX_BACKEND=auto` on Apple Silicon. It resolves to YOLOv8 and uses `VARBOX_YOLO_DEVICE=mps` when PyTorch MPS is available.
- The runtime now defaults to a larger YOLO inference size on Apple Silicon and disables MediaPipe segmentation unless explicitly re-enabled.
- Print the detected accelerator profile with `python -m boxing_analytics.app.cli --print-runtime-profile`.

## Quality Gates
```bash
ruff check src tests
black --check src tests
mypy src
pytest
```

## Entrypoints
- Legacy launcher: `python main.py`
- New package CLI: `python -m boxing_analytics.app.cli --print-disclaimer`
- Camera calibration CLI: `python -m boxing_analytics.calibration.calibrate_camera --help`
- Evaluation harness: `python -m boxing_analytics.eval.run --dataset tests/fixtures/eval/toy_dataset.json --model-config tests/fixtures/eval/toy_model_config.json`

## Bout Timing Controls
- `VARBOX_ROUNDS_COUNT`
- `VARBOX_ROUND_SECONDS`
- `VARBOX_REST_SECONDS`
- `VARBOX_WARMUP_SECONDS`
- `VARBOX_ROUND_START_OFFSET_SECONDS`
- `VARBOX_STOP_AT_BOUT_END`
- `VARBOX_UNKNOWN_CORNERS`
- `VARBOX_EVENT_COOLDOWN_SECONDS`
- `VARBOX_REF_EVENTS`
- `VARBOX_CALIBRATION_PROFILE`
- `VARBOX_MANUAL_CORRECTIONS`
- `VARBOX_METADATA_PATH`

## Identity Stability (Two-Hypothesis Viterbi Smoothing)
- The tracker always evaluates both mappings: `H0` (current RED/BLUE) and `H1` (swapped).
- Each role keeps an object file (Kalman motion state + appearance/pose EMA + last seen timestamp).
- Occlusions use object permanence and coasting instead of single-frame swap decisions.
- Appearance influence is automatically suppressed during overlap/keypoint collapse.
- Output identity is delayed-commitment Viterbi state over a rolling window, not per-frame greedy.
- Any identity hypothesis change is logged with `logp(H0/H1)`, delta, occlusion score, and dominant evidence term.

### New Identity Env Vars (with defaults)
- `VARBOX_ID_VITERBI_WINDOW=25`
- `VARBOX_ID_SWITCH_PENALTY=3.0` (deprecated fallback)
- `VARBOX_ID_BASE_SWITCH_PENALTY=4.0`
- `VARBOX_ID_OCCLUSION_SWITCH_BOOST=8.0`
- `VARBOX_ID_W_MOTION=1.2`
- `VARBOX_ID_W_APPEAR=0.5`
- `VARBOX_ID_W_POSE=0.3`
- `VARBOX_ID_SIGMA_MOTION_PX=45.0`
- `VARBOX_ID_IOU_OCCLUSION_FULL=0.25`
- `VARBOX_ID_OCCLUSION_APP_SUPPRESS=0.9`
- `VARBOX_ID_UPDATE_MAX_OCCLUSION=0.08`
- `VARBOX_ID_OCCLUSION_UPDATE_MAX=0.15` (deprecated fallback)
- `VARBOX_ID_MIN_DET_CONF=0.4`
- `VARBOX_ID_MIN_KP_VIS=0.60`
- `VARBOX_ID_MAX_COAST_FRAMES=60`
- `VARBOX_ID_MISS_PENALTY=8.0`
- `VARBOX_ID_FLOW_MAX_FRAMES=12`
- `VARBOX_ID_COMMIT_P_THRESHOLD=0.98`
- `VARBOX_ID_COMMIT_SECONDS=1.0`
- `VARBOX_ID_COMMIT_MAX_OCCLUSION=0.10`

### Stitcher Env Vars (with defaults)
- `VARBOX_STITCH_MAX_DIST_PX=80.0`
- `VARBOX_STITCH_EMBED_MAX=0.35`
- `VARBOX_STITCH_W_MOTION=1.0`
- `VARBOX_STITCH_W_APPEAR=0.6`
- `VARBOX_STITCH_W_POSE=0.2`

### Deprecated Identity Vars (still accepted)
- `VARBOX_ID_SWITCH_MARGIN`
- `VARBOX_ID_MIN_FRAMES_BEFORE_SWITCH`
- `VARBOX_ID_CLINCH_IOU_FREEZE`
- `VARBOX_ID_CLINCH_FREEZE_FRAMES`

## Evaluation Dataset Schema
`tests/fixtures/eval/toy_dataset.json` demonstrates the canonical fields:
- `videos[].video_id`
- `videos[].samples[]` with `timestamp_s`, `fighter_id`, `glove_box`/`glove_keypoints`, `ground_truth_label`, `predicted_label`, `ground_truth_zone`, `predicted_zone`, `ground_truth_track_id`, `predicted_track_id`, `evidence_clip`
- `videos[].round_markers_gt` and `videos[].round_markers_pred`
- `videos[].ref_events`

## Known Limitations
- Core production pipeline is still rooted in legacy modules at repository root.
- Contact classification remains heuristic despite the structured multi-stage pipeline.
- Calibration and evaluation are decision-support controls; they do not certify official scoring accuracy without external validation.
