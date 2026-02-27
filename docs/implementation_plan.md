# Boxing Analytics Assistant Refactor Plan

## Baseline
- Baseline tag: `baseline-pre-assistive-refactor-20260226`
- Baseline commit: `207e682`
- Active branch: `master`

## Current Entrypoints
- `main.py` (`python main.py`) for GUI/CLI launcher.
- `gui_app.py` (`main()`) for desktop UI.
- `creds.py` (`main()`) unrelated credential utility.

## Phased Delivery

### Phase 1: Tooling and Package Layout (completed)
Files added/updated:
- Add `pyproject.toml` with pinned `black`, `ruff`, `mypy`, `pytest`, `pre-commit`.
- Add `.pre-commit-config.yaml`.
- Add `.github/workflows/ci.yml`.
- Add `src/boxing_analytics/` package with required subpackages:
  - `app/`, `video/`, `calibration/`, `tracking/`, `detection/`, `scoring/`, `review/`, `eval/`
- Add initial architecture-safe modules:
  - `app/positioning.py` for mandatory decision-support messaging.
  - `video/timeline.py` and `video/rounds.py` for timestamp-first round infrastructure.
- Add initial tests under `tests/unit` and `tests/integration`.
- Add `README.md` with setup, constraints, and known limitations.

Acceptance in this phase:
- Project can be installed in editable mode.
- Lint, typecheck, and tests run in CI on the new `src` code path.
- Positioning language is encoded and test-verified.

### Phase 2: Timestamp Timebase and Bout Config
Planned file changes:
- Implement `src/boxing_analytics/video/timeline.py` decoder-backed timestamps.
- Implement `src/boxing_analytics/video/rounds.py` timeline marker segmentation.
- Add `src/boxing_analytics/video/capture.py` for PTS/monotonic timestamps.
- Update `gui_app.py` to add bout presets/custom settings and round-start alignment controls.
- Add integration test with short clip to validate FPS-invariant round boundaries.

Implemented in this step:
- Added timestamp resolver + canonical timeline rows in `src/boxing_analytics/video/timeline.py`.
- Added round-end detection helpers in `src/boxing_analytics/video/rounds.py`.
- Wired `video_processor.py` to timestamp-based round segmentation using decoder PTS with monotonic fallback.
- Added bout env config support (`VARBOX_ROUNDS_COUNT`, `VARBOX_ROUND_SECONDS`, `VARBOX_REST_SECONDS`,
  `VARBOX_WARMUP_SECONDS`, `VARBOX_ROUND_START_OFFSET_SECONDS`, `VARBOX_STOP_AT_BOUT_END`).
- Updated `gui_app.py` with bout presets/custom controls and round-start alignment picker.
- Changed manual fingerprint cancel behavior to continue with automatic tracking (no run abort).
- Added tests for timestamp resolution, round completion detection, and FPS-invariant round assignment.

Acceptance in this phase:
- Round boundaries derived from timestamps, not frame count.
- FPS override does not alter round segmentation results.

### Phase 3: Tracking and Identity
Planned file changes:
- Add new tracking stack under `src/boxing_analytics/tracking/`:
  - detector abstraction
  - multi-object tracker wrapper
  - re-identification embedding service
  - identity stability scoring and swap logging
- Deprecate color-prior assignment as primary identity path.
- Add referee handling and Fighter A/B mode.
- Add tests for occlusion, re-id, and swap logging.

Implemented in this step:
- Added `src/boxing_analytics/tracking/identity_manager.py` with motion + appearance assignment and no red/blue color priors.
- Wired `video_processor.py` to use `IdentityManager` as primary role assignment.
- Added swap and role-change logging to exported metadata.
- Added Unknown Corners mode in GUI (`Fighter A/B`) and runtime flag.
- Added unit tests for role assignment, referee candidate handling, and role-change logging.

Acceptance in this phase:
- Identity path does not depend on red/blue histogram priors.
- ID switch events are measurable and logged.

### Phase 4: Strike and Contact Pipeline
Planned file changes:
- Implement `src/boxing_analytics/detection/` modules for glove state, guard state, and contact classification.
- Implement time-based event dedup (remove FPS cooldown logic).
- Add evidence clip generation under `src/boxing_analytics/review/clip_export.py`.
- Add tests for landed_clean, landed_glancing, blocked, missed, clinch labels.

Implemented in this step:
- Added multi-stage detection modules under `src/boxing_analytics/detection/`:
  - `glove_state.py`, `guard_state.py`, `contact_classifier.py`, `pipeline.py`, `event_deduplicator.py`.
- Replaced frame-based attempt/contact suppression with time-based deduplication.
- Wired `video_processor.py` to use the new strike pipeline for RED->BLUE and BLUE->RED.
- Added evidence clip export support in `src/boxing_analytics/review/clip_export.py`.
- Scored events now carry classifier label, confidence, explainability features, evidence image, and evidence clip.

Acceptance in this phase:
- Every scored event has confidence and evidence clip.
- Head/body zones are explicit classes.

### Phase 5: Scoring Assistant and Review Workflow
Planned file changes:
- Add criteria model in `src/boxing_analytics/scoring/criteria.py`.
- Add round model gating for 10-point proposal only when prerequisites are met.
- Add review workflow modules and JSON audit log exporter.
- Update UI for timeline scrub, event filters, manual corrections, ref event confirm.

Implemented in this step:
- Added scoring assistant modules:
  - `src/boxing_analytics/scoring/criteria.py`
  - `src/boxing_analytics/scoring/assistant.py`
- Added hard scoring gate checks:
  - round alignment markers present
  - landed/blocked classifications present
  - evidence clips present for landed events
  - confirmed referee-event flags present
- Updated `video_processor.py` to:
  - parse confirmed referee events from `VARBOX_REF_EVENTS`
  - apply knockdowns/deductions/fouls
  - disable 10-point proposals unless gate passes
  - emit analytics-only mode with explicit reason when gate fails
- Added referee-event confirm controls and audit-log export in `gui_app.py`.
- Extended scorecard summary to include scoring mode and assistant criteria table.

Acceptance in this phase:
- System shows analytics-only mode when prerequisites for round proposals are missing.
- All manual changes are timestamped and audit-exportable.

### Phase 6: Calibration and Evaluation Harness
Planned file changes:
- Add OpenCV calibration tooling:
  - `src/boxing_analytics/calibration/calibrate_camera.py`
  - `src/boxing_analytics/calibration/models.py`
- Add dataset schema and runner:
  - `src/boxing_analytics/eval/schema.py`
  - `src/boxing_analytics/eval/run.py`
- Add toy dataset and regression tests.
- Quarantine experimental `boxing_var_mediapipe.py` under `experiments/` and ensure it is not imported by the production pipeline.

Implemented in this step:
- Implemented full calibration profile model IO in `src/boxing_analytics/calibration/models.py`:
  - typed profile validation
  - JSON/YAML-compatible save/load
- Implemented OpenCV chessboard calibration CLI in `src/boxing_analytics/calibration/calibrate_camera.py`:
  - image glob ingestion
  - corner detection + camera intrinsics calibration
  - reprojection error calculation
  - optional ring-plane homography import from JSON points
- Implemented evaluation schema + CLI harness:
  - `src/boxing_analytics/eval/schema.py` dataclasses and dataset validation
  - `src/boxing_analytics/eval/run.py` metrics:
    - contact precision/recall/F1
    - target-zone accuracy
    - ID switch rate
    - timing alignment error
    - evidence clip integrity
  - command supported:
    `python -m boxing_analytics.eval.run --dataset path --model-config path`
- Added toy regression fixture dataset and config under `tests/fixtures/eval/`.
- Wired calibration profile loading/status into UI + runtime metadata:
  - `gui_app.py` load/clear calibration controls
  - `video_processor.py` loads calibration profile and records status in report metadata
  - `scorecard_generator.py` includes calibration status in executive summary
- Quarantined experimental script:
  - moved `boxing_var_mediapipe.py` to `experiments/boxing_var_mediapipe.py`
  - added `experiments/README.md`

Acceptance in this phase:
- `python -m boxing_analytics.eval.run --dataset ... --model-config ...` produces metrics.
- Calibration profile can be loaded and status is visible in UI.

## Constraints Carried Into All Phases
- Decision support only; never referee replacement.
- No FPS-based cooldowns.
- No synthetic mirrored opponents.
- No authoritative 10-point display without prerequisite event support.
- No regression in testability; unit + integration tests required per phase.

### Phase 7: Expert Review Corrections and Traceability (in progress)
Implemented in this step:
- Added review-correction module in `src/boxing_analytics/review/corrections.py`:
  - parse queued manual corrections
  - apply relabel/reassign/invalidate edits by timestamp + role matching
- Added hash-chain audit utilities in `src/boxing_analytics/review/audit_log.py`:
  - deterministic event chain with `prev_hash` + `event_hash`
- Updated GUI workflow:
  - queue manual corrections (timestamp/role/relabel/reassign/invalidate)
  - export hash-chained audit JSON
  - pass corrections to pipeline via `VARBOX_MANUAL_CORRECTIONS`
- Updated processing pipeline:
  - apply manual corrections post-pass
  - recompute per-round landed counts from corrected events
  - rerun scoring assistant gating + criteria on corrected data
  - emit correction metadata and deterministic correction hash
- Updated scorecard summary:
  - manual correction count + correction hash snippet

Tests added:
- `tests/unit/test_review_corrections.py`
- `tests/unit/test_audit_log_chain.py`

### Phase 8: Timeline Review and In-App Score Preview (in progress)
Implemented in this step:
- Added timeline normalization/filtering module:
  - `src/boxing_analytics/review/timeline.py`
  - builds merged model+ref timeline and filterable event rows
- Added corrected scoring preview module:
  - `src/boxing_analytics/review/preview.py`
  - replays manual corrections + ref events and computes proposal eligibility/provisional round proposals
- Updated GUI review workflow:
  - timeline panel with filters and event selection
  - “Load Selected Into Correction” action to seed correction timestamp/role
  - “Preview Corrected Scoring” action for in-app proposal preview without full rerun
- Updated processing output persistence:
  - pipeline now writes `analysis_metadata.json` snapshot per run for review replay

Tests added:
- `tests/unit/test_review_timeline.py`
- `tests/unit/test_review_preview.py`

### Phase 9: Edit Session Persistence and Correction Queue Management (in progress)
Implemented in this step:
- Added review session persistence module:
  - `src/boxing_analytics/review/edit_session.py`
  - supports deterministic correction digest, save/load JSON sessions, and digest validation
- Expanded GUI correction workflow:
  - queued correction list panel with index-based remove/clear actions
  - save/load edit session buttons for expert review continuity
- Session payload includes:
  - input/output video references
  - metadata path
  - ref events
  - queued manual corrections
  - audit chain and final audit hash

Tests added:
- `tests/unit/test_review_edit_session.py`

### Phase 10: Runtime Progress Visibility (in progress)
Implemented in this step:
- Added progress snapshot utilities in `src/boxing_analytics/video/progress.py`.
- Updated `video_processor.py` to emit periodic frame/time/event progress updates with percentage when frame counts are known.
- Updated `gui_app.py` worker/UI to consume progress percentages and render determinate progress bar updates during recording + analysis.

Tests added:
- `tests/unit/test_video_progress.py`

### Phase 11: Operator Cancel/Interrupt Controls (in progress)
Implemented in this step:
- Added cooperative interruption support in `video_processor.py` via `should_stop` callback.
- Added cancellable worker flow in `gui_app.py`:
  - operator-triggered interruption during camera recording, frame processing, and report generation
  - explicit cancelled state handling in UI
- Updated action button behavior:
  - `Reset` becomes `Cancel Run` while processing, then returns to `Reset` on completion/cancel/error.
- Cancellation now writes partial metadata artifacts and exits cleanly without forcing PDF generation.
