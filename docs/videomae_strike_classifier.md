# Offline VideoMAE and SAM 2.1 for VAR Box

## Scope and safety

VAR Box can use a local VideoMAE model to classify temporal strike candidates and a local
SAM 2.1 model to improve RED/BLUE mask continuity. Neither model produces an official
fight score. The existing scoring gate remains authoritative and blocks provisional scoring
for smoke-trained models, missing metadata, excessive abstention, missing evidence, or
cancelled analysis.

The Olympic Boxing Punch Classification Video Dataset and the selected Kinetics-finetuned
VideoMAE checkpoint are restricted to non-commercial use. This configuration is intended
for the repository owner's stated non-commercial use.

## Architecture

```text
manual RED/BLUE seeds
  -> sampled SAM 2.1 mask propagation
  -> interpolated mask boxes
  -> unique matching to YOLO pose tracks

YOLO pose + stable role identity
  -> permissive punch candidate
  -> 0.6 second interaction crop
  -> local VideoMAE probabilities
  -> deterministic pose/identity fusion
  -> accepted or abstained event
  -> evidence, review, and scoring gate
```

SAM does not infer corner designation. Manual seed boxes establish RED and BLUE. SAM masks
then provide a persistent spatial prior while YOLO continues to supply pose keypoints and raw
track IDs. HMM/ReID remains the automatic fallback when SAM is unavailable.

## Local paths used in this repository

```text
Dataset:
/Users/shemking/Downloads/Olympic Boxing Punch Classification Video Dataset

Base VideoMAE:
models/videomae-base-finetuned-kinetics

Smoke model:
models/varbox-videomae-smoke/best

Current development baseline:
models/varbox-videomae-development-current/best

SAM 2.1 tiny:
models/sam2.1/sam2.1_hiera_tiny.pt
```

`models/`, `artifacts/`, and `training_cache/` are ignored by Git.

## Installation on Apple Silicon

Use the repository virtual environment. SAM's CUDA extension must be disabled:

```bash
source .venv/bin/activate
SAM2_BUILD_CUDA=0 python -m pip install -e '.[dev,runtime,ml]'
```

The checked installation uses Python 3.13, PyTorch 2.10, torchvision 0.25, and MPS.
The installed SAM package is pinned to commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4`; its CUDA extension is not built.

## Dataset audit

```bash
export VARBOX_OLYMPIC_DATASET_DIR="/Users/shemking/Downloads/Olympic Boxing Punch Classification Video Dataset"

python -m boxing_analytics.training.dataset_audit \
  --dataset-dir "$VARBOX_OLYMPIC_DATASET_DIR" \
  --output-dir artifacts/olympic_dataset_audit
```

The real dataset contains CVAT track JSON with Polish labels and per-frame punch rectangles.
The adapter groups synchronized camera files by bout-segment prefix. Camera-2 and camera-4
views from the same segment cannot cross splits.

The audit writes:

- `dataset_summary.json`
- `class_counts.csv`
- `source_files.csv`
- `annotation_schema.json`
- `invalid_annotations.json`
- `generated_manifest.jsonl`
- `split_summary.json`
- `sample_clips/`
- `sample_contact_sheet/contact_sheet.jpg`

Generated `no_punch` rows are deterministic weak labels sampled at least one second away
from annotated punches. They are not referee-verified.

## MPS smoke training

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

python -m boxing_analytics.training.train_videomae \
  --manifest artifacts/olympic_dataset_audit/generated_manifest.jsonl \
  --checkpoint models/videomae-base-finetuned-kinetics \
  --output-dir models/varbox-videomae-smoke \
  --label-schema five_class \
  --epochs 1 \
  --batch-size 1 \
  --device mps \
  --gradient-accumulation 2 \
  --freeze-backbone \
  --max-train-samples 10 \
  --max-validation-samples 5
```

Smoke models prove the pipeline, not classification quality. They are marked
`smoke_trained=1`, and the scoring gate rejects them.

## Full local training

Start with head-only training to establish a baseline:

```bash
python -m boxing_analytics.training.train_videomae \
  --manifest artifacts/olympic_dataset_audit/generated_manifest.jsonl \
  --checkpoint models/videomae-base-finetuned-kinetics \
  --output-dir models/varbox-videomae-head-baseline \
  --label-schema five_class \
  --epochs 10 \
  --batch-size 1 \
  --device mps \
  --learning-rate 0.001 \
  --gradient-accumulation 4 \
  --freeze-backbone
```

The training command also accepts the dataset directly and creates the audit/manifest when
`--manifest` is omitted:

```bash
python -m boxing_analytics.training.train_videomae \
  --dataset-dir "$VARBOX_OLYMPIC_DATASET_DIR" \
  --audit-output-dir artifacts/olympic_dataset_audit \
  --output-dir models/varbox-videomae-head-baseline \
  --device mps --batch-size auto --epochs 10 --freeze-backbone
```

For full encoder fine-tuning, omit `--freeze-backbone` and lower the learning rate, normally
to `5e-5` or less. This is substantially slower and uses more unified memory.

Training exports the best macro-F1 checkpoint, processor configuration, label mapping,
manifest digest, Git commit, per-epoch metrics, confusion matrix, device, and smoke status.
Decoded interaction crops and their timestamps/crop metadata are cached losslessly under
`training_cache/videomae_clips/`. Set `VARBOX_VIDEOMAE_CLIP_CACHE_DIR` or pass
`--clip-cache-dir` to move or disable that location. A full cached manifest can consume
roughly 5–8 GB, depending on visual complexity.

CUDA example:

```bash
python -m boxing_analytics.training.train_videomae \
  --manifest artifacts/olympic_dataset_audit/generated_manifest.jsonl \
  --checkpoint models/videomae-base-finetuned-kinetics \
  --output-dir models/varbox-videomae-cuda \
  --device cuda --batch-size 4 --epochs 10
```

CPU pipeline check:

```bash
python -m boxing_analytics.training.train_videomae \
  --manifest artifacts/olympic_dataset_audit/generated_manifest.jsonl \
  --checkpoint models/videomae-base-finetuned-kinetics \
  --output-dir models/varbox-videomae-cpu-smoke \
  --device cpu --batch-size 1 --epochs 1 --freeze-backbone \
  --max-train-samples 5 --max-validation-samples 5
```

## Evaluation

```bash
python -m boxing_analytics.training.evaluate_videomae \
  --manifest artifacts/olympic_dataset_audit/generated_manifest.jsonl \
  --model-dir models/varbox-videomae-head-baseline/best \
  --split test \
  --device mps \
  --output artifacts/videomae_test_metrics.json
```

The evaluator writes JSON plus a Markdown summary. Important metrics are macro F1, balanced
accuracy, per-class precision/recall/F1, confusion matrix, and false-punch rate on derived
negatives. The existing end-to-end evaluation harness also reports candidate recall,
abstention, non-abstained accuracy, identity uncertainty, evidence integrity, timing error,
duplicate rate, and per-bout metrics when those optional fields are present.

## Runtime

```bash
export VARBOX_STRIKE_BACKEND=hybrid_videomae
export VARBOX_VIDEOMAE_MODEL_DIR="$PWD/models/varbox-videomae-development-current/best"
export VARBOX_VIDEOMAE_DEVICE=mps
export VARBOX_IDENTITY_BACKEND=sam2
export VARBOX_SAM2_CHECKPOINT="$PWD/models/sam2.1/sam2.1_hiera_tiny.pt"
export VARBOX_SAM2_DEVICE=mps
python main.py
```

Supported strike backends:

- `local`
- `roboflow`
- `hybrid`
- `videomae`
- `hybrid_videomae`

`local`, `videomae`, and `hybrid_videomae` can run fully offline. Roboflow modes cannot.
Recorded-file analysis reads the complete pre/post-event window directly from the source, so
classification is naturally delayed until post-contact evidence exists. The bounded
`TemporalClipBuffer` stores frames, timestamps, boxes, identities, keypoints, confidence, and
ring ROI for a future direct live-stream path; the current camera workflow records locally
before analysis.

## SAM performance controls

SAM 2.1 Tiny runs on MPS without the optional CUDA extension. It is not real-time on this
M2. VAR Box therefore segments sampled frames in bounded chunks, interpolates mask boxes,
and matches them against every YOLO pose frame.

```text
VARBOX_SAM2_STRIDE=10
VARBOX_SAM2_CHUNK_SAMPLES=120
VARBOX_SAM2_MAX_SIDE=768
VARBOX_SAM2_ISOLATE_MPS=1
```

Lower stride improves mask fidelity and increases runtime. Smaller chunks reduce peak memory.
`VARBOX_SAM2_MAX_SIDE` also sets SAM's real square inference resolution (rounded to a multiple
of 16 and capped at 1024), rather than merely resizing the temporary JPEGs.
The generated identity track is cached under `training_cache/sam2/` and reused when the video,
seeds, checkpoint, and configuration are unchanged.

Recorded-video processing is a two-pass workflow. SAM first precomputes the sampled fighter
masks for the complete video, with progress and an estimated time remaining; the normal
YOLO/pose/strike analysis begins afterward. The first run is therefore slower. Later runs reuse
the cache when the video, seeds, checkpoint, and SAM settings are unchanged.

On Apple MPS, VAR Box keeps SAM's video-memory tensors in float32 so they match the model
weights. This avoids the native Metal matrix-multiplication abort caused by SAM 2.1's
CUDA-oriented bfloat16 memory compression. MPS propagation also runs in an isolated local
worker by default: if Metal encounters another native failure, the main analysis process stays
alive and uses HMM/ReID identity tracking for that run. Set `VARBOX_SAM2_DEVICE=cpu` for a
slower SAM-only compatibility path, or `VARBOX_SAM2_ISOLATE_MPS=0` only when debugging.

## Offline enforcement

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

The VideoMAE runtime always uses `local_files_only=True`. For a fully isolated installation,
prepare a wheelhouse for Python dependencies and install SAM 2 from a locally built wheel.
The current repository virtual environment and model directories are already sufficient for
offline inference and training on this Mac; reconnecting is only needed to rebuild the
environment unless a wheelhouse is prepared.

## Troubleshooting

- **Model directory missing:** select a directory containing `config.json`, processor files,
  weights, and `varbox_model_metadata.json`.
- **SAM unavailable:** confirm the checkpoint exists and SAM was installed with
  `SAM2_BUILD_CUDA=0`; VAR Box will retain the HMM/ReID fallback.
- **MPS memory pressure:** increase `VARBOX_SAM2_STRIDE`, reduce
  `VARBOX_SAM2_MAX_SIDE`, keep VideoMAE batch size at 1, and use gradient accumulation.
- **Slow first run:** SAM masks and VideoMAE crops are cached; unchanged reruns are faster.
- **Offline load failure:** set the three offline environment variables and confirm all model
  files are inside `models/`.

## Limitations

- The bundled current development baseline is deliberately not production-ready: only the
  classifier head was trained on a limited balanced subset.
- The dataset is strongly imbalanced toward landed head punches.
- Derived negatives may contain unannotated actions.
- A dataset-derived test result does not prove generalization to broadcast or phone footage.
- SAM masks can drift during long occlusion or scene cuts; chunk re-prompting limits but does
  not eliminate drift.
- If output orientation rotates the source frame, SAM identity safely falls back to HMM/ReID;
  VideoMAE crops apply the configured rotation before using pose boxes.
- MPS inference is slower than CUDA.
- Human review and qualified judge/referee confirmation remain mandatory.

## Licence notes

SAM 2 code and checkpoints are distributed by Meta under Apache 2.0. The selected
`MCG-NJU/videomae-base-finetuned-kinetics` Hugging Face checkpoint is published as
CC-BY-NC-4.0, and the Olympic boxing dataset is non-commercial. Keep this local
configuration non-commercial unless separate rights are obtained.
