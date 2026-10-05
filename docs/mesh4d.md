# 4D boxer meshes and contact analysis

`boxing_analytics.mesh4d` turns one or two fixed camera videos into a per-frame 3D body mesh for
every fighter, anchored to the ring floor, and measures glove-to-opponent contact on every frame.
It runs on a rented NVIDIA GPU; nothing here is meant to run on the Mac.

## Pipeline

| Stage | What it does | Output in the run directory |
|---|---|---|
| seed | YOLO pose + kit colour picks RED, BLUE (and the referee) in the first frame | `view_X/seeds.json`, `seed_preview.jpg` |
| masks | SAM 2.1 large propagates one mask per person through every frame, in chunks | `view_X/masks/` |
| body | SAM 3D Body (MHR mesh, hands included) per person per frame, prompted by box + mask; focal length estimated once with MoGe-2 | `view_X/body/` |
| world | floor plane from foot contacts; each person rescaled along camera rays so feet touch the floor; One Euro smoothing | `view_X/scene.npz` |
| fuse | camera B aligned to camera A from the fighters' own 3D keypoints (no checkerboard), visibility-weighted mesh average | `fused.npz`, `fusion.json` |
| contact | glove spheres vs opponent head / torso / guard / below-belt regions; punch windows from glove speed and extension | `report/frames.jsonl`, `events.json`, `summary.json` |
| render | meshes over the source video; each fighter alone on an orbiting camera | `render/*.mp4` |
| export | decimated mesh sequence + events for the browser viewer | `viewer/` |

Every stage caches its output; re-running the same command resumes. `--stages` limits a run to
some stages, for example `--stages contact,render,export` after changing contact settings.

Two cameras are synced from audio (a clap, the bell). `sync.json` holds the offset and a
confidence score; pass `--offset-s` to override.

## On the GPU machine

```bash
git clone -b feature/mesh4d https://github.com/shem2019/varbox.git ~/work/varbox
cd ~/work/varbox
hf auth login                      # token with access to facebook/sam-3d-body-dinov3
bash scripts/gpu/setup.sh
source ~/work/env.sh
bash scripts/gpu/fetch_olympic.sh  # needs ~/.kaggle/kaggle.json
bash scripts/gpu/run_olympic_4d.sh 08 60 30
python -m http.server -d runs/olympic_08_60s/viewer 8765
```

From the Mac: `ssh -L 8765:localhost:8765 <gpu-host>` and open http://localhost:8765.

## Limits

- Meshes are estimates. A single camera cannot separate "glove in front of the face" from
  "glove on the face", so the single-view contact tolerance is 9 cm against 5 cm with two views.
- Boxing gloves are modelled as 8.5 cm spheres at the hand; the mesh itself has bare hands.
- Clinches and heavy overlap lower event confidence; those windows need human review.
- Cameras must stay still for the whole window; the floor fit and focal length are per window.
- SAM 3D Body is under Meta's SAM License; the Olympic dataset is non-commercial.
- All outputs are decision support for licensed officials, never an official score.
