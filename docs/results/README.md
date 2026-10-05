# VAR Box results and runbook, 5 October 2026

The full 4D pipeline now runs end to end on rented GPUs and publishes to
https://varbox.guestpassvms.com. This page records where everything lives and what the first day
measured, so the next session starts from here.

## Where everything lives

| What | Where |
|---|---|
| Code (pipeline, website, GPU scripts) | GitHub `shem2019/varbox`, branches `master` and `feature/mesh4d` (identical) |
| Website | https://varbox.guestpassvms.com, deployed by `.github/workflows/deploy-web.yml` on push to `master` |
| Trained strike classifier | Hugging Face `shemking/varbox-videomae-strike` (private), epoch 14, validation macro F1 0.710 |
| SAM 3D Body | Hugging Face `facebook/sam-3d-body-dinov3` (gated; account `shemking` approved) |
| Olympic dataset | Kaggle `piotrstefaskiue/olympic-boxing-punch-classification-video-dataset` |
| Published analyses | the dashboard (server storage on the VPS) |

## Next session on a fresh GPU

1. Rent an NVIDIA GPU with 48 GB and as many CPU cores as offered (12+). The pipeline runs both
   cameras, several mask chunks and several jobs at once, sized from the cores and GPU memory.
2. Open the dashboard's **GPU** page and run its one command on the machine (Hugging Face token
   with read access). Setup takes about 25 minutes and the GPU then appears as online.
3. Record with the shoot checklist: red and blue kit, one plank crack per round, files `r1_A`,
   `r1_B`, and so on.
4. Upload both videos on the dashboard's **Import** page, or put them in a Google Drive folder and
   run `bash scripts/gpu/session.sh "<folder link>"` on the GPU.

## Accuracy on held-out Olympic clips (157 labelled punches)

Four 30-second clips from chapters the strike model never trained on, scored against the dataset's
hand labels. Details in `heldout_accuracy_2026-10-05.md`.

| | Value |
|---|---|
| Labelled punches found | 66% |
| Hand correct on found punches | 88% |
| Outcome (landed, blocked, missed), geometry only | 32% |
| Outcome with the strike model | 53% |

The labels cover one camera per punch, so unlabelled detections may still be real punches;
precision is unmeasured. The strike model alone scores macro F1 0.57 on the held-out test split
(`videomae_test_metrics.md`, training curve in `videomae_training_epochs.md`).

## Speed for one 30-second clip

| Pipeline | Time |
|---|---|
| Morning, first version | about 22 min |
| Batched meshes and parallel renders | 13.5 min |
| Plus parallel mask chunks, two clips side by side (6-core A6000) | about 8.5 min per clip |

Masks now run as independent 8-second chunks, each seeded by a colour-confirmed detection of both
boxers. On a clip where both boxers wore red singlets, the earlier sequential tracker drifted onto
a spectator and then the referee; the chunked tracker stayed on both boxers.

## Decisions taken

- SAM 2.1 large stays the mask model: base-plus was only 15% faster with no accuracy gain
  (`sam2_large_vs_base_plus.md`).
- Each pixel belongs to one boxer in SAM 2, and a track that collapses onto the other boxer is
  marked hidden for those frames instead of building a second body from the wrong person.

## Next steps

1. A real phone upload through the website, end to end.
2. Label 50 to 100 punches from VAR Box's own sparring footage and tune punch detection and outcome
   for those camera positions.
3. Test whether the one-boxer-per-pixel setting shrinks the boxes of partly hidden boxers enough to
   cost mesh quality.
