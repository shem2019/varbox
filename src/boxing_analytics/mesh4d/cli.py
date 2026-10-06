"""End-to-end 4D boxing analysis: masks, meshes, floor-anchored 4D, fusion, contact, outputs.

Every stage caches its results in the run directory, so a crashed or interrupted run resumes
where it stopped. Example (one camera, 30 s from the 60 s mark):

    python -m boxing_analytics.mesh4d.cli run --run-dir runs/olympic_08 \
        --video-a data/GH088416.mp4 --start-s 60 --duration-s 30

Two cameras (offset found from audio unless --offset-s is given):

    python -m boxing_analytics.mesh4d.cli run --run-dir runs/olympic_08 \
        --video-a data/GH088416.mp4 --video-b data/GH089681.mp4 --start-s 60 --duration-s 30
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from boxing_analytics.mesh4d.jsonsafe import safe_dumps
from boxing_analytics.mesh4d.video_io import VideoInfo, probe, read_frame

STAGES = ("seed", "masks", "body", "world", "fuse", "contact", "render", "export")


def _log_to(path: Path):  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def _window(info: VideoInfo, start_s: float, duration_s: float) -> tuple[int, int]:
    start = max(0, int(round(start_s * info.fps)))
    stop = (
        info.frame_count
        if duration_s <= 0
        else min(info.frame_count, start + int(round(duration_s * info.fps)))
    )
    if stop - start < 2:
        raise ValueError(f"empty window: start {start}, stop {stop} of {info.frame_count}")
    return start, stop


def _occlusion_series(
    mask_dir: Path, start: int, stop: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-role occlusion and the red/blue mask overlap for every frame of the window."""
    from boxing_analytics.mesh4d.masklets import MaskStore

    store = MaskStore(mask_dir)
    out = {r: np.ones(stop - start, dtype=np.float32) for r in store.roles}
    pair = np.zeros(stop - start, dtype=np.float32)
    for i, frame in enumerate(range(start, stop)):
        for role, value in store.occlusion(frame).items():
            out[role][i] = value
        pair[i] = store.pair_iou(frame)
    return out, pair


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Two runs on one directory overwrite each other's videos; hold an exclusive lock.
    # Per-camera child processes run under their parent's lock.
    lock = (run_dir / ".lock").open("a")
    if not args.only_view:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another run is already using {run_dir}; not starting a second one")
            return 75
    log = _log_to(run_dir / "run.log")
    only = set(args.stages.split(",")) if args.stages else set(STAGES)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    views: dict[str, dict[str, Any]] = {}
    info_a = probe(args.video_a)
    start_a, stop_a = _window(info_a, args.start_s, args.duration_s)
    views["A"] = {"info": info_a, "start": start_a, "stop": stop_a, "seeds": args.seed_a}
    sync_meta: dict[str, Any] | None = None
    if args.video_b:
        from boxing_analytics.mesh4d.sync import SyncResult, best_offset, map_frame
        from boxing_analytics.mesh4d.video_io import extract_audio

        info_b = probe(args.video_b)
        sync_path = run_dir / "sync.json"
        if args.offset_s is not None:
            sync = SyncResult(float(args.offset_s), float("inf"), "manual")
        elif sync_path.exists():
            payload = json.loads(sync_path.read_text(encoding="utf-8"))
            sync = SyncResult(payload["offset_s"], payload["confidence"], payload["method"])
        else:
            audio_a = extract_audio(args.video_a, sample_rate=16000)
            audio_b = extract_audio(args.video_b, sample_rate=16000)
            if audio_a is None or audio_b is None:
                raise RuntimeError("these videos carry no audio to sync from; pass --offset-s")
            sync = best_offset(audio_a, audio_b, 16000)
        sync_path.write_text(safe_dumps(sync.to_dict(), indent=2), encoding="utf-8")
        log(f"sync: B = A + {sync.offset_s:.3f}s ({sync.method}, confidence {sync.confidence:.1f})")
        if sync.confidence < 6:
            log("sync: low confidence; check run_dir/sync.json or pass --offset-s")
        # Phones start recording at different moments: keep the stretch both cameras saw.
        first_a = int(np.ceil(max(0.0, -sync.offset_s) * info_a.fps))
        last_a = int(np.floor((info_b.frame_count / info_b.fps - sync.offset_s) * info_a.fps))
        overlap_start, overlap_stop = max(start_a, first_a), min(stop_a, last_a)
        if overlap_stop - overlap_start < int(2 * info_a.fps):
            log("sync: the two cameras share under two seconds of this window; camera A alone")
        else:
            if (overlap_start, overlap_stop) != (start_a, stop_a):
                log(
                    f"sync: window trimmed to the overlap of both cameras, frames "
                    f"{overlap_start}-{overlap_stop} of camera A"
                )
                start_a, stop_a = overlap_start, overlap_stop
                views["A"].update(start=start_a, stop=stop_a)
            start_b = max(0, map_frame(start_a, info_a.fps, info_b.fps, sync.offset_s))
            stop_b = min(
                info_b.frame_count, map_frame(stop_a, info_a.fps, info_b.fps, sync.offset_s)
            )
            views["B"] = {"info": info_b, "start": start_b, "stop": stop_b, "seeds": args.seed_b}
            sync_meta = {**sync.to_dict(), "start_b": start_b, "stop_b": stop_b}

    run_meta = {
        "video_a": args.video_a,
        "video_b": args.video_b,
        "window_a": [start_a, stop_a],
        "sync": sync_meta,
        "roles": roles,
        "sam3d_repo": args.hf_repo,
        "inference_type": args.inference_type,
    }
    if not args.only_view:
        (run_dir / "run.json").write_text(safe_dumps(run_meta, indent=2), encoding="utf-8")

    # Two cameras: the heavy per-camera stages run as two processes at once, one per camera.
    per_camera = only & {"seed", "masks", "body"}
    if args.only_view:
        views = {k: v for k, v in views.items() if k == args.only_view}
    elif len(views) == 2 and per_camera and not args.serial:
        log(f"cameras A and B processed in parallel ({', '.join(sorted(per_camera))})")
        base = list(sys.argv[1:])
        if "--stages" in base:
            i = base.index("--stages")
            del base[i : i + 2]
        children = [
            subprocess.Popen(
                [sys.executable, "-m", "boxing_analytics.mesh4d.cli", *base]
                + ["--stages", ",".join(sorted(per_camera)), "--only-view", name]
            )
            for name in views
        ]
        codes = [c.wait() for c in children]
        if any(codes):
            raise RuntimeError(f"per-camera processing failed (exit codes {codes})")
        only -= per_camera

    # ---------------------------------------------------------------- seed + masks
    from boxing_analytics.mesh4d.masklets import MaskConfig, track_masks, track_masks_parallel
    from boxing_analytics.mesh4d.seeding import (
        Seed,
        draw_seeds,
        has_fighters,
        load_seeds,
        parse_manual_seeds,
        save_seeds,
        scan_for_seed,
    )

    mask_cfg = MaskConfig(
        checkpoint=args.sam2_checkpoint,
        model_config=args.sam2_config,
        device=args.device,
        max_side=args.mask_max_side,
        workers=args.mask_workers,
        share=2 if args.only_view else 1,
        yolo_model=args.yolo_model,
    )
    for name, view in views.items():
        vdir = run_dir / f"view_{name}"
        vdir.mkdir(exist_ok=True)
        seeds_path = vdir / "seeds.json"
        info: VideoInfo = view["info"]
        if "seed" in only and not seeds_path.exists():
            scan_stop = min(view["stop"], view["start"] + int(args.seed_scan_s * info.fps))
            seeds, quality = scan_for_seed(
                info.path,
                view["start"],
                scan_stop,
                model_path=args.yolo_model,
                device=args.device,
                with_referee="referee" in roles,
            )
            if not has_fighters(seeds) and scan_stop < view["stop"]:
                # Nobody clean in the first seconds: keep looking through the whole window.
                log(f"seed {name}: scanning the full window for a clean view of both boxers")
                seeds, quality = scan_for_seed(
                    info.path,
                    scan_stop,
                    view["stop"],
                    model_path=args.yolo_model,
                    device=args.device,
                    with_referee="referee" in roles,
                    step=max(5, int(info.fps / 2)),
                )
            seed_frame = next(iter(seeds.values())).frame_index if seeds else view["start"]
            frame = read_frame(info.path, seed_frame)
            log(f"seed {name}: best frame {seed_frame} (quality {quality:.2f})")
            for role, box in parse_manual_seeds(view["seeds"]).items():
                seeds[role] = Seed(role, seed_frame, box, "manual", {})
            seeds = {r: s for r, s in seeds.items() if r in roles}
            if "referee" in roles and "referee" not in seeds:
                log(f"seed {name}: referee not found cleanly; tracking the two boxers only")
            missing = [r for r in roles if r not in seeds and r != "referee"]
            if missing:
                raise RuntimeError(
                    f"view {name}: could not seed {missing}; "
                    f"pass --seed-{name.lower()} role=x1,y1,x2,y2"
                )
            save_seeds(seeds_path, seeds)
            cv2.imwrite(str(vdir / "seed_preview.jpg"), draw_seeds(frame, seeds))
            described = ", ".join(
                f"{r}={tuple(round(v) for v in s.box)} ({s.source})" for r, s in seeds.items()
            )
            log(f"seed {name}: {described}")
        if seeds_path.exists():
            # Mask tracking can only reach back within its first chunk; footage before a late
            # seed frame is pre-round anyway, so the window starts at the seed.
            first_seed = min(s.frame_index for s in load_seeds(seeds_path).values())
            # Parallel chunks find both boxers themselves, so only the sequential tracker skips ahead.
            if args.mask_workers == 1 and first_seed - view["start"] > 200:
                log(f"seed {name}: window now starts at the seed frame {first_seed}")
                view["start"] = first_seed
                if name == "A":
                    run_meta["window_a"] = [first_seed, view["stop"]]
                    (run_dir / "run.json").write_text(
                        safe_dumps(run_meta, indent=2), encoding="utf-8"
                    )
        if "masks" in only:
            tracker = track_masks if args.mask_workers == 1 else track_masks_parallel
            tracker(
                info,
                view["start"],
                view["stop"],
                load_seeds(seeds_path),
                vdir / "masks",
                mask_cfg,
                log,
            )

    # ---------------------------------------------------------------- body meshes
    from boxing_analytics.mesh4d.body import BodyConfig, BodyRunner, load_body
    from boxing_analytics.mesh4d.masklets import MaskStore

    if "body" in only:
        runner: BodyRunner | None = None
        for name, view in views.items():
            vdir = run_dir / f"view_{name}"
            if (vdir / "body" / "meta.json").exists():
                log(f"body {name}: cached")
                continue
            if runner is None:
                runner = BodyRunner(
                    BodyConfig(
                        repo_dir=args.sam3d_dir,
                        hf_repo_id=args.hf_repo,
                        device=args.device,
                        inference_type=args.inference_type,
                        use_mask=args.mask_prompt,
                    ),
                    log,
                )
            runner.reset_camera()
            runner.run(
                view["info"], view["start"], view["stop"], MaskStore(vdir / "masks"), vdir / "body"
            )

    if args.only_view:
        return 0

    # ---------------------------------------------------------------- floor-anchored 4D per view
    from boxing_analytics.mesh4d.reconstruct import (
        ViewScene,
        align_views,
        build_view_scene,
        fuse_scenes,
    )

    scenes: dict[str, ViewScene] = {}
    for name, view in views.items():
        vdir = run_dir / f"view_{name}"
        path = vdir / "scene.npz"
        if "world" in only and not path.exists():
            body = load_body(vdir / "body", view["start"], view["stop"])
            occ, pair_iou = _occlusion_series(vdir / "masks", view["start"], view["stop"])
            scene = build_view_scene(body, occ, view["info"].fps, log, pair_iou=pair_iou)
            scene.save(path)
        if path.exists():
            scenes[name] = ViewScene.load(path)

    # ---------------------------------------------------------------- two-view fusion
    final = scenes.get("A")
    n_views = 1
    b_index = None
    if "B" in scenes and "A" in scenes:
        from boxing_analytics.mesh4d.sync import map_frame

        a, b = scenes["A"], scenes["B"]
        fps_a, fps_b = views["A"]["info"].fps, views["B"]["info"].fps
        offset = float(sync_meta["offset_s"])  # type: ignore[index]
        b_index = np.array(
            [map_frame(int(f), fps_a, fps_b, offset) - views["B"]["start"] for f in a.frames]
        )
        b_index[(b_index < 0) | (b_index >= b.frames.shape[0])] = -1
        fused_path = run_dir / "fused.npz"
        if "fuse" in only and not fused_path.exists():
            try:
                fusion = align_views(a, b, b_index, log)
                (run_dir / "fusion.json").write_text(
                    safe_dumps(fusion.to_dict(), indent=2), encoding="utf-8"
                )
                if fusion.median_residual_m > 0.25:
                    log(
                        f"fuse: residual {fusion.median_residual_m:.2f} m is too high; "
                        "keeping camera A alone"
                    )
                else:
                    fuse_scenes(a, b, b_index, fusion).save(fused_path)
            except RuntimeError as exc:
                log(f"fuse: skipped ({exc})")
        if fused_path.exists():
            final = ViewScene.load(fused_path)
            n_views = 2
    if final is None:
        log("no scene available yet; run the earlier stages first")
        return 1

    # ---------------------------------------------------------------- contact + reports
    from boxing_analytics.mesh4d.analysis import analyse, write_reports

    report_dir = run_dir / "report"
    contacts = None
    events: list[dict[str, Any]] = []
    if "contact" in only or "render" in only or "export" in only:
        contacts, punch_events, _ = analyse(final, n_views, log=log)
        events = [e.to_dict() for e in punch_events]
        if "contact" in only:
            summary = write_reports(
                report_dir, final, contacts, punch_events, n_views=n_views, run_meta=run_meta
            )
            log(f"summary: {safe_dumps(summary['tally'])} -> {summary['suggestion'].get('lean')}")

    # ---------------------------------------------------------------- renders + viewer
    if "render" in only:
        from concurrent.futures import ThreadPoolExecutor

        from boxing_analytics.mesh4d.render import render_isolated, render_overlay

        render_dir = run_dir / "render"
        render_dir.mkdir(exist_ok=True)
        tasks: list[tuple[Any, ...]] = [
            (render_overlay, final, views["A"]["info"].path, render_dir / "overlay_A.mp4", events)
        ]
        if "B" in scenes:
            from boxing_analytics.mesh4d.sync import map_frame

            events_b = []
            for e in events:
                e2 = dict(e)
                for k in ("start_frame", "peak_frame", "end_frame", "contact_frame"):
                    if e2.get(k) is not None:
                        e2[k] = map_frame(
                            int(e2[k]),
                            views["A"]["info"].fps,
                            views["B"]["info"].fps,
                            float(sync_meta["offset_s"]),  # type: ignore[index]
                        )
                events_b.append(e2)
            tasks.append(
                (
                    render_overlay,
                    scenes["B"],
                    views["B"]["info"].path,
                    render_dir / "overlay_B.mp4",
                    events_b,
                )
            )
        for role in final.roles:
            tasks.append((render_isolated, final, role, render_dir / f"isolated_{role}.mp4"))
        # Every video renders at once: each is mostly decoding, splatting and encoding.
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(fn, *a, device=args.device, log=log) for fn, *a in tasks]
            for future in futures:
                log(f"render: {future.result()}")

    if "export" in only:
        from boxing_analytics.mesh4d.export import export_viewer

        summary_payload = None
        if (report_dir / "summary.json").exists():
            summary_payload = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        out = export_viewer(
            final,
            contacts,
            events,
            run_dir / "viewer",
            max_frames=args.viewer_max_frames,
            summary=summary_payload,
        )
        log(f"viewer: {out} (serve with: python -m http.server -d {out} 8765)")
    log("done")
    return 0


def _final_scene(run_dir: Path):  # type: ignore[no-untyped-def]
    from boxing_analytics.mesh4d.reconstruct import ViewScene

    fused = run_dir / "fused.npz"
    if fused.exists():
        return ViewScene.load(fused), 2
    return ViewScene.load(run_dir / "view_A" / "scene.npz"), 1


def combine(args: argparse.Namespace) -> int:
    """Rescore a finished run's punch events with VideoMAE; writes report_combined/ and renders."""
    from boxing_analytics.mesh4d.analysis import DISCLAIMER, analyse
    from boxing_analytics.mesh4d.combine import VideoMAEScorer, combine_events
    from boxing_analytics.mesh4d.contact import ContactConfig
    from boxing_analytics.mesh4d.export import export_viewer
    from boxing_analytics.mesh4d.render import render_overlay

    run_dir = Path(args.run_dir)
    log = _log_to(run_dir / "run.log")
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    scene, n_views = _final_scene(run_dir)
    events = json.loads((run_dir / "report" / "events.json").read_text(encoding="utf-8"))
    scorer = VideoMAEScorer(args.model_dir, args.device)
    tol = ContactConfig.for_views(n_views).contact_tol_m
    combined = combine_events(
        scene,
        events,
        meta["video_a"],
        scorer,
        mesh_weight=args.mesh_weight,
        contact_tol_m=tol,
        log=log,
    )
    out = run_dir / "report_combined"
    out.mkdir(exist_ok=True)
    (out / "events.json").write_text(safe_dumps(combined, indent=2), encoding="utf-8")
    fighters = [r for r in ("red", "blue") if r in scene.roles]
    tally = {
        r: {"thrown": 0, "landed_head": 0, "landed_torso": 0, "blocked": 0, "missed": 0}
        for r in fighters
    }
    for e in combined:
        row = tally[e["attacker"]]
        row["thrown"] += 1
        key = f"landed_{e['target']}" if e["outcome"] == "landed" else e["outcome"]
        row[key] = row.get(key, 0) + 1
    summary = {
        "disclaimer": DISCLAIMER,
        "method": "mesh events rescored by VideoMAE",
        "model_dir": args.model_dir,
        "mesh_weight": args.mesh_weight,
        "tally": tally,
    }
    (out / "summary.json").write_text(safe_dumps(summary, indent=2), encoding="utf-8")
    log(f"combine: tally {safe_dumps(tally)}")
    if not args.skip_render:
        contacts, _, _ = analyse(scene, n_views, log=log)
        render_dir = run_dir / "render"
        render_dir.mkdir(exist_ok=True)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            overlay = pool.submit(
                render_overlay,
                scene,
                meta["video_a"],
                render_dir / "overlay_A_combined.mp4",
                combined,
                device=args.device,
                log=log,
            )
            export_viewer(scene, contacts, combined, run_dir / "viewer_combined", summary=summary)
            overlay.result()
    log("combine: done")
    return 0


def evaluate_runs(args: argparse.Namespace) -> int:
    """Before/after accuracy against the Olympic labels for every run directory given."""
    from boxing_analytics.mesh4d.combine import OUTCOMES
    from boxing_analytics.mesh4d.olympic_eval import combine_reports, evaluate, load_ground_truth

    variants: dict[str, list[dict[str, Any]]] = {
        "mesh_only": [],
        "videomae_only": [],
        "combined": [],
    }
    lines = [
        "| clip | method | labelled | matched | recall | hand | outcome (4 class) "
        "| landed/blocked/missed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in args.runs:
        run_dir = Path(run)
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        task_dir = Path(meta["video_a"]).parent.parent
        lo, hi = meta["window_a"]
        gt = load_ground_truth(task_dir / "annotations.json", lo, hi)
        combined_path = run_dir / "report_combined" / "events.json"
        if not combined_path.exists():
            print(f"skip {run}: no report_combined (run the combine command first)")
            continue
        combined = json.loads(combined_path.read_text(encoding="utf-8"))
        mesh = json.loads((run_dir / "report" / "events.json").read_text(encoding="utf-8"))
        video_only = []
        for e in combined:
            p = e["videomae_probabilities"]
            label = max(OUTCOMES, key=lambda k: p.get(k, 0.0))
            v = dict(e)
            v["outcome"] = "landed" if label.startswith("landed") else label
            v["target"] = {"landed_head": "head", "landed_body": "torso"}.get(label)
            video_only.append(v)
        for name, evs in (
            ("mesh_only", mesh),
            ("videomae_only", video_only),
            ("combined", combined),
        ):
            r = evaluate(gt, evs)
            variants[name].append(r)
            lines.append(
                f"| {run_dir.name} | {name} | {r['labelled_punches']} | {r['matched']} "
                f"| {r['recall']:.0%} | "
                f"{r['hand_accuracy']:.0%} | {r['outcome_accuracy_4class']:.0%} | "
                f"{r['outcome_accuracy_landed_blocked_missed']:.0%} |"
            )
    totals = {name: combine_reports(rs) for name, rs in variants.items() if rs}
    for name, r in totals.items():
        lines.append(
            f"| **all** | **{name}** | {r['labelled_punches']} | {r['matched']} "
            f"| {r['recall']:.0%} | "
            f"{r['hand_accuracy']:.0%} | {r['outcome_accuracy_4class']:.0%} | "
            f"{r['outcome_accuracy_landed_blocked_missed']:.0%} |"
        )
    table = "\n".join(lines)
    print(table)
    out = Path(args.output)
    out.write_text(table + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(safe_dumps(totals, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="boxing-analytics-4d",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run (or resume) the 4D pipeline")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--video-a", required=True)
    r.add_argument("--video-b")
    r.add_argument("--offset-s", type=float, help="manual sync: time in B = time in A + offset")
    r.add_argument("--start-s", type=float, default=0.0)
    r.add_argument("--duration-s", type=float, default=30.0, help="0 = to the end of the video")
    r.add_argument("--roles", default="red,blue", help="comma list from red,blue,referee")
    r.add_argument("--seed-a", action="append", help="manual seed for camera A: role=x1,y1,x2,y2")
    r.add_argument("--seed-b", action="append", help="manual seed for camera B: role=x1,y1,x2,y2")
    r.add_argument(
        "--seed-scan-s", type=float, default=3.0, help="seconds scanned for a clean seed frame"
    )
    r.add_argument("--stages", help=f"comma list from {','.join(STAGES)} (default: all)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--yolo-model", default="yolo11m-pose.pt")
    r.add_argument(
        "--sam2-checkpoint",
        default=os.environ.get("VARBOX_SAM2_CHECKPOINT", "models/sam2.1/sam2.1_hiera_large.pt"),
    )
    r.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    r.add_argument("--mask-max-side", type=int, default=1024)
    r.add_argument("--sam3d-dir", default=os.environ.get("SAM3D_BODY_DIR", "../sam-3d-body"))
    r.add_argument("--hf-repo", default="facebook/sam-3d-body-dinov3")
    r.add_argument("--inference-type", default="body", help="SAM 3D Body decoder: body | full")
    r.add_argument(
        "--mask-prompt", action="store_true", help="also pass masks (checkpoint must support it)"
    )
    r.add_argument("--viewer-max-frames", type=int, default=1500)
    r.add_argument(
        "--serial", action="store_true", help="process the two cameras one after another"
    )
    r.add_argument(
        "--mask-workers",
        type=int,
        default=0,
        help="SAM 2 processes tracking chunks at once (0 = from CPU cores, 1 = sequential)",
    )
    r.add_argument("--only-view", choices=["A", "B"], help=argparse.SUPPRESS)
    c = sub.add_parser("combine", help="rescore a finished run's events with VideoMAE")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--model-dir", default="models/varbox-videomae-cuda/best")
    c.add_argument("--mesh-weight", type=float, default=0.5)
    c.add_argument("--device", default="cuda")
    c.add_argument("--skip-render", action="store_true")
    e = sub.add_parser("evaluate", help="before/after accuracy against Olympic labels")
    e.add_argument("runs", nargs="+")
    e.add_argument("--output", default="runs/evaluation.md")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "combine":
        return combine(args)
    if args.command == "evaluate":
        return evaluate_runs(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
