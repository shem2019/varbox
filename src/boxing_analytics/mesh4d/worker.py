"""GPU worker for the VAR Box dashboard.

The worker only makes outbound HTTPS calls, so any rented GPU can serve the dashboard without
opening ports. Two modes:

    publish  upload a finished run's web package as an analysis
    serve    loop: heartbeat, claim an imported job, process it, stream progress, publish

    python -m boxing_analytics.mesh4d.worker serve --server https://varbox.guestpassvms.com
    (token from VARBOX_WORKER_TOKEN)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

CHUNK = 8 * 1024 * 1024


class Dashboard:
    def __init__(self, server: str, token: str) -> None:
        self.server = server.rstrip("/")
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {token}"
        self.http.headers["User-Agent"] = "varbox-worker/1"

    def _url(self, path: str) -> str:
        return f"{self.server}/api/worker/{path.lstrip('/')}"

    def call(self, method: str, path: str, **kwargs: Any) -> Any:
        for attempt in range(5):
            try:
                r = self.http.request(method, self._url(path), timeout=120, **kwargs)
                if r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
                r.raise_for_status()
                return r.json() if r.content else {}
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                if attempt == 4 or (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code < 500
                ):
                    raise
                time.sleep(2 * (attempt + 1))
        return None

    def upload_file(self, analysis_id: int, local: Path, rel: str) -> None:
        size = local.stat().st_size
        with local.open("rb") as fh:
            offset = 0
            while True:
                chunk = fh.read(CHUNK)
                self.call(
                    "PUT",
                    f"analyses/{analysis_id}/files",
                    params={"path": rel, "offset": offset, "total": size},
                    data=chunk,
                    headers={"Content-Type": "application/octet-stream"},
                )
                offset += len(chunk)
                if offset >= size:
                    break

    def download(self, url_path: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.http.get(self._url(url_path), stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as fh:
                for block in r.iter_content(CHUNK):
                    fh.write(block)
            tmp.replace(dest)


def publish(dash: Dashboard, web_dir: Path, title: str, job_id: int | None = None) -> int:
    analysis = json.loads((web_dir / "analysis.json").read_text(encoding="utf-8"))
    analysis["title"] = title
    created = dash.call("POST", "analyses", json={"job_id": job_id, "analysis": analysis})
    analysis_id = int(created["id"])
    files = sorted(f for f in web_dir.rglob("*") if f.is_file())
    for i, f in enumerate(files, 1):
        rel = f.relative_to(web_dir).as_posix()
        print(f"upload {i}/{len(files)} {rel} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)
        dash.upload_file(analysis_id, f, rel)
    dash.call("POST", f"analyses/{analysis_id}/finalize", json={"analysis": analysis})
    print(f"published analysis {analysis_id}: {title}", flush=True)
    return analysis_id


def gpu_status() -> dict[str, Any]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        name, used, total, util = (x.strip() for x in out.splitlines()[0].split(","))
        return {
            "gpu": name,
            "mem_used_mb": int(used),
            "mem_total_mb": int(total),
            "util": int(util),
        }
    except Exception:
        return {"gpu": "unknown"}


STAGE_PATTERNS = [
    ("sync", re.compile(r"sync: ")),
    ("seed", re.compile(r"seed \w+: ")),
    ("masks", re.compile(r"masks: chunk \d+ frames (\d+)-(\d+)")),
    ("meshes", re.compile(r"body: frame (\d+)")),
    ("world", re.compile(r"world: ")),
    ("fusion", re.compile(r"fuse: ")),
    ("contact", re.compile(r"contact: |summary: ")),
    ("render", re.compile(r"render( overlay| isolated)?")),
    ("rescoring", re.compile(r"combine: ")),
]


def _progress_from_log(line: str, window: tuple[int, int]) -> tuple[str, float | None]:
    start, stop = window
    span = max(1, stop - start)
    for stage, pattern in STAGE_PATTERNS:
        m = pattern.search(line)
        if not m:
            continue
        if stage == "masks" and m.groups():
            return stage, min(1.0, (int(m.group(2)) - start) / span)
        if stage == "meshes":
            return stage, min(1.0, (int(m.group(1)) - start) / span)
        return stage, None
    return "", None


def _tail_progress(dash: Dashboard, job_id: int, log_path: Path, stop: threading.Event) -> None:
    """Forward pipeline log lines as job progress every few seconds."""
    seen = 0
    window = (0, 1)
    last_sent = 0.0
    stage, pct, message = "starting", None, ""
    previews_sent: set[str] = set()
    while not stop.is_set():
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[seen:]:
                text = line.split("] ", 1)[-1]
                s, p = _progress_from_log(text, window)
                if s:
                    if s != stage:
                        pct = None
                    stage, message = s, text
                    if p is not None:
                        pct = p
            seen = len(lines)
            run_json = log_path.parent / "run.json"
            if run_json.exists() and window == (0, 1):
                meta = json.loads(run_json.read_text(encoding="utf-8"))
                lo, hi = meta.get("window_a", [0, 1])
                window = (int(lo), int(hi))
            for preview in sorted(log_path.parent.glob("view_*/seed_preview.jpg")):
                key = str(preview)
                if key not in previews_sent:
                    with preview.open("rb") as fh:
                        dash.call(
                            "POST",
                            f"jobs/{job_id}/preview",
                            params={"name": preview.parent.name},
                            data=fh.read(),
                            headers={"Content-Type": "image/jpeg"},
                        )
                    previews_sent.add(key)
        if time.monotonic() - last_sent > 4:
            dash.call(
                "POST",
                f"jobs/{job_id}/progress",
                json={"stage": stage, "pct": pct, "message": message[:300]},
            )
            last_sent = time.monotonic()
        stop.wait(2)


def run_job(dash: Dashboard, job: dict[str, Any], workdir: Path, repo: Path) -> None:
    job_id = int(job["id"])
    jdir = workdir / f"job_{job_id}"
    inputs = {}
    for cam in ("A", "B"):
        upload = job.get(f"camera_{cam.lower()}")
        if not upload:
            continue
        dest = jdir / "raw" / f"{cam}_{upload['name']}"
        if not dest.exists():
            dash.call(
                "POST",
                f"jobs/{job_id}/progress",
                json={"stage": "download", "message": f"downloading camera {cam}"},
            )
            dash.download(f"uploads/{upload['id']}", dest)
        prepared = jdir / f"camera_{cam}.mp4"
        if not prepared.exists():
            fps = str(job.get("fps") or 30)
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(dest),
                    "-vf", f"fps={fps},scale='min(1920,iw)':-2", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-ar", "48000", "-ac", "1",
                    "-movflags", "+faststart", str(prepared),
                ],
                check=True,
            )  # fmt: skip
        inputs[cam] = prepared
    run_dir = repo / "runs" / "jobs" / f"job_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable, "-m", "boxing_analytics.mesh4d.cli", "run",
        "--run-dir", str(run_dir), "--video-a", str(inputs["A"]),
        "--start-s", str(job.get("start_s") or 0), "--duration-s", str(job.get("duration_s") or 0),
        "--roles", "red,blue", "--seed-scan-s", "20",
    ]  # fmt: skip
    if "B" in inputs:
        args += ["--video-b", str(inputs["B"])]
    stop = threading.Event()
    tail = threading.Thread(
        target=_tail_progress, args=(dash, job_id, run_dir / "run.log", stop), daemon=True
    )
    tail.start()
    try:
        subprocess.run(args, check=True, cwd=repo)
        model = repo / "models" / "varbox-videomae-cuda" / "best" / "model.safetensors"
        if model.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "boxing_analytics.mesh4d.cli",
                    "combine",
                    "--run-dir",
                    str(run_dir),
                ],
                check=True,
                cwd=repo,
            )
        from boxing_analytics.mesh4d.webexport import export_web

        dash.call(
            "POST",
            f"jobs/{job_id}/progress",
            json={"stage": "packaging", "message": "building layers"},
        )
        web = export_web(run_dir, title=job.get("title"))
        dash.call(
            "POST",
            f"jobs/{job_id}/progress",
            json={"stage": "uploading", "message": "sending results"},
        )
        publish(dash, web, job.get("title") or f"Job {job_id}", job_id=job_id)
        dash.call("POST", f"jobs/{job_id}/done", json={})
    finally:
        stop.set()


def serve(dash: Dashboard, workdir: Path, repo: Path, name: str) -> None:
    print(f"worker {name} serving {dash.server}", flush=True)
    while True:
        try:
            dash.call("POST", "heartbeat", json={"name": name, **gpu_status()})
            job = dash.call("POST", "claim", json={"name": name}).get("job")
        except Exception as exc:  # network blips: keep the worker alive
            print(f"dashboard unreachable: {exc}", flush=True)
            time.sleep(15)
            continue
        if not job:
            time.sleep(10)
            continue
        print(f"claimed job {job['id']}: {job.get('title')}", flush=True)
        try:
            run_job(dash, job, workdir, repo)
        except Exception as exc:
            print(f"job {job['id']} failed: {exc}", flush=True)
            with contextlib.suppress(Exception):
                dash.call("POST", f"jobs/{job['id']}/fail", json={"message": str(exc)[:500]})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="varbox-worker")
    p.add_argument(
        "--server", default=os.environ.get("VARBOX_SERVER", "https://varbox.guestpassvms.com")
    )
    p.add_argument("--token", default=os.environ.get("VARBOX_WORKER_TOKEN", ""))
    sub = p.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish", help="upload a run's web package (builds it when missing)")
    pub.add_argument("run_dir")
    pub.add_argument("--title")
    pub.add_argument("--rebuild", action="store_true")
    srv = sub.add_parser("serve", help="process jobs imported on the dashboard")
    srv.add_argument("--workdir", default=str(Path.home() / "work" / "jobs"))
    srv.add_argument("--name", default=socket.gethostname())
    args = p.parse_args(argv)
    if not args.token:
        p.error("set VARBOX_WORKER_TOKEN or pass --token")
    dash = Dashboard(args.server, args.token)
    repo = Path(__file__).resolve().parents[3]
    if args.command == "publish":
        run_dir = Path(args.run_dir)
        web = run_dir / "web"
        if args.rebuild or not (web / "analysis.json").exists():
            from boxing_analytics.mesh4d.webexport import export_web

            web = export_web(run_dir, title=args.title)
        publish(dash, web, args.title or run_dir.name)
        return 0
    serve(dash, Path(args.workdir), repo, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
