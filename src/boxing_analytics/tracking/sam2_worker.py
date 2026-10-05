"""Isolated SAM 2.1 MPS precomputation worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from boxing_analytics.tracking.sam2_identity import Sam2FighterIdentityTrack


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if len(sys.argv) != 2:
        return 64
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    progress_path = Path(request["progress_path"])
    tracker = Sam2FighterIdentityTrack(
        checkpoint_path=str(request["checkpoint_path"]),
        model_config=str(request["model_config"]),
        device=str(request["device"]),
        stride=int(request["stride"]),
        chunk_samples=int(request["chunk_samples"]),
        max_side=int(request["max_side"]),
    )

    def report_progress(message: str, percent: int | None) -> None:
        _write_json(progress_path, {"message": message, "percent": percent})

    try:
        ok = tracker.precompute(
            video_path=str(request["video_path"]),
            manual_seeds_payload=str(request["manual_seeds_payload"]),
            cache_path=str(request["cache_path"]),
            progress_cb=report_progress,
        )
        result = {
            "ok": ok,
            "status": tracker.status,
            "error": tracker.error,
            "diagnostics": tracker.diagnostics(),
        }
        _write_json(result_path, result)
        return 0 if ok else 2
    except Exception as exc:
        _write_json(
            result_path,
            {"ok": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
