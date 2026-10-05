"""CLI for auditing the Olympic dataset and generating a training manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from boxing_analytics.training.annotation_schema import ORIGINAL_LABEL_MAPPING
from boxing_analytics.training.clip_extractor import decode_manifest_clip, export_preview_clip
from boxing_analytics.training.manifest import ClipManifestRow, manifest_digest, write_manifest
from boxing_analytics.training.olympic_dataset_adapter import (
    build_manifest_rows,
    load_annotations,
)
from boxing_analytics.training.split_dataset import (
    split_summary,
    validate_no_group_leakage,
)

DEFAULT_DATASET_ENV = "VARBOX_OLYMPIC_DATASET_DIR"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet(
    rows: list[ClipManifestRow],
    output_path: Path,
    *,
    sample_limit: int,
) -> None:
    tiles: list[np.ndarray] = []
    for row in rows[:sample_limit]:
        try:
            decoded = decode_manifest_clip(row, frame_count=9, output_size=192)
        except (FileNotFoundError, ValueError):
            continue
        frame = cv2.cvtColor(decoded.frames_rgb[len(decoded.frames_rgb) // 2], cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (0, 0), (192, 28), (12, 18, 28), -1)
        cv2.putText(
            frame,
            f"{row.label_five} | {row.split}",
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 245, 255),
            1,
        )
        tiles.append(frame)
    if not tiles:
        return
    columns = 4
    rows_needed = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows_needed * 192, columns * 192, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row_idx, col_idx = divmod(index, columns)
        canvas[row_idx * 192 : (row_idx + 1) * 192, col_idx * 192 : (col_idx + 1) * 192] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def run_audit(
    *,
    dataset_dir: str,
    output_dir: str,
    seed: int = 42,
    clip_duration_s: float = 0.60,
    safety_margin_s: float = 1.0,
    negatives_per_video: int = 24,
    sample_limit: int = 20,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    annotations, sources, invalid = load_annotations(dataset_dir)
    manifest_rows = build_manifest_rows(
        annotations,
        sources,
        seed=seed,
        clip_duration_s=clip_duration_s,
        safety_margin_s=safety_margin_s,
        negatives_per_video=negatives_per_video,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    validate_no_group_leakage([row.to_dict() for row in manifest_rows])
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "generated_manifest.jsonl"
    write_manifest(manifest_rows, manifest_path)

    original_counts = Counter(row.original_label for row in manifest_rows if not row.derived_label)
    mapped_counts = Counter(row.label_five for row in manifest_rows)
    split_counts = Counter(row.split for row in manifest_rows)
    invalid_reason_counts = Counter(str(row.get("reason", "unknown")) for row in invalid)
    per_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest_rows:
        per_group[row.source_group][row.label_five] += 1

    summary = {
        "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
        "source_video_count": len(sources),
        "source_group_count": len({source.source_group for source in sources}),
        "annotation_count": len(annotations),
        "derived_negative_count": len([row for row in manifest_rows if row.derived_label]),
        "manifest_row_count": len(manifest_rows),
        "invalid_annotation_count": len(invalid),
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "missing_video_count": sum(
            count
            for reason, count in invalid_reason_counts.items()
            if reason.startswith("expected_one_video_found_0")
        ),
        "duplicate_annotation_count": invalid_reason_counts.get("duplicate_annotation", 0),
        "original_label_counts": dict(sorted(original_counts.items())),
        "mapped_label_counts": dict(sorted(mapped_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "clip_duration_s": clip_duration_s,
        "negative_safety_margin_s": max(1.0, safety_margin_s),
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "manifest_digest": manifest_digest(manifest_rows),
        "automatic_negative_notice": (
            "no_punch rows are derived from unannotated intervals and are not referee-verified"
        ),
    }
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "annotation_schema.json").write_text(
        json.dumps(
            {
                "format": "CVAT tracks JSON",
                "label_mapping": {
                    key: {"five_class": value[0], "eight_class": value[1], "hand": value[2]}
                    for key, value in ORIGINAL_LABEL_MAPPING.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "invalid_annotations.json").write_text(
        json.dumps(invalid, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class_rows = [
        {"schema": "original", "label": label, "count": count}
        for label, count in sorted(original_counts.items())
    ] + [
        {"schema": "five_class", "label": label, "count": count}
        for label, count in sorted(mapped_counts.items())
    ]
    _write_csv(out / "class_counts.csv", ["schema", "label", "count"], class_rows)
    source_rows = [
        {
            "task_name": source.task_name,
            "source_video": source.path,
            "source_group": source.source_group,
            "camera": source.camera,
            "fps": source.fps,
            "frame_count": source.frame_count,
            "duration_s": round(source.duration_s, 3),
            "width": source.width,
            "height": source.height,
        }
        for source in sources
    ]
    _write_csv(
        out / "source_files.csv",
        [
            "task_name",
            "source_video",
            "source_group",
            "camera",
            "fps",
            "frame_count",
            "duration_s",
            "width",
            "height",
        ],
        source_rows,
    )
    assignments = {row.source_group: row.split for row in manifest_rows}
    split_payload = split_summary(assignments)
    split_payload["per_group_class_counts"] = {
        group: dict(sorted(counts.items())) for group, counts in sorted(per_group.items())
    }
    (out / "split_summary.json").write_text(
        json.dumps(split_payload, indent=2) + "\n", encoding="utf-8"
    )

    sample_candidates: dict[str, list[ClipManifestRow]] = defaultdict(list)
    for row in sorted(
        manifest_rows,
        key=lambda item: (item.source_group, item.event_time_s, item.clip_id),
    ):
        sample_candidates[row.label_five].append(row)
    sample_rows: list[ClipManifestRow] = []
    labels = sorted(sample_candidates)
    while len(sample_rows) < sample_limit and labels:
        remaining: list[str] = []
        for label in labels:
            values = sample_candidates[label]
            if values and len(sample_rows) < sample_limit:
                sample_rows.append(values.pop(0))
            if values:
                remaining.append(label)
        labels = remaining
    clips_dir = out / "sample_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stale_clip in clips_dir.glob("*.mp4"):
        stale_clip.unlink()
    for row in sample_rows:
        try:
            export_preview_clip(row, clips_dir / f"{row.clip_id}.mp4")
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
    _contact_sheet(
        sample_rows,
        out / "sample_contact_sheet" / "contact_sheet.jpg",
        sample_limit=sample_limit,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the Olympic boxing punch dataset")
    parser.add_argument(
        "--dataset-dir",
        default=os.getenv(DEFAULT_DATASET_ENV, ""),
        help=f"Dataset directory (or set {DEFAULT_DATASET_ENV}).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-duration", type=float, default=0.60)
    parser.add_argument("--negative-safety-margin", type=float, default=1.0)
    parser.add_argument("--negatives-per-video", type=int, default=24)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not str(args.dataset_dir).strip():
        raise SystemExit(
            f"Dataset directory is required. Pass --dataset-dir or set {DEFAULT_DATASET_ENV}."
        )
    summary = run_audit(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        clip_duration_s=float(args.clip_duration),
        safety_margin_s=float(args.negative_safety_margin),
        negatives_per_video=int(args.negatives_per_video),
        sample_limit=int(args.sample_limit),
        train_ratio=float(args.train_ratio),
        validation_ratio=float(args.validation_ratio),
        test_ratio=float(args.test_ratio),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
