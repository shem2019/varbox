"""Evaluation harness CLI implementation."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from boxing_analytics.eval.schema import CONTACT_LABELS, EvalDataset, EventSample, load_dataset

LANDED_LABELS = {"landed_clean", "landed_glancing"}


@dataclass(frozen=True, slots=True)
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _load_model_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("model config must be a JSON object")
    return raw


def _contact_metrics(samples: list[EventSample]) -> dict[str, dict[str, float]]:
    counts = {label: _Counts() for label in CONTACT_LABELS}
    mutable = {label: {"tp": 0, "fp": 0, "fn": 0} for label in CONTACT_LABELS}

    for sample in samples:
        gt = sample.ground_truth_label
        pred = sample.predicted_label
        for label in CONTACT_LABELS:
            if pred == label and gt == label:
                mutable[label]["tp"] += 1
            elif pred == label and gt != label:
                mutable[label]["fp"] += 1
            elif pred != label and gt == label:
                mutable[label]["fn"] += 1

    metrics: dict[str, dict[str, float]] = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0

    for label in CONTACT_LABELS:
        counted = _Counts(
            tp=mutable[label]["tp"],
            fp=mutable[label]["fp"],
            fn=mutable[label]["fn"],
        )
        counts[label] = counted
        precision = _safe_div(float(counted.tp), float(counted.tp + counted.fp))
        recall = _safe_div(float(counted.tp), float(counted.tp + counted.fn))
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": float(counted.tp + counted.fn),
        }

    divisor = float(len(CONTACT_LABELS))
    metrics["macro"] = {
        "precision": round(macro_p / divisor, 4),
        "recall": round(macro_r / divisor, 4),
        "f1": round(macro_f1 / divisor, 4),
        "support": float(len(samples)),
    }
    return metrics


def _target_zone_accuracy(samples: list[EventSample]) -> float:
    total = 0
    correct = 0
    for sample in samples:
        if sample.ground_truth_label in LANDED_LABELS:
            total += 1
            if sample.ground_truth_zone == sample.predicted_zone:
                correct += 1
    return round(_safe_div(float(correct), float(total)), 4)


def _id_switch_rate(samples: list[EventSample]) -> float:
    by_fighter: dict[str, list[EventSample]] = {}
    for sample in samples:
        by_fighter.setdefault(sample.fighter_id, []).append(sample)

    switches = 0
    transitions = 0
    for rows in by_fighter.values():
        ordered = sorted(rows, key=lambda row: row.timestamp_s)
        for idx in range(1, len(ordered)):
            transitions += 1
            if ordered[idx].predicted_track_id != ordered[idx - 1].predicted_track_id:
                switches += 1
    return round(_safe_div(float(switches), float(transitions)), 4)


def _round_alignment_error(dataset: EvalDataset) -> float:
    errors: list[float] = []
    for video in dataset.videos:
        gt = {marker.round_no: marker.timestamp_s for marker in video.round_markers_gt}
        pred = {marker.round_no: marker.timestamp_s for marker in video.round_markers_pred}
        for round_no, gt_ts in gt.items():
            if round_no in pred:
                errors.append(abs(gt_ts - pred[round_no]))
    if not errors:
        return 0.0
    return round(sum(errors) / float(len(errors)), 4)


def _clip_integrity(
    *,
    samples: list[EventSample],
    dataset_dir: Path,
    evidence_required_labels: set[str],
) -> float:
    required = [sample for sample in samples if sample.predicted_label in evidence_required_labels]
    if not required:
        return 1.0
    ok = 0
    for sample in required:
        clip = sample.evidence_clip.strip()
        if not clip:
            continue
        clip_path = (dataset_dir / clip).resolve()
        if clip_path.is_file():
            ok += 1
    return round(_safe_div(float(ok), float(len(required))), 4)


def run_evaluation(
    *,
    dataset_path: str,
    model_config_path: str,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    model_cfg = _load_model_config(model_config_path)
    required_labels_raw = model_cfg.get("evidence_required_labels", sorted(LANDED_LABELS))
    evidence_required_labels = {
        str(label).strip().lower() for label in required_labels_raw if str(label).strip()
    }
    if not evidence_required_labels:
        evidence_required_labels = set(LANDED_LABELS)

    dataset_dir = Path(dataset_path).resolve().parent
    samples = [sample for video in dataset.videos for sample in video.samples]

    metrics: dict[str, Any] = {
        "dataset_name": dataset.dataset_name,
        "sample_count": len(samples),
        "contact_metrics": _contact_metrics(samples),
        "target_zone_accuracy": _target_zone_accuracy(samples),
        "id_switch_rate": _id_switch_rate(samples),
        "timing_alignment_error_s": _round_alignment_error(dataset),
        "clip_evidence_integrity": _clip_integrity(
            samples=samples,
            dataset_dir=dataset_dir,
            evidence_required_labels=evidence_required_labels,
        ),
    }

    # compact single score for regression thresholds
    contact_metrics = cast(dict[str, Any], metrics["contact_metrics"])
    macro_metrics = cast(dict[str, Any], contact_metrics["macro"])
    macro_f1 = float(macro_metrics["f1"])
    metrics["quality_index"] = round(
        (0.45 * macro_f1)
        + (0.20 * float(metrics["target_zone_accuracy"]))
        + (0.20 * (1.0 - float(metrics["id_switch_rate"])))
        + (0.15 * float(metrics["clip_evidence_integrity"])),
        4,
    )
    if math.isnan(float(metrics["quality_index"])):
        metrics["quality_index"] = 0.0
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run boxing analytics evaluation")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--model-config", required=True, help="Path to model config JSON")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for metrics JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metrics = run_evaluation(
        dataset_path=str(args.dataset),
        model_config_path=str(args.model_config),
    )
    payload = json.dumps(metrics, indent=2)
    if args.output:
        out_path = Path(str(args.output))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
