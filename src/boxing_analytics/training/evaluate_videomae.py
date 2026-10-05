"""Evaluate an exported local VideoMAE checkpoint on a manifest split."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from boxing_analytics.training.manifest import load_manifest
from boxing_analytics.training.metrics import classification_metrics
from boxing_analytics.training.train_videomae import resolve_device
from boxing_analytics.training.videomae_dataset import VideoMAEManifestDataset, select_rows


def evaluate(
    *,
    manifest_path: str,
    model_dir: str,
    split: str,
    device_request: str,
    max_samples: int | None,
    clip_cache_dir: str = "",
    confidence_threshold: float = 0.55,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVideoClassification

    device = resolve_device(device_request)
    local_model = Path(model_dir).expanduser().resolve()
    if not local_model.is_dir():
        raise FileNotFoundError(f"Local VideoMAE model not found: {local_model}")
    config = AutoConfig.from_pretrained(local_model, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        local_model, local_files_only=True, use_fast=False
    )
    model = AutoModelForVideoClassification.from_pretrained(local_model, local_files_only=True).to(
        device
    )
    id2label = {int(key): value for key, value in config.id2label.items()}
    labels = [id2label[index] for index in sorted(id2label)]
    label2id = {label: index for index, label in id2label.items()}
    rows = select_rows(
        load_manifest(manifest_path),
        split=split,
        max_samples=max_samples,
        seed=73,
    )
    dataset = VideoMAEManifestDataset(
        rows=rows,
        processor=processor,
        label2id=label2id,
        frame_count=int(getattr(config, "num_frames", 16) or 16),
        image_size=int(getattr(config, "image_size", 224) or 224),
        allow_padding=False,
        cache_dir=clip_cache_dir,
    )
    loader: Any = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    truth: list[int] = []
    predicted: list[int] = []
    accepted: list[bool] = []
    source_groups: list[str] = []
    probabilities: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
            expected = int(batch["labels"].item())
            prediction = int(probs.argmax().item())
            truth.append(expected)
            predicted.append(prediction)
            accepted.append(float(probs[prediction]) >= confidence_threshold)
            source_groups.append(str(batch["source_group"][0]))
            probabilities.append(
                {
                    "clip_id": str(batch["clip_id"][0]),
                    "expected": labels[expected],
                    "predicted": labels[prediction],
                    "abstained": int(not accepted[-1]),
                    "probabilities": {
                        label: round(float(probs[index]), 6) for index, label in enumerate(labels)
                    },
                }
            )
    metrics = classification_metrics(truth, predicted, labels).to_dict()
    accepted_indices = [index for index, value in enumerate(accepted) if value]
    metrics["abstention_rate"] = round(
        1.0 - len(accepted_indices) / max(1, len(accepted)),
        6,
    )
    metrics["accuracy_non_abstained"] = round(
        sum(truth[index] == predicted[index] for index in accepted_indices)
        / max(1, len(accepted_indices)),
        6,
    )
    landed_labels = {"landed_head", "landed_body"}
    landed_indices = [index for index, value in enumerate(truth) if labels[value] in landed_labels]
    metrics["landed_target_zone_accuracy"] = round(
        sum(truth[index] == predicted[index] for index in landed_indices)
        / max(1, len(landed_indices)),
        6,
    )
    no_punch_id = label2id.get("no_punch")
    if no_punch_id is not None:
        negative_indices = [idx for idx, value in enumerate(truth) if value == no_punch_id]
        false_punches = sum(1 for idx in negative_indices if predicted[idx] != no_punch_id)
        metrics["false_punch_rate_on_negatives"] = round(
            false_punches / max(1, len(negative_indices)), 6
        )
        false_accepted_punches = sum(
            1 for idx in negative_indices if accepted[idx] and predicted[idx] != no_punch_id
        )
        metrics["false_accepted_punch_rate_on_negatives"] = round(
            false_accepted_punches / max(1, len(negative_indices)),
            6,
        )
    per_bout: dict[str, dict[str, object]] = {}
    for group in sorted(set(source_groups)):
        indices = [index for index, value in enumerate(source_groups) if value == group]
        per_bout[group] = classification_metrics(
            [truth[index] for index in indices],
            [predicted[index] for index in indices],
            labels,
        ).to_dict()
    return {
        "model_dir": str(local_model),
        "manifest": str(Path(manifest_path).resolve()),
        "split": split,
        "device": device,
        "confidence_threshold": confidence_threshold,
        "evaluation_scope": "model_classification_only",
        "metrics": metrics,
        "per_bout_metrics": per_bout,
        "tracker_metrics": {
            "identity_uncertain_event_rate": "not_applicable_model_only",
            "end_to_end_event_timing_error": "not_applicable_model_only",
            "duplicate_event_rate": "not_applicable_model_only",
            "evidence_clip_integrity": "not_applicable_model_only",
        },
        "predictions": probabilities,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local boxing VideoMAE model")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--clip-cache-dir",
        default=os.getenv(
            "VARBOX_VIDEOMAE_CLIP_CACHE_DIR",
            "training_cache/videomae_clips",
        ),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(
        manifest_path=str(args.manifest),
        model_dir=str(args.model_dir),
        split=str(args.split),
        device_request=str(args.device),
        max_samples=args.max_samples,
        clip_cache_dir=str(args.clip_cache_dir),
        confidence_threshold=float(args.confidence_threshold),
    )
    payload = json.dumps(result, indent=2)
    if args.output:
        output = Path(str(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        markdown = output.with_suffix(".md")
        metrics = cast(dict[str, object], result["metrics"])
        markdown.write_text(
            "# VideoMAE Evaluation\n\n"
            f"- Split: `{result['split']}`\n"
            f"- Device: `{result['device']}`\n"
            f"- Macro F1: `{metrics['macro_f1']}`\n"
            f"- Balanced accuracy: `{metrics['balanced_accuracy']}`\n"
            f"- Accuracy: `{metrics['accuracy']}`\n"
            f"- Abstention rate: `{metrics['abstention_rate']}`\n"
            f"- Accuracy among non-abstained: `{metrics['accuracy_non_abstained']}`\n"
            f"- Landed target-zone accuracy: `{metrics['landed_target_zone_accuracy']}`\n",
            encoding="utf-8",
        )
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
