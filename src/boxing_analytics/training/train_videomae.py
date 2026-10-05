"""Offline VideoMAE fine-tuning CLI with conservative Apple Silicon defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from boxing_analytics.training.annotation_schema import EIGHT_CLASS_LABELS, FIVE_CLASS_LABELS
from boxing_analytics.training.manifest import ClipManifestRow, load_manifest, manifest_digest
from boxing_analytics.training.metrics import classification_metrics
from boxing_analytics.training.videomae_dataset import VideoMAEManifestDataset, select_rows


def resolve_device(requested: str) -> str:
    import torch

    value = requested.strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if value not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(rows: Sequence[ClipManifestRow], label2id: dict[str, int], device: str) -> Any:
    import torch

    counts = Counter(row.label_five for row in rows)
    total = sum(counts.values())
    values = [
        total / max(1, len(label2id) * counts.get(label, 0))
        for label, _ in sorted(label2id.items(), key=lambda row: row[1])
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _evaluate(model: Any, loader: Any, device: str, labels: list[str]) -> dict[str, object]:
    import torch

    model.eval()
    truth: list[int] = []
    predictions: list[int] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            expected = batch["labels"].to(device)
            output = model(pixel_values=pixel_values, labels=expected)
            losses.append(float(output.loss.detach().cpu()))
            predicted = output.logits.argmax(dim=-1)
            truth.extend(int(value) for value in expected.detach().cpu().tolist())
            predictions.extend(int(value) for value in predicted.detach().cpu().tolist())
    metrics = classification_metrics(truth, predictions, labels).to_dict()
    metrics["loss"] = round(sum(losses) / max(1, len(losses)), 6)
    return metrics


def train(
    *,
    manifest_path: str,
    checkpoint: str,
    output_dir: str,
    label_schema: str,
    epochs: int,
    batch_size: int,
    device_request: str,
    learning_rate: float,
    gradient_accumulation: int,
    patience: int,
    seed: int,
    freeze_backbone: bool,
    max_train_samples: int | None,
    max_validation_samples: int | None,
    resume_from: str,
    clip_cache_dir: str,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVideoClassification

    _seed_everything(seed)
    device = resolve_device(device_request)
    labels = list(FIVE_CLASS_LABELS if label_schema == "five_class" else EIGHT_CLASS_LABELS)
    if label_schema == "eight_class":
        labels.append("no_punch")
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    rows = load_manifest(manifest_path)
    train_rows = select_rows(
        rows,
        split="train",
        max_samples=max_train_samples,
        seed=seed,
    )
    validation_rows = select_rows(
        rows,
        split="validation",
        max_samples=max_validation_samples,
        seed=seed + 1,
    )
    if not train_rows or not validation_rows:
        raise ValueError("Manifest must contain non-empty train and validation splits")
    source_path = Path(resume_from or checkpoint).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"VideoMAE checkpoint not found at {source_path}. "
            "Download it before running offline training."
        )
    processor = AutoImageProcessor.from_pretrained(
        str(source_path),
        local_files_only=True,
        use_fast=False,
    )
    source_config = AutoConfig.from_pretrained(str(source_path), local_files_only=True)
    frame_count = int(getattr(source_config, "num_frames", 16) or 16)
    image_size = int(getattr(source_config, "image_size", 224) or 224)
    model = AutoModelForVideoClassification.from_pretrained(
        str(source_path),
        local_files_only=True,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier")
    model.to(device)
    train_dataset = VideoMAEManifestDataset(
        rows=train_rows,
        processor=processor,
        label2id=label2id,
        frame_count=frame_count,
        image_size=image_size,
        allow_padding=False,
        cache_dir=clip_cache_dir,
    )
    validation_dataset = VideoMAEManifestDataset(
        rows=validation_rows,
        processor=processor,
        label2id=label2id,
        frame_count=frame_count,
        image_size=image_size,
        allow_padding=False,
        cache_dir=clip_cache_dir,
    )
    effective_batch = max(1, batch_size)
    train_loader: Any = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=0,
    )
    validation_loader: Any = DataLoader(
        validation_dataset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=0,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    class_weights = _class_weights(train_rows, label2id, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    stale_epochs = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            expected = batch["labels"].to(device)
            output = model(pixel_values=pixel_values)
            loss = criterion(output.logits, expected) / max(1, gradient_accumulation)
            loss.backward()
            epoch_losses.append(float(loss.detach().cpu()) * max(1, gradient_accumulation))
            if step % max(1, gradient_accumulation) == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch}/{epochs} step={step}/{len(train_loader)} "
                    f"loss={epoch_losses[-1]:.4f}",
                    flush=True,
                )
        validation_metrics = _evaluate(model, validation_loader, device, labels)
        epoch_row = {
            "epoch": epoch,
            "train_loss": round(sum(epoch_losses) / max(1, len(epoch_losses)), 6),
            "validation": validation_metrics,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        macro_f1 = float(str(validation_metrics["macro_f1"]))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            stale_epochs = 0
            best_dir = out / "best"
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, patience):
                break

    smoke_trained = bool(
        max_train_samples is not None
        or max_validation_samples is not None
        or epochs < 3
        or freeze_backbone
    )
    training_metadata = {
        "model_type": "videomae_temporal_strike_classifier",
        "base_checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "resumed_model_weights_from": (
            str(Path(resume_from).expanduser().resolve()) if resume_from else ""
        ),
        "label_schema": label_schema,
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
        "device": device,
        "frame_count": frame_count,
        "image_size": image_size,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": effective_batch,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "seed": seed,
        "freeze_backbone": int(freeze_backbone),
        "smoke_trained": int(smoke_trained),
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "manifest_digest": manifest_digest(rows),
        "training_split_digest": manifest_digest(train_rows),
        "validation_split_digest": manifest_digest(validation_rows),
        "manifest_sha256": _sha256(Path(manifest_path)),
        "git_commit": _git_commit(),
        "best_validation_macro_f1": round(best_macro_f1, 6),
        "duration_s": round(time.monotonic() - started, 3),
        "automatic_negative_labels_used": int(any(row.derived_label for row in train_rows)),
        "clip_cache_dir": str(Path(clip_cache_dir).expanduser().resolve()),
        "history": history,
    }
    best_dir = out / "best"
    (best_dir / "varbox_model_metadata.json").write_text(
        json.dumps(training_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return training_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE for boxing strike outcomes")
    parser.add_argument(
        "--manifest",
        default="",
        help="Existing generated manifest. If omitted, --dataset-dir is audited first.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.getenv("VARBOX_OLYMPIC_DATASET_DIR", ""),
        help="Olympic dataset root (or VARBOX_OLYMPIC_DATASET_DIR).",
    )
    parser.add_argument(
        "--audit-output-dir",
        default=os.getenv(
            "VARBOX_OLYMPIC_AUDIT_DIR",
            "artifacts/olympic_dataset_audit",
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv(
            "VARBOX_VIDEOMAE_BASE_CHECKPOINT",
            "models/videomae-base-finetuned-kinetics",
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--label-schema", choices=["five_class", "eight_class"], default="five_class"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume-from", default="")
    parser.add_argument(
        "--clip-cache-dir",
        default=os.getenv(
            "VARBOX_VIDEOMAE_CLIP_CACHE_DIR",
            "training_cache/videomae_clips",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_device = resolve_device(str(args.device))
    if str(args.batch_size).lower() == "auto":
        batch_size = 1 if resolved_device in {"mps", "cpu"} else 2
    else:
        batch_size = max(1, int(args.batch_size))
    manifest_path = str(args.manifest).strip()
    if not manifest_path:
        dataset_dir = str(args.dataset_dir).strip()
        if not dataset_dir:
            raise SystemExit("Pass --manifest or --dataset-dir, or set VARBOX_OLYMPIC_DATASET_DIR.")
        from boxing_analytics.training.dataset_audit import run_audit

        audit_output = Path(str(args.audit_output_dir)).expanduser().resolve()
        run_audit(
            dataset_dir=dataset_dir,
            output_dir=str(audit_output),
            seed=int(args.seed),
            train_ratio=float(args.train_ratio),
            validation_ratio=float(args.validation_ratio),
            test_ratio=float(args.test_ratio),
        )
        manifest_path = str(audit_output / "generated_manifest.jsonl")
    result = train(
        manifest_path=manifest_path,
        checkpoint=str(args.checkpoint),
        output_dir=str(args.output_dir),
        label_schema=str(args.label_schema),
        epochs=max(1, int(args.epochs)),
        batch_size=batch_size,
        device_request=resolved_device,
        learning_rate=float(args.learning_rate),
        gradient_accumulation=max(1, int(args.gradient_accumulation)),
        patience=max(1, int(args.patience)),
        seed=int(args.seed),
        freeze_backbone=bool(args.freeze_backbone),
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        resume_from=str(args.resume_from),
        clip_cache_dir=str(args.clip_cache_dir),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
