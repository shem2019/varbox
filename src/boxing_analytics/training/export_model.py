"""Validate and fingerprint a locally exported VideoMAE model directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint_model(model_dir: str) -> dict[str, object]:
    root = Path(model_dir).expanduser().resolve()
    required = ("config.json", "preprocessor_config.json")
    missing = [name for name in required if not (root / name).is_file()]
    weights = [path for path in root.glob("*.safetensors") if path.is_file()]
    if not weights:
        weights = [path for path in root.glob("pytorch_model*.bin") if path.is_file()]
    if missing or not weights:
        raise FileNotFoundError(
            f"Incomplete model export at {root}; missing={missing}, weights={len(weights)}"
        )
    digest = hashlib.sha256()
    files = sorted([root / name for name in required] + weights)
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    metadata_path = root / "varbox_model_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    )
    return {
        "model_dir": str(root),
        "sha256": digest.hexdigest(),
        "files": [path.name for path in files],
        "labels": config.get("id2label", {}),
        "model_type": config.get("model_type", ""),
        "training_metadata": metadata,
    }
