from pathlib import Path

import pytest
import torch

from boxing_analytics.detection.temporal_classifier import (
    VideoMAETemporalStrikeClassifier,
    resolve_inference_device,
)


def test_temporal_classifier_rejects_missing_local_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Local VideoMAE model"):
        VideoMAETemporalStrikeClassifier(model_dir=str(tmp_path / "missing"))


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_inference_device("auto") == "cpu"


def test_auto_device_prefers_mps_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_inference_device("auto") == "mps"
