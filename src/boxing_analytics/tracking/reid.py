"""Person ReID embedding extractors with OSNet-AIN and lightweight fallbacks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
FrameArray = NDArray[np.uint8]

_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_TORCHREID_OSNET_MODELS = {
    "osnet_ain_x1_0",
    "osnet_ain_x0_75",
    "osnet_ain_x0_5",
    "osnet_ain_x0_25",
}


class ReIDEmbedder(Protocol):
    name: str
    device: str

    def embed(self, crop: FrameArray) -> FloatArray: ...


def _l2_normalize(vec: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _resolve_torch_device(requested: str) -> str:
    import torch

    req = (requested or "auto").strip().lower()
    if req == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return req or "cpu"


@lru_cache(maxsize=1)
def _load_torchreid_osnet_module() -> Any:
    package_spec = importlib.util.find_spec("torchreid")
    if package_spec is None or not package_spec.origin:
        raise ModuleNotFoundError("torchreid is not installed")

    module_path = Path(package_spec.origin).resolve().parent / "reid" / "models" / "osnet_ain.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Torchreid OSNet module not found at {module_path}")

    module_spec = importlib.util.spec_from_file_location("_varbox_torchreid_osnet_ain", module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to load Torchreid OSNet module from {module_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class HandcraftedReIDEmbedder:
    """Low-cost fallback embedding that works without torch or torchreid."""

    name: str = "handcrafted"
    device: str = "cpu"
    image_size: int = 96

    def embed(self, crop: FrameArray) -> FloatArray:
        if crop.size == 0:
            return np.zeros((80,), dtype=np.float32)

        patch = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        _, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        hist_ang = cv2.calcHist([ang.astype(np.float32)], [0], None, [16], [0, 360]).flatten()
        hist_int = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        vec = np.concatenate([hist_ang, hist_int, hist_h, hist_s, hist_v]).astype(np.float32)
        return _l2_normalize(vec)


class TorchvisionReIDEmbedder:
    """Feature extractor backed by lightweight torchvision classification backbones."""

    def __init__(
        self, model_name: str = "mobilenet_v3_small", device: str = "auto", image_size: int = 160
    ):
        import torch
        from torchvision import models

        resolved_device = _resolve_torch_device(device)
        self._torch = torch
        self._device = torch.device(resolved_device)
        self.device = resolved_device
        self.image_size = max(96, int(image_size))
        self.name = model_name

        if model_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
            self._forward = self._mobilenet_forward
        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            self._forward = self._resnet_forward
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            self.name = "mobilenet_v3_small"
            self._forward = self._mobilenet_forward

        self._model = model.eval().to(self._device)

    def _preprocess(self, crop: FrameArray) -> Any:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        patch = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = self._torch.from_numpy(patch).to(self._device, dtype=self._torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = self._torch.tensor(_IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)
        std = self._torch.tensor(_IMAGENET_STD, device=self._device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _mobilenet_forward(self, tensor: Any) -> Any:
        feats = self._model.features(tensor)
        pooled = self._model.avgpool(feats)
        return self._torch.flatten(pooled, 1)

    def _resnet_forward(self, tensor: Any) -> Any:
        model = self._model
        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return self._torch.flatten(x, 1)

    def embed(self, crop: FrameArray) -> FloatArray:
        if crop.size == 0:
            return np.zeros((576,), dtype=np.float32)
        with self._torch.inference_mode():
            features = self._forward(self._preprocess(crop))
        vec = features.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        return _l2_normalize(vec)


class TorchreidOSNetEmbedder:
    """OSNet-AIN person ReID inference loaded directly from Torchreid model sources."""

    def __init__(
        self, model_name: str = "osnet_ain_x1_0", device: str = "auto", image_size: int = 256
    ):
        import torch

        if model_name not in _TORCHREID_OSNET_MODELS:
            raise KeyError(f"Unsupported Torchreid OSNet model: {model_name}")

        osnet_module = _load_torchreid_osnet_module()
        model_factory = getattr(osnet_module, model_name, None)
        if model_factory is None:
            raise AttributeError(f"Torchreid OSNet factory not found for {model_name}")

        resolved_device = _resolve_torch_device(device)
        self._torch = torch
        self._device = torch.device(resolved_device)
        self.device = resolved_device
        self.name = model_name
        self.image_height = max(128, int(image_size))
        self.image_width = max(64, int(round(self.image_height * 0.5)))
        self.embedding_dim = 512

        model = model_factory(
            num_classes=1000,
            pretrained=True,
            loss="softmax",
        )
        self._model = model.eval().to(self._device)
        self.embedding_dim = int(
            getattr(model, "feature_dim", self.embedding_dim) or self.embedding_dim
        )

    def _preprocess(self, crop: FrameArray) -> Any:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        patch = cv2.resize(
            rgb,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = self._torch.from_numpy(patch).to(self._device, dtype=self._torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = self._torch.tensor(_IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)
        std = self._torch.tensor(_IMAGENET_STD, device=self._device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def embed(self, crop: FrameArray) -> FloatArray:
        if crop.size == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32)
        with self._torch.inference_mode():
            features = self._model(self._preprocess(crop))
        vec = features.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        return _l2_normalize(vec)


def build_reid_embedder(
    model_name: str = "auto",
    device: str = "auto",
    image_size: int = 160,
) -> ReIDEmbedder:
    requested = (model_name or "auto").strip().lower()
    if requested in {"", "auto"}:
        requested = "osnet_ain_x1_0"
    if requested == "handcrafted":
        return HandcraftedReIDEmbedder(image_size=min(int(image_size), 128))
    try:
        if requested in _TORCHREID_OSNET_MODELS:
            return TorchreidOSNetEmbedder(
                model_name=requested,
                device=device,
                image_size=max(160, int(image_size)),
            )
        return TorchvisionReIDEmbedder(
            model_name=requested,
            device=device,
            image_size=image_size,
        )
    except Exception:
        return HandcraftedReIDEmbedder(image_size=min(int(image_size), 128))
