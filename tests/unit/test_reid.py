import numpy as np

from boxing_analytics.tracking import reid as reid_module
from boxing_analytics.tracking.reid import HandcraftedReIDEmbedder, build_reid_embedder


def test_handcrafted_reid_embedder_returns_normalized_vector() -> None:
    crop = np.zeros((64, 48, 3), dtype=np.uint8)
    crop[:, :] = (40, 70, 180)

    vec = HandcraftedReIDEmbedder().embed(crop)

    assert vec.shape == (80,)
    assert np.isfinite(vec).all()
    assert np.linalg.norm(vec) > 0.99


def test_build_reid_embedder_supports_explicit_handcrafted_mode() -> None:
    embedder = build_reid_embedder(model_name="handcrafted", device="cpu", image_size=128)

    assert embedder.name == "handcrafted"
    assert embedder.device == "cpu"


def test_build_reid_embedder_prefers_osnet_for_auto(monkeypatch) -> None:
    class DummyEmbedder:
        name = "osnet_ain_x1_0"
        device = "cpu"

        def embed(self, crop):
            return np.ones((4,), dtype=np.float32)

    monkeypatch.setattr(
        reid_module,
        "TorchreidOSNetEmbedder",
        lambda model_name, device, image_size: DummyEmbedder(),
    )

    embedder = build_reid_embedder(model_name="auto", device="cpu", image_size=192)

    assert embedder.name == "osnet_ain_x1_0"


def test_build_reid_embedder_falls_back_to_handcrafted_when_osnet_init_fails(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("osnet unavailable")

    monkeypatch.setattr(reid_module, "TorchreidOSNetEmbedder", _raise)

    embedder = build_reid_embedder(model_name="osnet_ain_x1_0", device="cpu", image_size=192)

    assert isinstance(embedder, HandcraftedReIDEmbedder)
