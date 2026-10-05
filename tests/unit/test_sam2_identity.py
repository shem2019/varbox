import pytest

from boxing_analytics.tracking.sam2_identity import (
    Sam2FighterIdentityTrack,
    SamRoleSample,
    _force_float32_sam2_memory,
)


def test_sam_track_interpolates_boxes_and_assigns_unique_pose_ids(tmp_path) -> None:
    tracker = Sam2FighterIdentityTrack(checkpoint_path=str(tmp_path / "missing.pt"))
    tracker.samples = {
        "RED": [
            SamRoleSample(0, "RED", (0, 0, 100, 200), 100, 0.9),
            SamRoleSample(10, "RED", (20, 0, 120, 200), 100, 0.9),
        ],
        "BLUE": [
            SamRoleSample(0, "BLUE", (300, 0, 400, 200), 100, 0.9),
            SamRoleSample(10, "BLUE", (280, 0, 380, 200), 100, 0.9),
        ],
    }
    tracker._refresh_frame_index()
    assert tracker.box_for_role("RED", 5) == (10, 0, 110, 200)
    assignments = tracker.role_ids_for_poses(
        5,
        {
            7: {"box": (12, 0, 112, 200)},
            9: {"box": (278, 0, 378, 200)},
        },
    )
    assert assignments == {"RED": 7, "BLUE": 9}
    assert tracker.match_confidence_for_role("RED") > 0.8
    assert tracker.match_confidence_for_role("BLUE") > 0.8


def test_mps_compatibility_wrapper_restores_float32_memory() -> None:
    pytest.importorskip("sam2", reason="SAM 2 is installed only on GPU and Apple machines")

    class FakeMemory:
        def __init__(self, dtype: str) -> None:
            self.dtype = dtype

        def to(self, *args, **kwargs) -> "FakeMemory":
            return FakeMemory("float32")

    class FakePredictor:
        fill_hole_area = 0

        def _get_image_feature(self, inference_state, frame_idx, batch_size):
            return None, None, "features", "positions", "sizes"

        def track_step(self, **kwargs):
            return {
                "maskmem_features": FakeMemory("bfloat16"),
                "maskmem_pos_enc": ["position"],
                "pred_masks": FakeMemory("float32"),
                "obj_ptr": "pointer",
                "object_score_logits": "scores",
            }

        def _encode_new_memory(self, **kwargs):
            return FakeMemory("bfloat16"), ["position"]

        def _get_maskmem_pos_enc(self, inference_state, current_output):
            return current_output["maskmem_pos_enc"]

    predictor = FakePredictor()
    _force_float32_sam2_memory(predictor)

    state = {"num_frames": 1, "storage_device": "cpu"}
    compact_output, prediction = predictor._run_single_frame_inference(
        state,
        {},
        0,
        1,
        True,
        None,
        None,
        False,
        True,
    )
    memory, position = predictor._run_memory_encoder(
        state,
        0,
        1,
        "masks",
        "scores",
        True,
    )

    assert compact_output["maskmem_features"].dtype == "float32"
    assert prediction.dtype == "float32"
    assert memory.dtype == "float32"
    assert position == ["position"]
