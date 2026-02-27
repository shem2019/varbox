import numpy as np

from boxing_analytics.tracking.identity_manager import IdentityManager


def _make_frame() -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:180, 30:110] = (25, 25, 200)
    frame[60:180, 210:290] = (200, 30, 30)
    frame[70:170, 145:175] = (180, 180, 180)
    return frame


def test_identity_manager_assigns_two_fighters() -> None:
    frame = _make_frame()
    poses = {
        10: {"box": (30, 60, 110, 180), "keypoints": {}, "det_conf": 0.9},
        20: {"box": (210, 60, 290, 180), "keypoints": {}, "det_conf": 0.9},
    }

    manager = IdentityManager()
    manager.update(frame, poses, frame_idx=1, timestamp_s=0.033)

    assert manager.id_for_role("RED") is not None
    assert manager.id_for_role("BLUE") is not None
    assert manager.id_for_role("RED") != manager.id_for_role("BLUE")


def test_identity_manager_assigns_referee_candidate() -> None:
    frame = _make_frame()
    poses = {
        10: {"box": (30, 60, 110, 180), "keypoints": {}, "det_conf": 0.95},
        20: {"box": (210, 60, 290, 180), "keypoints": {}, "det_conf": 0.95},
        30: {"box": (145, 70, 175, 170), "keypoints": {}, "det_conf": 0.8},
    }

    manager = IdentityManager()
    manager.update(frame, poses, frame_idx=1, timestamp_s=0.033)

    assert manager.id_for_role("REF") == 30


def test_identity_manager_exposes_hypothesis_debug_fields() -> None:
    frame = _make_frame()
    manager = IdentityManager(viterbi_window=8, switch_penalty=2.5)

    for i in range(1, 10):
        ts = i / 30.0
        overlap = 0 if i < 5 else min(25, (i - 4) * 7)
        poses = {
            1: {
                "box": (30 + overlap, 60, 110 + overlap, 180),
                "keypoints": {},
                "det_conf": 0.9,
            },
            2: {
                "box": (210 - overlap, 60, 290 - overlap, 180),
                "keypoints": {},
                "det_conf": 0.9,
            },
        }
        manager.update(frame, poses, frame_idx=i, timestamp_s=ts)

    debug = manager.hypothesis_log()[-1]
    assert {"logp_h0", "logp_h1", "chosen_hypothesis", "delta", "occlusion_score"}.issubset(
        debug.keys()
    )
