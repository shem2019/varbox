from __future__ import annotations

import numpy as np

from boxing_analytics.mesh4d.contact import (
    REGIONS,
    ContactConfig,
    compute_frame_contacts,
    detect_punches,
    label_vertices,
)
from boxing_analytics.mesh4d.export import cluster_decimate
from boxing_analytics.mesh4d.geometry import (
    KP,
    LEFT_KNUCKLES,
    RIGHT_KNUCKLES,
    apply_transform,
    fit_floor_plane,
    ground_scale_factors,
    one_euro,
    ransac_umeyama,
    world_from_camera,
)
from boxing_analytics.mesh4d.sync import audio_offset, map_frame


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _figure(
    origin: np.ndarray, facing: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Crude standing boxer in world coords (y up): 70 MHR keypoints and a surface point cloud."""
    f = facing / np.linalg.norm(facing)
    side = np.cross(np.array([0.0, 1.0, 0.0]), f)  # fighter's left
    kp = np.zeros((70, 3))

    def at(h: float, s: float = 0.0, fwd: float = 0.0) -> np.ndarray:
        return origin + np.array([0, h, 0]) + s * side + fwd * f

    kp[KP["nose"]] = at(1.62, 0, 0.1)
    kp[KP["left_eye"]], kp[KP["right_eye"]] = at(1.66, 0.03, 0.08), at(1.66, -0.03, 0.08)
    kp[KP["left_ear"]], kp[KP["right_ear"]] = at(1.62, 0.08), at(1.62, -0.08)
    kp[KP["neck"]] = at(1.48)
    kp[KP["left_shoulder"]], kp[KP["right_shoulder"]] = at(1.42, 0.2), at(1.42, -0.2)
    kp[KP["left_elbow"]], kp[KP["right_elbow"]] = at(1.2, 0.25, 0.15), at(1.2, -0.25, 0.15)
    kp[KP["left_wrist"]], kp[KP["right_wrist"]] = at(1.4, 0.15, 0.3), at(1.4, -0.15, 0.3)
    for i in LEFT_KNUCKLES:
        kp[i] = at(1.42, 0.15, 0.38)
    for i in RIGHT_KNUCKLES:
        kp[i] = at(1.42, -0.15, 0.38)
    kp[KP["left_hip"]], kp[KP["right_hip"]] = at(0.95, 0.12), at(0.95, -0.12)
    kp[KP["left_knee"]], kp[KP["right_knee"]] = at(0.5, 0.13), at(0.5, -0.13)
    kp[KP["left_ankle"]], kp[KP["right_ankle"]] = at(0.08, 0.14), at(0.08, -0.14)
    for i in (15, 16, 17):
        kp[i] = at(0.0, 0.14)
    for i in (18, 19, 20):
        kp[i] = at(0.0, -0.14)
    pts = []
    head_c = at(1.62)
    d = rng.normal(size=(600, 3))
    pts.append(head_c + 0.11 * d / np.linalg.norm(d, axis=1, keepdims=True))
    for _ in range(1500):  # torso surface: front/back plates
        u, v = rng.uniform(-0.17, 0.17), rng.uniform(0.95, 1.45)
        pts.append((origin + np.array([0, v, 0]) + u * side + rng.choice([-0.11, 0.11]) * f)[None])
    for a, b in (
        (KP["left_shoulder"], KP["left_elbow"]),
        (KP["right_shoulder"], KP["right_elbow"]),
        (KP["left_elbow"], KP["left_wrist"]),
        (KP["right_elbow"], KP["right_wrist"]),
        (KP["left_hip"], KP["left_knee"]),
        (KP["right_hip"], KP["right_knee"]),
    ):
        t = rng.uniform(0, 1, size=(200, 1))
        pts.append(kp[a] + t * (kp[b] - kp[a]) + rng.normal(scale=0.03, size=(200, 3)))
    return kp, np.concatenate(pts)


def test_floor_fit_and_ground_scale_recover_true_depth() -> None:
    rng = np.random.default_rng(0)
    # Camera at 1.5 m height looking slightly down; floor is y_cam = 1.5 rotated by pitch.
    pitch = np.deg2rad(10)
    r = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    floor_world = np.stack([rng.uniform(-3, 3, 400), np.zeros(400), rng.uniform(3, 9, 400)], 1)
    cam = (floor_world - [0, 1.5, 0]) * [1, -1, 1]  # world y-up -> camera y-down
    cam = cam @ r.T + rng.normal(scale=0.01, size=cam.shape)
    up_hint = np.array([0, -np.cos(pitch), -np.sin(pitch)])
    plane, inliers = fit_floor_plane(cam, up_hint)
    assert inliers.mean() > 0.9
    assert np.allclose(np.abs(plane.height(cam)).mean(), 0, atol=0.02)
    # A foot reconstructed 20% too close is pushed back onto the floor.
    true_foot = cam[0]
    k = ground_scale_factors((0.8 * true_foot)[None], plane)
    assert abs(k[0] - 1.25) < 0.05
    t = world_from_camera(plane, cam[:10].mean(0))
    assert np.allclose(apply_transform(t, cam)[:, 1], 0, atol=0.03)


def test_ransac_umeyama_recovers_view_transform_with_outlier_frames() -> None:
    rng = np.random.default_rng(1)
    rot, s, t = _rot_y(73), 1.07, np.array([0.4, 0.02, -1.3])
    src, dst = [], []
    for i in range(60):
        x = rng.normal(size=(14, 3))
        y = s * x @ rot.T + t + rng.normal(scale=0.01, size=x.shape)
        if i % 6 == 0:
            y = y + rng.normal(scale=1.0, size=y.shape)  # mismatched frame
        src.append(x)
        dst.append(y)
    s_hat, r_hat, t_hat, inl = ransac_umeyama(src, dst)
    assert abs(s_hat - s) < 0.02
    assert np.allclose(r_hat, rot, atol=0.02)
    assert np.allclose(t_hat, t, atol=0.05)
    assert inl.sum() >= 48


def test_audio_offset_finds_clap() -> None:
    sr = 8000
    rng = np.random.default_rng(2)
    a = rng.normal(scale=0.01, size=sr * 20).astype(np.float32)
    b = rng.normal(scale=0.01, size=sr * 20).astype(np.float32)
    for t_a in (3.0, 9.5, 14.2):  # claps heard in A at t_a, in B at t_a + 1.37
        a[int(t_a * sr) : int(t_a * sr) + 400] += 0.8
        b[int((t_a + 1.37) * sr) : int((t_a + 1.37) * sr) + 400] += 0.5
    result = audio_offset(a, b, sr)
    assert abs(result.offset_s - 1.37) < 0.01
    assert result.confidence > 6
    assert map_frame(100, 50.0, 60.0, 1.37) == round((2.0 + 1.37) * 60)


def test_one_euro_keeps_fast_motion_and_damps_jitter() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(200) / 50.0
    still = 1.0 + rng.normal(scale=0.01, size=200)
    assert one_euro(still, 50.0)[50:].std() < still[50:].std() * 0.6
    jab = np.where(t > 2.0, np.minimum((t - 2.0) * 6.0, 0.6), 0.0)  # 6 m/s extension
    out = one_euro(jab, 50.0)
    assert out[int(2.1 * 50)] > 0.45  # follows the jab within 100 ms


def test_cluster_decimate_keeps_valid_topology() -> None:
    rng = np.random.default_rng(4)
    verts = rng.uniform(size=(500, 3))
    faces = rng.integers(0, 500, size=(1000, 3))
    rep, new_faces = cluster_decimate(verts, faces, 0.2)
    assert rep.max() < 500
    assert new_faces.max() < rep.shape[0]
    assert np.all(new_faces[:, 0] != new_faces[:, 1])


def _bout(punch_reaches_head: bool) -> tuple[dict, dict, dict, dict, float]:
    rng = np.random.default_rng(5)
    fps, n = 50.0, 60
    red_kp, red_v = _figure(np.array([0.0, 0, 0]), np.array([0.0, 0, 1]), rng)
    blue_kp, blue_v = _figure(np.array([0.0, 0, 0.95]), np.array([0.0, 0, -1]), rng)
    kps = {"red": np.repeat(red_kp[None], n, 0), "blue": np.repeat(blue_kp[None], n, 0)}
    verts = {"red": np.repeat(red_v[None], n, 0), "blue": np.repeat(blue_v[None], n, 0)}
    head = blue_kp[KP["nose"]] * 0.5 + blue_kp[KP["neck"]] * 0.5
    # Red right hand travels from guard to the head (or stops 25 cm short) in 0.1 s, then retracts.
    start = red_kp[KP["right_wrist"]].copy()
    goal = head + np.array([0.0, 0.05, 0.0]) - np.array([0, 0, 0.1 if punch_reaches_head else 0.35])
    for i in range(n):
        s = np.clip((i - 20) / 5.0, 0, 1)
        if i > 27:
            s = np.clip(1 - (i - 27) / 12.0, 0, 1)
        delta = s * (goal - start)
        idx = [KP["right_wrist"], *RIGHT_KNUCKLES]
        kps["red"][i, idx] = red_kp[idx] + delta
        kps["red"][i, KP["right_elbow"]] = red_kp[KP["right_elbow"]] + 0.6 * delta
    valid = {"red": np.ones(n, bool), "blue": np.ones(n, bool)}
    labels = {r: label_vertices(verts[r][0], kps[r][0]) for r in ("red", "blue")}
    return verts, kps, valid, labels, fps


def test_labels_cover_regions() -> None:
    _, kps, _, labels, _ = _bout(True)
    counts = np.bincount(labels["blue"], minlength=len(REGIONS))
    assert counts[REGIONS.index("head")] > 400
    assert counts[REGIONS.index("torso")] > 800


def test_straight_punch_to_head_is_landed() -> None:
    verts, kps, valid, labels, fps = _bout(True)
    cfg = ContactConfig(vertex_stride=1)
    contacts = compute_frame_contacts(("red", "blue"), verts, kps, valid, labels, fps, cfg)
    events = detect_punches(("red", "blue"), contacts, fps, cfg)
    red_right = [e for e in events if e.attacker == "red" and e.hand == "right"]
    assert len(red_right) == 1
    assert red_right[0].outcome == "landed"
    assert red_right[0].target == "head"


def test_punch_short_of_target_is_missed() -> None:
    verts, kps, valid, labels, fps = _bout(False)
    cfg = ContactConfig(vertex_stride=1)
    contacts = compute_frame_contacts(("red", "blue"), verts, kps, valid, labels, fps, cfg)
    events = detect_punches(("red", "blue"), contacts, fps, cfg)
    red_right = [e for e in events if e.attacker == "red" and e.hand == "right"]
    assert len(red_right) == 1
    assert red_right[0].outcome == "missed"


def test_fusion_prefers_classifier_but_mesh_breaks_ties() -> None:
    from boxing_analytics.mesh4d.combine import fuse, mesh_distribution

    far = {"min_gap_m": 0.30, "guard_gap_m": 0.30, "target": None}
    touching = {"min_gap_m": -0.02, "guard_gap_m": 0.20, "target": "head"}
    assert max(mesh_distribution(far, 0.09).items(), key=lambda kv: kv[1])[0] == "missed"
    assert max(mesh_distribution(touching, 0.09).items(), key=lambda kv: kv[1])[0] == "landed_head"
    confident = {
        "landed_head": 0.05,
        "landed_body": 0.02,
        "blocked": 0.03,
        "missed": 0.85,
        "no_punch": 0.05,
    }
    out = fuse(confident, mesh_distribution(touching, 0.09), 0.5)
    assert max(out, key=out.get) == "missed"
    torn = {
        "landed_head": 0.35,
        "landed_body": 0.05,
        "blocked": 0.05,
        "missed": 0.35,
        "no_punch": 0.2,
    }
    out = fuse(torn, mesh_distribution(touching, 0.09), 0.5)
    assert max(out, key=out.get) == "landed_head"
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_olympic_matching_is_one_to_one() -> None:
    from boxing_analytics.mesh4d.olympic_eval import evaluate

    gt = [
        {"start": 100, "end": 110, "label": "landed_head", "hand": "left"},
        {"start": 300, "end": 310, "label": "missed", "hand": "right"},
    ]
    events = [
        {
            "contact_frame": 104,
            "peak_frame": 102,
            "hand": "left",
            "outcome": "landed",
            "target": "head",
        },
        {
            "contact_frame": 106,
            "peak_frame": 105,
            "hand": "left",
            "outcome": "blocked",
            "target": None,
        },
        {
            "contact_frame": None,
            "peak_frame": 500,
            "hand": "right",
            "outcome": "missed",
            "target": None,
        },
    ]
    r = evaluate(gt, events)
    assert r["matched"] == 1 and r["recall"] == 0.5
    assert r["hand_accuracy"] == 1.0 and r["outcome_accuracy_4class"] == 1.0
