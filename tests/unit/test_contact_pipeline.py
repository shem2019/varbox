from boxing_analytics.detection.pipeline import evaluate_strike


def _pose(*, nose, lw, rw, ls, rs, lh, rh):
    return {
        0: nose,
        11: ls,
        12: rs,
        15: lw,
        16: rw,
        23: lh,
        24: rh,
    }


def test_pipeline_detects_blocked_guarded() -> None:
    attacker = _pose(
        nose=(80, 60),
        lw=(145, 80),
        rw=(146, 82),
        ls=(70, 90),
        rs=(90, 90),
        lh=(72, 130),
        rh=(88, 130),
    )
    defender = _pose(
        nose=(150, 80),
        lw=(148, 82),
        rw=(152, 84),
        ls=(140, 95),
        rs=(160, 95),
        lh=(142, 135),
        rh=(158, 135),
    )

    out = evaluate_strike(
        attacker_keypoints=attacker,
        defender_keypoints=defender,
        prev_wrists={"L": (132, 80), "R": (133, 82)},
        attacker_box=(60, 50, 170, 180),
        defender_box=(120, 55, 220, 185),
    )
    assert out.label in {"blocked_guarded", "clinch"}


def test_pipeline_detects_landed_or_glancing() -> None:
    attacker = _pose(
        nose=(80, 60),
        lw=(145, 70),
        rw=(155, 76),
        ls=(70, 90),
        rs=(90, 90),
        lh=(72, 130),
        rh=(88, 130),
    )
    defender = _pose(
        nose=(150, 78),
        lw=(125, 95),
        rw=(175, 95),
        ls=(140, 95),
        rs=(160, 95),
        lh=(142, 135),
        rh=(158, 135),
    )

    out = evaluate_strike(
        attacker_keypoints=attacker,
        defender_keypoints=defender,
        prev_wrists={"L": (120, 70), "R": (120, 76)},
        attacker_box=(60, 50, 155, 180),
        defender_box=(140, 55, 235, 185),
    )
    assert out.label in {"landed_clean", "landed_glancing"}
    assert out.target_zone in {"Head", "Body"}
