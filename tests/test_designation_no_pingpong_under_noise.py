from boxing_analytics.tracking.identity_hmm import (
    IdentityHMMConfig,
    TrackObservation,
    TwoFighterIdentityHMM,
)


def _obs(track_id: int, cx: float, cy: float, ts: float, vis: float = 0.95) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(int(cx - 22), int(cy - 46), int(cx + 22), int(cy + 46)),
        center=(cx, cy),
        det_conf=0.95,
        timestamp_s=ts,
        mean_visibility=vis,
    )


def test_designation_no_pingpong_under_noise() -> None:
    hmm = TwoFighterIdentityHMM(
        IdentityHMMConfig(viterbi_window=25, commit_seconds=1.0, commit_p_threshold=0.98)
    )

    flips = []

    # Warmup: stable hypothesis H0.
    for frame in range(1, 16):
        ts = frame * 0.1
        result = hmm.update(ts, _obs(1, 80.0, 120.0, ts), _obs(2, 230.0, 120.0, ts))
        if result["debug"]["hypothesis_changed"]:
            flips.append(ts)

    # Noise phase: alternating per-frame advantages should not satisfy sustained commitment.
    for frame in range(16, 136):
        ts = frame * 0.1
        if frame % 2 == 0:
            a = _obs(1, 80.0, 120.0, ts)
            b = _obs(2, 230.0, 120.0, ts)
        else:
            a = _obs(1, 230.0, 120.0, ts)
            b = _obs(2, 80.0, 120.0, ts)
        result = hmm.update(ts, a, b)
        if result["debug"]["hypothesis_changed"]:
            flips.append(ts)

    # No rapid oscillation allowed: at most one designation change in any 5-second window.
    assert all((later - earlier) >= 5.0 for earlier, later in zip(flips, flips[1:], strict=False))
