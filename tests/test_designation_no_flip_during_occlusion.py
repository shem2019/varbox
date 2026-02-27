from boxing_analytics.tracking.identity_hmm import (
    IdentityHMMConfig,
    TrackObservation,
    TwoFighterIdentityHMM,
)


def _obs(track_id: int, cx: float, cy: float, ts: float, vis: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(int(cx - 24), int(cy - 48), int(cx + 24), int(cy + 48)),
        center=(cx, cy),
        det_conf=0.9,
        timestamp_s=ts,
        mean_visibility=vis,
    )


def test_designation_no_flip_during_occlusion() -> None:
    cfg = IdentityHMMConfig(commit_p_threshold=0.98, commit_seconds=1.0, commit_max_occlusion=0.10)
    hmm = TwoFighterIdentityHMM(cfg)

    flips = 0

    for frame in range(1, 16):
        ts = frame * 0.1
        result = hmm.update(ts, _obs(1, 80.0, 120.0, ts, 0.95), _obs(2, 230.0, 120.0, ts, 0.95))
        flips += int(result["debug"]["hypothesis_changed"])

    uncertain_seen = False
    # High-occlusion period for 2 seconds with opposite best-state evidence.
    for frame in range(16, 36):
        ts = frame * 0.1
        result = hmm.update(
            ts,
            _obs(1, 156.0, 120.0, ts, 0.15),
            _obs(2, 160.0, 120.0, ts, 0.15),
        )
        flips += int(result["debug"]["hypothesis_changed"])
        uncertain_seen = uncertain_seen or (result["debug"]["status"] == "uncertain")

    assert flips == 0
    assert uncertain_seen
    assert result["debug"]["current_state"] == 0
