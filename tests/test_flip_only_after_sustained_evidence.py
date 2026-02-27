import numpy as np

from boxing_analytics.tracking.identity_hmm import (
    IdentityHMMConfig,
    TrackObservation,
    TwoFighterIdentityHMM,
)


def _obs(
    track_id: int,
    cx: float,
    cy: float,
    ts: float,
    emb0: float,
    emb1: float,
    vis: float = 0.95,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(int(cx - 22), int(cy - 46), int(cx + 22), int(cy + 46)),
        center=(cx, cy),
        det_conf=0.95,
        timestamp_s=ts,
        mean_visibility=vis,
        appearance_embed=np.asarray([emb0, emb1], dtype=np.float32),
    )


def test_flip_only_after_sustained_evidence() -> None:
    cfg = IdentityHMMConfig(
        commit_p_threshold=0.98,
        commit_seconds=1.0,
        commit_max_occlusion=0.10,
        w_motion=0.0,
        w_appear=1.0,
        min_flip_interval_s=0.0,
    )
    hmm = TwoFighterIdentityHMM(cfg)

    # Stable initialization.
    for frame in range(1, 21):
        ts = frame * 0.1
        hmm.update(
            ts,
            _obs(1, 80.0, 120.0, ts, 1.0, 0.0, 0.98),
            _obs(2, 230.0, 120.0, ts, 0.0, 1.0, 0.98),
        )

    switch_start_s = 2.1
    flips = []
    for frame in range(21, 71):
        ts = frame * 0.1
        result = hmm.update(
            ts,
            _obs(1, 230.0, 120.0, ts, 0.0, 1.0, 0.91),
            _obs(2, 80.0, 120.0, ts, 1.0, 0.0, 0.91),
        )
        if result["debug"]["hypothesis_changed"]:
            flips.append(ts)

    assert len(flips) == 1
    assert flips[0] - switch_start_s >= 1.0
