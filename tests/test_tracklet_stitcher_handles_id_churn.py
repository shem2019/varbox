from boxing_analytics.tracking.identity_hmm import TrackObservation
from boxing_analytics.tracking.tracklet_stitcher import TwoFighterTrackletStitcher


def _obs(track_id: int, cx: float, cy: float, ts: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(int(cx - 20), int(cy - 45), int(cx + 20), int(cy + 45)),
        center=(cx, cy),
        det_conf=0.9,
        timestamp_s=ts,
        mean_visibility=0.95,
    )


def test_tracklet_stitcher_handles_id_churn() -> None:
    stitcher = TwoFighterTrackletStitcher()

    left_slot_idx = None
    saw_churn = False

    for frame in range(1, 22):
        ts = frame / 10.0
        left_id = 101 if frame <= 8 else 303
        tracks = [
            _obs(left_id, 80.0 + min(frame, 8) * 1.5, 110.0, ts),
            _obs(202, 240.0 - min(frame, 8) * 1.5, 110.0, ts),
        ]
        if frame % 2 == 0:
            tracks = list(reversed(tracks))

        cand0, cand1, debug = stitcher.update(ts, tracks)
        assert cand0 is not None
        assert cand1 is not None

        if left_slot_idx is None and frame >= 4:
            left_slot_idx = 0 if cand0.center[0] < cand1.center[0] else 1

        if left_slot_idx == 0:
            assert cand0.center[0] < cand1.center[0]
        elif left_slot_idx == 1:
            assert cand1.center[0] < cand0.center[0]

        if debug["churn_events"]["last_2s"] > 0:
            saw_churn = True

    assert left_slot_idx in (0, 1)
    assert saw_churn
