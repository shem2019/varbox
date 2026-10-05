import numpy as np
import pytest

from boxing_analytics.detection.temporal_buffer import (
    TemporalClipBuffer,
    TemporalFrameSample,
)


def _sample(timestamp_s: float) -> TemporalFrameSample:
    return TemporalFrameSample(
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp_s=timestamp_s,
        red_box=(0, 0, 4, 8),
        blue_box=(4, 0, 8, 8),
        fighter_identities={"RED": 1, "BLUE": 2},
        pose_keypoints={"RED": {}, "BLUE": {}},
        identity_confidence=0.9,
        ring_roi=((0, 0), (8, 0), (8, 8), (0, 8)),
    )


def test_buffer_waits_for_post_event_evidence_and_is_bounded() -> None:
    buffer = TemporalClipBuffer(retention_seconds=1.0)
    for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0):
        buffer.append(_sample(timestamp))
    assert not buffer.is_ready(0.9, post_event_s=0.2)
    buffer.append(_sample(1.25))
    assert buffer.is_ready(0.9, post_event_s=0.2)
    assert [row.timestamp_s for row in buffer.window(0.9, pre_event_s=0.4, post_event_s=0.3)] == [
        0.5,
        0.75,
        1.0,
    ]
    assert len(buffer) == 5


def test_buffer_rejects_out_of_order_timestamps() -> None:
    buffer = TemporalClipBuffer()
    buffer.append(_sample(1.0))
    with pytest.raises(ValueError, match="timestamp order"):
        buffer.append(_sample(0.5))
