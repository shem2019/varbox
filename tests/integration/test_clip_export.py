from pathlib import Path

import cv2
import numpy as np

from boxing_analytics.review import export_evidence_clip


def test_export_evidence_clip(tmp_path) -> None:
    input_video = tmp_path / "input.mp4"
    out_dir = tmp_path / "clips"

    writer = cv2.VideoWriter(
        str(input_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (160, 120),
    )
    for idx in range(60):
        frame = np.full((120, 160, 3), idx, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    clip_path = export_evidence_clip(
        input_video=str(input_video),
        timestamp_s=1.2,
        out_dir=str(out_dir),
        tag="red_landed",
        pre_s=0.4,
        post_s=0.4,
    )

    assert clip_path is not None
    assert Path(clip_path).exists()
