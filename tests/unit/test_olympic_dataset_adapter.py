import pytest

from boxing_analytics.training.annotation_schema import (
    FIVE_CLASS_LABELS,
    ORIGINAL_LABEL_MAPPING,
    AnnotationBox,
    OlympicPunchAnnotation,
)
from boxing_analytics.training.olympic_dataset_adapter import (
    SourceVideo,
    _negative_centers,
)
from boxing_analytics.training.split_dataset import (
    grouped_split,
    validate_no_group_leakage,
)


def test_all_eight_original_labels_map_to_expected_five_classes() -> None:
    assert len(ORIGINAL_LABEL_MAPPING) == 8
    mapped = {row[0] for row in ORIGINAL_LABEL_MAPPING.values()}
    assert mapped == set(FIVE_CLASS_LABELS) - {"no_punch"}
    hands = {row[2] for row in ORIGINAL_LABEL_MAPPING.values()}
    assert hands == {"LEFT", "RIGHT"}


def test_grouped_split_is_deterministic_and_has_no_leakage() -> None:
    groups = [f"bout_{index:02d}" for index in range(16)]
    first = grouped_split(groups, seed=42)
    second = grouped_split(list(reversed(groups)), seed=42)
    assert first == second
    rows = [
        {"source_group": group, "split": split} for group, split in first.items() for _ in range(3)
    ]
    validate_no_group_leakage(rows)
    assert set(first.values()) == {"train", "validation", "test"}


def test_group_leakage_validation_fails_for_shared_bout() -> None:
    with pytest.raises(ValueError, match="leakage"):
        validate_no_group_leakage(
            [
                {"source_group": "bout_01", "split": "train"},
                {"source_group": "bout_01", "split": "test"},
            ]
        )


def test_automatic_negatives_respect_event_safety_margin() -> None:
    source = SourceVideo(
        task_name="task",
        path="/tmp/not-decoded.mp4",
        source_group="bout",
        camera="cam",
        fps=10.0,
        frame_count=100,
        width=64,
        height=48,
    )
    event = OlympicPunchAnnotation(
        annotation_id="event",
        task_name="task",
        source_video=source.path,
        source_group=source.source_group,
        camera=source.camera,
        original_label="Głowa lewą ręką",
        five_class_label="landed_head",
        eight_class_label="head_left",
        hand="LEFT",
        start_frame=48,
        end_frame=52,
        peak_frame=50,
        start_time_s=4.8,
        end_time_s=5.2,
        peak_time_s=5.0,
        boxes=(AnnotationBox(50, (10.0, 10.0, 20.0, 20.0), False, False),),
    )
    negatives = _negative_centers(
        source,
        [event],
        safety_margin_s=1.0,
        clip_duration_s=0.6,
        limit=20,
        seed=42,
    )
    assert negatives
    assert all(not 3.5 <= center <= 6.5 for center, _nearest in negatives)
    assert all(nearest >= 1.0 for _center, nearest in negatives)
