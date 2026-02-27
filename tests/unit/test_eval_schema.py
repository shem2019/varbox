from __future__ import annotations

from boxing_analytics.eval.schema import load_dataset


def test_load_dataset_toy_fixture():
    dataset = load_dataset("tests/fixtures/eval/toy_dataset.json")

    assert dataset.dataset_name == "toy_eval_v1"
    assert len(dataset.videos) == 1
    video = dataset.videos[0]
    assert video.video_id == "toy_bout_001"
    assert len(video.samples) == 9
    assert video.samples[0].ground_truth_label == "landed_clean"
    assert video.samples[0].predicted_label == "landed_clean"
    assert len(video.round_markers_gt) == 2
    assert len(video.ref_events) == 1
