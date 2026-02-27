from __future__ import annotations

import json

import pytest

from boxing_analytics.eval.run import main, run_evaluation


@pytest.mark.integration
def test_eval_runner_metrics_and_report(tmp_path):
    dataset = "tests/fixtures/eval/toy_dataset.json"
    model_config = "tests/fixtures/eval/toy_model_config.json"

    metrics = run_evaluation(dataset_path=dataset, model_config_path=model_config)

    assert metrics["dataset_name"] == "toy_eval_v1"
    assert metrics["sample_count"] == 9
    assert metrics["contact_metrics"]["macro"]["f1"] == pytest.approx(0.7048)
    assert metrics["target_zone_accuracy"] == pytest.approx(0.6)
    assert metrics["id_switch_rate"] == pytest.approx(0.2857)
    assert metrics["timing_alignment_error_s"] == pytest.approx(0.155)
    assert metrics["clip_evidence_integrity"] == pytest.approx(1.0)
    assert metrics["quality_index"] == pytest.approx(0.73)

    out_path = tmp_path / "metrics.json"
    exit_code = main(
        [
            "--dataset",
            dataset,
            "--model-config",
            model_config,
            "--output",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["quality_index"] == pytest.approx(0.73)
