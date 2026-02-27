"""Dataset schema and evaluation entrypoints."""

from typing import Any

from boxing_analytics.eval.schema import EvalDataset, EventSample, VideoEvalRecord, load_dataset

__all__ = ["EvalDataset", "EventSample", "VideoEvalRecord", "load_dataset", "run_evaluation"]


def run_evaluation(*, dataset_path: str, model_config_path: str) -> dict[str, Any]:
    from boxing_analytics.eval.run import run_evaluation as _run_evaluation

    return _run_evaluation(dataset_path=dataset_path, model_config_path=model_config_path)
