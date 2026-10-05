"""JSON that browsers accept: NaN and infinities become null, numpy values become plain types."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(clean(value), allow_nan=False, **kwargs)
