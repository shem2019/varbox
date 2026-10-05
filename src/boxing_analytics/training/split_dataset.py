"""Deterministic grouped splitting with explicit leakage validation."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict


def grouped_split(
    groups: list[str],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    if not groups:
        return {}
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train, validation, and test ratios must add to 1.0")
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    n_groups = len(unique)
    n_train = max(1, int(round(n_groups * train_ratio)))
    n_validation = max(1, int(round(n_groups * validation_ratio))) if n_groups >= 3 else 0
    if n_train + n_validation >= n_groups:
        n_train = max(1, n_groups - n_validation - 1)
    assignments: dict[str, str] = {}
    for index, group in enumerate(unique):
        if index < n_train:
            split = "train"
        elif index < n_train + n_validation:
            split = "validation"
        else:
            split = "test"
        assignments[group] = split
    return assignments


def validate_no_group_leakage(rows: list[dict[str, object]]) -> None:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[str(row["source_group"])].add(str(row["split"]))
    leaked = {group: splits for group, splits in seen.items() if len(splits) > 1}
    if leaked:
        rendered = ", ".join(
            f"{group}={sorted(splits)}" for group, splits in sorted(leaked.items())
        )
        raise ValueError(f"Source-group leakage detected: {rendered}")


def split_summary(assignments: dict[str, str]) -> dict[str, object]:
    counts = Counter(assignments.values())
    payload = "\n".join(f"{group}:{assignments[group]}" for group in sorted(assignments))
    return {
        "group_count": len(assignments),
        "groups_by_split": dict(sorted(counts.items())),
        "assignments": dict(sorted(assignments.items())),
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
