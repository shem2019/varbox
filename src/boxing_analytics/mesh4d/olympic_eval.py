"""Score 4D punch events against the Olympic dataset's hand labels.

Annotators labelled each punch in whichever camera showed it best, so an event with no label in
this camera may still be a real punch. Precision here is therefore a lower bound; recall, hand
accuracy and outcome accuracy on matched punches are the meaningful numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from boxing_analytics.training.annotation_schema import ORIGINAL_LABEL_MAPPING


def load_ground_truth(annotations_path: Path, frame_lo: int, frame_hi: int) -> list[dict[str, Any]]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    data = data[0] if isinstance(data, list) else data
    gt = []
    for track in data.get("tracks", []):
        shapes = [s for s in track.get("shapes", []) if not s.get("outside")]
        if not shapes or track.get("label") not in ORIGINAL_LABEL_MAPPING:
            continue
        f0 = min(s["frame"] for s in shapes)
        f1 = max(s["frame"] for s in shapes)
        if frame_lo <= f0 < frame_hi:
            five, _, hand = ORIGINAL_LABEL_MAPPING[track["label"]]
            gt.append({"start": f0, "end": f1, "label": five, "hand": hand.lower()})
    return sorted(gt, key=lambda g: g["start"])


def event_label(event: dict[str, Any]) -> str:
    if event["outcome"] == "landed":
        return "landed_body" if event.get("target") == "torso" else "landed_head"
    return str(event["outcome"])


def match(gt: list[dict[str, Any]], events: list[dict[str, Any]], slack: int = 15) -> list[tuple[int, int]]:
    """One-to-one time matching: an event matches a label when it falls inside its frames ± slack."""
    pairs = []
    for gi, g in enumerate(gt):
        mid = 0.5 * (g["start"] + g["end"])
        for ei, e in enumerate(events):
            t = int(e["contact_frame"] or e["peak_frame"])
            if g["start"] - slack <= t <= g["end"] + slack:
                pairs.append((abs(t - mid), gi, ei))
    used_g, used_e, out = set(), set(), []
    for _, gi, ei in sorted(pairs):
        if gi not in used_g and ei not in used_e:
            used_g.add(gi)
            used_e.add(ei)
            out.append((gi, ei))
    return out


def three_way(label: str) -> str:
    return "landed" if label.startswith("landed") else label


def evaluate(gt: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = match(gt, events)
    hand_ok = sum(gt[g]["hand"] == events[e]["hand"] for g, e in pairs)
    exact = sum(gt[g]["label"] == event_label(events[e]) for g, e in pairs)
    coarse = sum(three_way(gt[g]["label"]) == three_way(event_label(events[e])) for g, e in pairs)
    confusion = Counter((gt[g]["label"], event_label(events[e])) for g, e in pairs)
    n = max(1, len(pairs))
    return {
        "labelled_punches": len(gt),
        "events": len(events),
        "matched": len(pairs),
        "recall": round(len(pairs) / max(1, len(gt)), 3),
        "hand_accuracy": round(hand_ok / n, 3),
        "outcome_accuracy_4class": round(exact / n, 3),
        "outcome_accuracy_landed_blocked_missed": round(coarse / n, 3),
        "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(confusion.items())},
    }


def combine_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("labelled_punches", "events", "matched")
    total = {k: sum(r[k] for r in reports) for k in keys}
    m = max(1, total["matched"])
    total["recall"] = round(total["matched"] / max(1, total["labelled_punches"]), 3)
    for k in ("hand_accuracy", "outcome_accuracy_4class", "outcome_accuracy_landed_blocked_missed"):
        total[k] = round(sum(r[k] * r["matched"] for r in reports) / m, 3)
    conf: Counter[str] = Counter()
    for r in reports:
        conf.update(r["confusion"])
    total["confusion"] = dict(sorted(conf.items()))
    return total
