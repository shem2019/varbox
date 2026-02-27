"""Review edit-session persistence for expert workflows."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def correction_digest(corrections: list[dict[str, object]]) -> str:
    payload = json.dumps(corrections, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewEditSession:
    created_utc: str
    decision_support_notice: str
    input_video: str
    output_video: str
    metadata_path: str
    ref_events: list[dict[str, object]]
    manual_corrections: list[dict[str, object]]
    correction_digest: str
    audit_events: list[dict[str, object]]
    final_audit_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReviewEditSession:
        ref_events_raw = payload.get("ref_events", [])
        corrections_raw = payload.get("manual_corrections", [])
        audit_raw = payload.get("audit_events", [])
        ref_events = (
            [dict(row) for row in ref_events_raw if isinstance(row, dict)]
            if isinstance(ref_events_raw, list)
            else []
        )
        corrections = (
            [dict(row) for row in corrections_raw if isinstance(row, dict)]
            if isinstance(corrections_raw, list)
            else []
        )
        audit_events = (
            [dict(row) for row in audit_raw if isinstance(row, dict)]
            if isinstance(audit_raw, list)
            else []
        )

        created_utc = str(payload.get("created_utc", "")).strip() or _utc_now_iso()
        decision_support_notice = str(payload.get("decision_support_notice", "")).strip()
        input_video = str(payload.get("input_video", "")).strip()
        output_video = str(payload.get("output_video", "")).strip()
        metadata_path = str(payload.get("metadata_path", "")).strip()
        final_hash = str(payload.get("final_audit_hash", "")).strip() or "GENESIS"
        stored_digest = str(payload.get("correction_digest", "")).strip()
        computed_digest = correction_digest(corrections)
        digest = stored_digest or computed_digest
        if stored_digest and stored_digest != computed_digest:
            raise ValueError("Review session correction_digest mismatch")

        return cls(
            created_utc=created_utc,
            decision_support_notice=decision_support_notice,
            input_video=input_video,
            output_video=output_video,
            metadata_path=metadata_path,
            ref_events=ref_events,
            manual_corrections=corrections,
            correction_digest=digest,
            audit_events=audit_events,
            final_audit_hash=final_hash,
        )


def build_review_session(
    *,
    input_video: str,
    output_video: str,
    metadata_path: str,
    ref_events: list[dict[str, object]],
    manual_corrections: list[dict[str, object]],
    audit_events: list[dict[str, object]],
    final_audit_hash: str,
) -> ReviewEditSession:
    decision_support_notice = (
        "Assistive judging/analytics only. Human judge/referee confirmation required."
    )
    return ReviewEditSession(
        created_utc=_utc_now_iso(),
        decision_support_notice=decision_support_notice,
        input_video=input_video,
        output_video=output_video,
        metadata_path=metadata_path,
        ref_events=[dict(row) for row in ref_events],
        manual_corrections=[dict(row) for row in manual_corrections],
        correction_digest=correction_digest(manual_corrections),
        audit_events=[dict(row) for row in audit_events],
        final_audit_hash=final_audit_hash,
    )


def save_review_session(session: ReviewEditSession, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(session.to_dict(), handle, indent=2)
    return output_path


def load_review_session(session_path: str) -> ReviewEditSession:
    with open(session_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Review session file must contain a JSON object")
    return ReviewEditSession.from_dict(payload)
