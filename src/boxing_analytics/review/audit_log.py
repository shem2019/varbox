"""Simple audit log event model for human review actions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class AuditEvent:
    timestamp_s: float
    actor: str
    action: str
    details: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_audit_chain(events: list[AuditEvent]) -> list[dict[str, object]]:
    """Return immutable audit records linked by hash chain."""

    prev_hash = "GENESIS"
    chain: list[dict[str, object]] = []
    for idx, event in enumerate(events):
        payload = {
            "index": idx,
            "timestamp_s": float(event.timestamp_s),
            "actor": event.actor,
            "action": event.action,
            "details": event.details,
            "prev_hash": prev_hash,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        event_hash = hashlib.sha256(encoded).hexdigest()
        chain.append(
            {
                **event.to_dict(),
                "index": idx,
                "prev_hash": prev_hash,
                "event_hash": event_hash,
            }
        )
        prev_hash = event_hash
    return chain


def final_audit_hash(events: list[AuditEvent]) -> str:
    chain = build_audit_chain(events)
    if not chain:
        return "GENESIS"
    return str(chain[-1]["event_hash"])
