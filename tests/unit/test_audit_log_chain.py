from __future__ import annotations

from boxing_analytics.review import AuditEvent, build_audit_chain, final_audit_hash


def test_audit_chain_is_deterministic():
    events = [
        AuditEvent(timestamp_s=1.0, actor="operator", action="run", details="start"),
        AuditEvent(timestamp_s=2.0, actor="operator", action="ref_event", details="kd blue"),
    ]

    chain_a = build_audit_chain(events)
    chain_b = build_audit_chain(events)

    assert chain_a == chain_b
    assert chain_a[0]["prev_hash"] == "GENESIS"
    assert chain_a[1]["prev_hash"] == chain_a[0]["event_hash"]
    assert final_audit_hash(events) == chain_a[-1]["event_hash"]
