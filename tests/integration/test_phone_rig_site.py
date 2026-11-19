"""End-to-end test of the phone upload path through the PHP site.

Boots the PHP dev server against a throwaway storage directory and walks a recording from a phone's
token through two camera uploads to the pending task, the email log, Process, and a worker claim
that carries the sidecars. Skipped where the `php` binary is unavailable (for example in CI).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

PHP = shutil.which("php")
REPO = Path(__file__).resolve().parents[2]
DOCROOT = REPO / "web" / "public"

pytestmark = pytest.mark.skipif(PHP is None, reason="php binary not available")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def site(tmp_path: Path) -> Iterator[dict[str, object]]:
    storage = tmp_path / "storage"
    storage.mkdir()
    worker_token = "worker-secret-token"
    (storage / "config.json").write_text(
        json.dumps(
            {
                "worker_token": worker_token,
                "setup_code": "setupcode",
                "site_url": "https://varbox.example",
                "mail": {},
            }
        ),
        encoding="utf-8",
    )
    port = _free_port()
    env = {**os.environ, "VARBOX_STORAGE": str(storage)}
    proc = subprocess.Popen(
        [str(PHP), "-S", f"127.0.0.1:{port}", "-t", str(DOCROOT), str(DOCROOT / "index.php")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                requests.get(base + "/login", timeout=1)
                break
            except requests.ConnectionError:
                time.sleep(0.05)
        else:
            raise RuntimeError("php dev server did not start")
        yield {"base": base, "storage": storage, "worker_token": worker_token}
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _owner_session(base: str) -> requests.Session:
    s = requests.Session()
    s.headers["X-VarBox"] = "1"
    r = s.post(
        base + "/api/setup",
        json={
            "code": "setupcode",
            "email": "shem@example.com",
            "name": "Shem",
            "password": "a-long-password",
        },
    )
    assert r.status_code == 200, r.text
    return s


def _upload_camera(base: str, token: str, rec_uid: str, letter: str, frames_ms: list[int]) -> int:
    phone = {"Authorization": f"Bearer {token}"}
    body = b"\x00" * 2048  # stand-in for the video bytes
    r = requests.post(
        base + "/api/phone/uploads",
        headers=phone,
        json={
            "rec_uid": rec_uid,
            "letter": letter,
            "name": f"r1_{letter}.mp4",
            "size": len(body),
            "fps": 30,
            "expected_cameras": 2,
            "device": f"Pixel {letter}",
            "title": "Sparring, round 1",
        },
    )
    assert r.status_code == 200, r.text
    upload_id = int(r.json()["upload"]["id"])
    r = requests.put(
        base + f"/api/phone/uploads/{upload_id}?offset=0",
        headers={**phone, "Content-Type": "application/octet-stream"},
        data=body,
    )
    assert r.status_code == 200 and r.json()["status"] == "ready", r.text
    sidecar = {
        "device": f"Pixel {letter}",
        "settings": {"fps": 30, "width": 1920, "height": 1080},
        "clock_offset_ms": 12.0,
        "clock_uncertainty_ms": 3.0,
        "frames": frames_ms,
    }
    r = requests.put(
        base + f"/api/phone/uploads/{upload_id}/sidecar",
        headers={**phone, "Content-Type": "application/json"},
        data=json.dumps(sidecar),
    )
    assert r.status_code == 200, r.text
    return upload_id


def test_phone_recording_becomes_pending_task_and_worker_claim_has_sidecars(
    site: dict[str, object],
) -> None:
    base = str(site["base"])
    storage = site["storage"]
    assert isinstance(storage, Path)
    owner = _owner_session(base)

    # The dashboard adds a phone; the token rides inside the pairing QR payload.
    r = owner.post(base + "/api/phones", json={"name": "Ringside A"})
    assert r.status_code == 200, r.text
    pairing = json.loads(r.json()["pairing"])
    token = pairing["token"]
    phone_id = r.json()["phone"]["id"]

    # Two cameras of one take arrive; the second completes the recording.
    rec_uid = "rec-123"
    frames_a = [1000 + i * 33 for i in range(30)]
    frames_b = [1005 + i * 33 for i in range(30)]
    _upload_camera(base, token, rec_uid, "A", frames_a)
    jobs = owner.get(base + "/api/jobs").json()["jobs"]
    assert not jobs, "no task until every expected camera has arrived"

    _upload_camera(base, token, rec_uid, "B", frames_b)
    jobs = owner.get(base + "/api/jobs").json()["jobs"]
    assert len(jobs) == 1, jobs
    job = jobs[0]
    assert job["status"] == "pending"
    assert job["camera_b"] is not None

    # Mail is unconfigured in the test, so the message is logged rather than sent.
    log = (storage / "mail.log").read_text(encoding="utf-8")
    assert "ready to process" in log and "not configured" in log

    # Workers claim only queued jobs; pressing Process queues this one.
    worker_headers = {
        "Authorization": f"Bearer {site['worker_token']}",
        "Content-Type": "application/json",
    }

    def claim() -> dict | None:
        r = requests.post(base + "/api/worker/claim", headers=worker_headers, json={"name": "gpu"})
        return r.json()["job"]

    assert claim() is None, "pending job is not claimable"

    r = owner.post(base + f"/api/jobs/{job['id']}/process")
    assert r.status_code == 200 and r.json()["job"]["status"] == "queued", r.text

    claimed = claim()
    assert claimed is not None
    assert claimed["camera_a"]["sidecar"]["frames"] == frames_a
    assert claimed["camera_b"]["sidecar"]["frames"] == frames_b

    # Revoking the phone stops further uploads.
    assert owner.delete(base + f"/api/phones/{phone_id}").status_code == 200
    r = requests.post(
        base + "/api/phone/uploads",
        headers={"Authorization": f"Bearer {token}"},
        json={"rec_uid": "rec-999", "letter": "A", "name": "x.mp4", "size": 10},
    )
    assert r.status_code == 401, r.text


def test_web_camera_clock_resume_and_sync_test_download(site: dict[str, object]) -> None:
    base = str(site["base"])
    owner = _owner_session(base)

    # The shared clock answers with the server's receive and reply times in milliseconds.
    before = time.time() * 1000
    clock = requests.get(base + "/api/clock").json()
    after = time.time() * 1000
    assert before - 5 <= clock["t1"] <= clock["t2"] <= after + 5

    # Adding a phone gives the camera link; the token rides in the fragment.
    r = owner.post(base + "/api/phones", json={"name": "Phone A"})
    token = json.loads(r.json()["pairing"])["token"]
    assert r.json()["camera_url"] == f"https://varbox.example/camera#{token}"
    phone = {"Authorization": f"Bearer {token}"}
    for page in ("/camera", "/sync-flash"):
        assert requests.get(base + page).status_code == 200

    # The web camera sends the sidecar first, then the video, resuming from the server's count.
    body = bytes(range(256)) * 12
    r = requests.post(
        base + "/api/phone/uploads",
        headers=phone,
        json={"rec_uid": "sync-test", "letter": "A", "size": len(body), "expected_cameras": 2},
    )
    upload_id = int(r.json()["upload"]["id"])
    sidecar = {"frame_times_source": "web-videoframe-capture", "fps": 30, "frames": [1.5, 34.8]}
    r = requests.put(
        base + f"/api/phone/uploads/{upload_id}/sidecar", headers=phone, data=json.dumps(sidecar)
    )
    assert r.status_code == 200, r.text
    status = requests.get(base + f"/api/phone/uploads/{upload_id}", headers=phone).json()
    assert status == {"received": 0, "size": len(body), "status": "receiving", "sidecar": True}

    r = requests.put(
        base + f"/api/phone/uploads/{upload_id}?offset=0", headers=phone, data=body[:1000]
    )
    assert r.json()["received"] == 1000
    # After a drop the phone asks where to carry on.
    received = requests.get(base + f"/api/phone/uploads/{upload_id}", headers=phone).json()
    assert received["received"] == 1000
    r = requests.put(
        base + f"/api/phone/uploads/{upload_id}?offset=1000", headers=phone, data=body[1000:]
    )
    assert r.json() == {"received": len(body), "status": "ready"}

    # Another phone cannot read this phone's upload.
    other = owner.post(base + "/api/phones", json={"name": "Phone B"})
    other_token = json.loads(other.json()["pairing"])["token"]
    r = requests.get(
        base + f"/api/phone/uploads/{upload_id}", headers={"Authorization": f"Bearer {other_token}"}
    )
    assert r.status_code == 403

    # The sync test fetches a take's cameras and sidecars with the worker token.
    worker = {"Authorization": f"Bearer {site['worker_token']}"}
    info = requests.get(base + "/api/worker/recordings/sync-test", headers=worker).json()
    assert [c["letter"] for c in info["cameras"]] == ["A"]
    assert info["cameras"][0]["sidecar"]["frame_times_source"] == "web-videoframe-capture"
    video = requests.get(base + f"/api/worker/uploads/{upload_id}", headers=worker)
    assert video.content == body
    assert requests.get(base + "/api/worker/recordings/nope", headers=worker).status_code == 404
    assert requests.get(base + "/api/worker/recordings/sync-test").status_code == 401


def test_rig_link_names_the_first_page_reference_and_relays_signals_once(
    site: dict[str, object],
) -> None:
    base = str(site["base"])
    owner = _owner_session(base)
    tokens = []
    for name in ("Phone A", "Phone B"):
        r = owner.post(base + "/api/phones", json={"name": name})
        tokens.append(json.loads(r.json()["pairing"])["token"])

    def poll(token: str, session: str, signals: list[dict[str, object]] | None = None) -> dict:
        r = requests.post(
            base + "/api/phone/peer",
            headers={"Authorization": f"Bearer {token}"},
            json={"session": session, "signals": signals or []},
        )
        assert r.status_code == 200, r.text
        return dict(r.json())

    a, b = "a" * 32, "b" * 32
    first = poll(tokens[0], a)
    assert first["reference"] == a and [p["name"] for p in first["peers"]] == ["Phone A"]
    second = poll(tokens[1], b, [{"to": a, "data": {"type": "offer", "sdp": "v=0"}}])
    assert second["reference"] == a
    assert [p["session"] for p in second["peers"]] == [a, b]

    # The reference receives the offer once, then its inbox is empty.
    got = poll(tokens[0], a)
    assert got["inbox"] == [{"from": b, "data": {"type": "offer", "sdp": "v=0"}}]
    assert poll(tokens[0], a)["inbox"] == []

    # Signals to unknown pages and oversize bodies are dropped; a bad session id is refused.
    poll(tokens[1], b, [{"to": "c" * 32, "data": 1}, {"to": a, "data": "x" * 20000}])
    assert poll(tokens[0], a)["inbox"] == []
    r = requests.post(
        base + "/api/phone/peer",
        headers={"Authorization": f"Bearer {tokens[0]}"},
        json={"session": "nope"},
    )
    assert r.status_code == 400
    assert requests.post(base + "/api/phone/peer", json={"session": a}).status_code == 401


def test_rig_link_drops_a_page_unseen_for_fifteen_seconds(site: dict[str, object]) -> None:
    base = str(site["base"])
    storage = site["storage"]
    assert isinstance(storage, Path)
    owner = _owner_session(base)
    r = owner.post(base + "/api/phones", json={"name": "Phone A"})
    token = json.loads(r.json()["pairing"])["token"]
    phone = {"Authorization": f"Bearer {token}"}
    a, b = "a" * 32, "b" * 32
    requests.post(base + "/api/phone/peer", headers=phone, json={"session": a})
    # Age page A by 20 s, as if its phone had gone away.
    import sqlite3

    with sqlite3.connect(storage / "varbox.sqlite") as db:
        db.execute("UPDATE peer_sessions SET seen_at = seen_at - 20, joined_at = joined_at - 20")
    out = requests.post(base + "/api/phone/peer", headers=phone, json={"session": b}).json()
    assert out["reference"] == b and [p["session"] for p in out["peers"]] == [b]
