"""Google sign-in through the PHP site, with a stub for Google's tokeninfo endpoint.

Boots the PHP dev server pointed at a local stub that stands in for
https://oauth2.googleapis.com/tokeninfo, then checks that a team member's token signs in, that bad
tokens and unknown accounts are refused, that the owner's password still works, and that another
member's password does not. Skipped where the `php` binary is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

requests = pytest.importorskip("requests")

PHP = shutil.which("php")
REPO = Path(__file__).resolve().parents[2]
DOCROOT = REPO / "web" / "public"

CLIENT_ID = "test-client-id.apps.googleusercontent.com"
MEMBER_EMAIL = "coach@example.com"

pytestmark = pytest.mark.skipif(PHP is None, reason="php binary not available")


def _claims_for(token: str) -> dict[str, object]:
    now = int(time.time())
    base = {
        "aud": CLIENT_ID,
        "iss": "https://accounts.google.com",
        "exp": now + 3600,
        "email_verified": "true",
        "email": MEMBER_EMAIL,
        "sub": "sub-member",
    }
    if token == "member":
        return base
    if token == "wrongaud":
        return {**base, "aud": "someone-else.apps.googleusercontent.com"}
    if token == "expired":
        return {**base, "exp": now - 10}
    if token == "unverified":
        return {**base, "email_verified": "false"}
    if token == "stranger":
        return {**base, "email": "stranger@example.com", "sub": "sub-stranger"}
    if token == "same-email-other-account":
        return {**base, "sub": "sub-other"}
    return {"error": "invalid_token"}


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (name set by BaseHTTPRequestHandler)
        token = parse_qs(urlparse(self.path).query).get("id_token", [""])[0]
        body = json.dumps(_claims_for(token)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # keep the test output quiet
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def site(tmp_path: Path) -> Iterator[str]:
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    stub_port = stub.server_address[1]
    threading.Thread(target=stub.serve_forever, daemon=True).start()

    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "config.json").write_text(
        json.dumps(
            {
                "worker_token": "worker-secret",
                "setup_code": "setupcode",
                "google_client_id": CLIENT_ID,
                "google_tokeninfo_url": f"http://127.0.0.1:{stub_port}/tokeninfo",
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
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        stub.shutdown()


def _owner(base: str) -> requests.Session:
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


def _google(base: str, token: str) -> requests.Response:
    return requests.post(
        base + "/api/login/google",
        headers={"X-VarBox": "1"},
        json={"credential": token},
    )


def test_team_member_signs_in_with_google(site: str) -> None:
    owner = _owner(site)
    added = owner.post(site + "/api/users", json={"name": "Coach", "google_email": MEMBER_EMAIL})
    assert added.status_code == 200, added.text

    session = requests.Session()
    session.headers["X-VarBox"] = "1"
    r = session.post(site + "/api/login/google", json={"credential": "member"})
    assert r.status_code == 200, r.text
    me = session.get(site + "/api/me").json()
    assert me["user"]["name"] == "Coach"
    assert me["user"]["owner"] is False

    # First sign-in links the account, which the Team page then shows.
    users = owner.get(site + "/api/users").json()["users"]
    coach = next(u for u in users if u["google_email"] == MEMBER_EMAIL)
    assert coach["google_linked"] is True


def test_wrong_audience_is_rejected(site: str) -> None:
    _owner(site)
    assert _google(site, "wrongaud").status_code == 401


def test_expired_token_is_rejected(site: str) -> None:
    _owner(site)
    assert _google(site, "expired").status_code == 401


def test_unverified_email_is_rejected(site: str) -> None:
    _owner(site)
    assert _google(site, "unverified").status_code == 401


def test_unknown_account_is_told_to_ask(site: str) -> None:
    _owner(site)
    r = _google(site, "stranger")
    assert r.status_code == 403
    assert "not on the VAR Box team" in r.json()["error"]


def test_linked_member_refuses_another_account_with_the_same_email(site: str) -> None:
    owner = _owner(site)
    owner.post(site + "/api/users", json={"name": "Coach", "google_email": MEMBER_EMAIL})
    assert _google(site, "member").status_code == 200

    r = _google(site, "same-email-other-account")
    assert r.status_code == 403
    assert "not on the VAR Box team" in r.json()["error"]

    # Setting the email again unlinks the row, so the new account can claim it.
    users = owner.get(site + "/api/users").json()["users"]
    coach = next(u for u in users if u["google_email"] == MEMBER_EMAIL)
    reset = owner.patch(site + f"/api/users/{coach['id']}", json={"google_email": MEMBER_EMAIL})
    assert reset.status_code == 200, reset.text
    assert _google(site, "same-email-other-account").status_code == 200


def test_owner_password_still_works(site: str) -> None:
    _owner(site)
    r = requests.post(
        site + "/api/login",
        headers={"X-VarBox": "1"},
        json={"email": "shem@example.com", "password": "a-long-password"},
    )
    assert r.status_code == 200, r.text


def test_other_members_password_is_refused(site: str) -> None:
    owner = _owner(site)
    owner.post(site + "/api/users", json={"name": "Coach", "google_email": MEMBER_EMAIL})
    r = requests.post(
        site + "/api/login",
        headers={"X-VarBox": "1"},
        json={"email": MEMBER_EMAIL, "password": "a-long-password"},
    )
    assert r.status_code == 401
