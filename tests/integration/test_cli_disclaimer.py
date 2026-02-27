import subprocess
import sys


def test_cli_prints_disclaimer() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "boxing_analytics.app.cli", "--print-disclaimer"],
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout.lower()
    assert "decision support" in out
    assert "human judges" in out
