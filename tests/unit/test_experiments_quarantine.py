from __future__ import annotations

from pathlib import Path


def test_experimental_script_quarantined():
    assert not Path("boxing_var_mediapipe.py").exists()
    assert Path("experiments/boxing_var_mediapipe.py").exists()
