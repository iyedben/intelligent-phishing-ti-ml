"""Keep developer credentials isolated from the test suite."""

from __future__ import annotations

import pytest
from pathlib import Path

from src.threat_intel import config


@pytest.fixture(autouse=True)
def isolate_threat_intelligence_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never load a developer's real `.env` or inherited TI keys in tests."""
    monkeypatch.delenv("PHISHTANK_APP_KEY", raising=False)
    monkeypatch.delenv("URLHAUS_AUTH_KEY", raising=False)
    monkeypatch.setattr(config, "DOTENV_PATH", tmp_path / "no-real-credentials.env")
