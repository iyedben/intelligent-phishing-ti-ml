"""Local-development environment loading for threat-intelligence credentials."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_local_environment(dotenv_path: str | Path | None = None) -> None:
    """Load local `.env` defaults without overriding process environment.

    No values are logged or returned. Environment variables supplied by the
    shell, IDE, or deployment environment take precedence over local file values.
    """
    load_dotenv(dotenv_path=Path(dotenv_path) if dotenv_path is not None else DOTENV_PATH, override=False)


def get_setting(name: str) -> str:
    """Read one setting after loading the local development environment."""
    load_local_environment()
    return os.getenv(name, "")
