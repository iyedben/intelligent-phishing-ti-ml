"""Validate the project's Python version, runtime imports, and root directory."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_IMPORTS = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "shap",
    "joblib",
    "requests",
    "matplotlib",
    "seaborn",
)


def main() -> int:
    failures: list[str] = []
    version = sys.version_info
    print(f"Python detected: {version.major}.{version.minor}.{version.micro}")
    if (version.major, version.minor) != (3, 12):
        failures.append("Python 3.12 is required.")

    project_root = Path(__file__).resolve().parents[1]
    if (project_root / "README.md").is_file() and (project_root / "requirements.txt").is_file():
        print(f"Project root detected: {project_root}")
    else:
        failures.append(f"Could not identify project root from {project_root}.")

    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"Could not import {module_name}: {exc}")
        else:
            print(f"Import OK: {module_name}")

    if failures:
        print("Environment validation FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Environment validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
