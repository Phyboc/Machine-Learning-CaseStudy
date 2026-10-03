"""Shared pytest fixtures for the GUI/inference test suite.

The tests are read-only: they load the artifacts and results that already exist
in the repository and never train, tune or modify a model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "app")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.paths import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

TRACKS = ("classification", "regression")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def metadata(request) -> dict:
    track = getattr(request, "param", "classification")
    from core.metadata import load_metadata

    return load_metadata(track)


@pytest.fixture(scope="session")
def classification_metadata() -> dict:
    from core.metadata import load_metadata

    return load_metadata("classification")


@pytest.fixture(scope="session")
def regression_metadata() -> dict:
    from core.metadata import load_metadata

    return load_metadata("regression")
