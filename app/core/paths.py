"""Filesystem paths for the Streamlit GUI.

The repository root is resolved from this file's location (``app/core/paths.py``)
so the application works regardless of the process working directory, which is
what ``streamlit run app/main.py`` relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path

# app/core/paths.py -> app/core -> app -> <repo root>
APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SRC_DIR = REPO_ROOT / "src"

# Track-specific locations --------------------------------------------------- #
CLASSIFICATION_MODELS_DIR = MODELS_DIR / "classification"
REGRESSION_MODELS_DIR = MODELS_DIR / "regression"

CLASSIFICATION_RESULTS_DIR = RESULTS_DIR / "classification"
REGRESSION_RESULTS_DIR = RESULTS_DIR / "regression"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"

CLASSIFICATION_PROCESSED_DIR = PROCESSED_DIR / "classification"
REGRESSION_PROCESSED_DIR = PROCESSED_DIR / "regression"
CLUSTERING_PROCESSED_DIR = PROCESSED_DIR / "clustering"

MODELS_DIR_BY_TRACK = {
    "classification": CLASSIFICATION_MODELS_DIR,
    "regression": REGRESSION_MODELS_DIR,
}

RESULTS_DIR_BY_TRACK = {
    "classification": CLASSIFICATION_RESULTS_DIR,
    "regression": REGRESSION_RESULTS_DIR,
    "clustering": CLUSTERING_RESULTS_DIR,
}

# Raw sources (used only to derive input defaults, never to retrain) --------- #
FRAUD_RAW_PATH = RAW_DIR / "fraud_detection_bank_dataset_raw.csv"
SUPERCONDUCTIVITY_RAW_PATH = RAW_DIR / "train.csv"


def ensure_src_on_path() -> Path:
    """Make ``src.preprocessing`` importable from inside the app."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def relative_to_repo(path: Path | str) -> str:
    """Return a repo-relative POSIX-style path for display."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "APP_DIR",
    "REPO_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "SRC_DIR",
    "CLASSIFICATION_MODELS_DIR",
    "REGRESSION_MODELS_DIR",
    "CLASSIFICATION_RESULTS_DIR",
    "REGRESSION_RESULTS_DIR",
    "CLUSTERING_RESULTS_DIR",
    "CLASSIFICATION_PROCESSED_DIR",
    "REGRESSION_PROCESSED_DIR",
    "CLUSTERING_PROCESSED_DIR",
    "MODELS_DIR_BY_TRACK",
    "RESULTS_DIR_BY_TRACK",
    "FRAUD_RAW_PATH",
    "SUPERCONDUCTIVITY_RAW_PATH",
    "ensure_src_on_path",
    "relative_to_repo",
]
