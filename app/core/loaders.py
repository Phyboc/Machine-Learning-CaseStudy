"""Cached loaders for artifacts, metadata, results and datasets.

Everything the GUI needs is *read* here, never computed: model artifacts are
deserialized, metadata/CSV results are parsed, and raw rows are only summarised
to produce example input defaults.  No model is ever fitted in the application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from .metadata import load_metadata
from .paths import (
    CLASSIFICATION_PROCESSED_DIR,
    CLASSIFICATION_RESULTS_DIR,
    CLUSTERING_PROCESSED_DIR,
    CLUSTERING_RESULTS_DIR,
    FRAUD_RAW_PATH,
    MODELS_DIR_BY_TRACK,
    REGRESSION_PROCESSED_DIR,
    REGRESSION_RESULTS_DIR,
    SUPERCONDUCTIVITY_RAW_PATH,
    ensure_src_on_path,
    relative_to_repo,
)
from .schemas import FeatureSchema, build_schema

RESULTS_DIR_BY_TRACK = {
    "classification": CLASSIFICATION_RESULTS_DIR,
    "regression": REGRESSION_RESULTS_DIR,
    "clustering": CLUSTERING_RESULTS_DIR,
}

PROCESSED_DIR_BY_TRACK = {
    "classification": CLASSIFICATION_PROCESSED_DIR,
    "regression": REGRESSION_PROCESSED_DIR,
    "clustering": CLUSTERING_PROCESSED_DIR,
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
TABLE_SUFFIXES = {".csv"}
TEXT_SUFFIXES = {".txt", ".md"}


def _cache_data(func):
    """``st.cache_data`` inside a Streamlit run, a plain call elsewhere.

    Streamlit's cache decorators need an active script run context, so applying
    them unconditionally would make the loaders unusable from tests and scripts.
    """
    if st.runtime.exists():
        return st.cache_data(show_spinner=False)(func)
    return func


def _cache_resource(func):
    """``st.cache_resource`` inside a Streamlit run, a plain call elsewhere."""
    if st.runtime.exists():
        return st.cache_resource(show_spinner=False)(func)
    return func


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@_cache_data
def cached_metadata(track: str) -> dict[str, Any]:
    return load_metadata(track)


def model_entries(track: str) -> list[dict[str, Any]]:
    return list(cached_metadata(track)["models"])


def model_entry(track: str, artifact: str) -> dict[str, Any]:
    for entry in model_entries(track):
        if entry["artifact"] == artifact:
            return entry
    raise KeyError(f"No metadata entry for artifact {artifact!r} in {track!r}")


def metrics_table(track: str) -> pd.DataFrame:
    """Tidy test-set metrics for every model in a track."""
    rows = []
    for entry in model_entries(track):
        row = {"Model": entry["name"], "Artifact": entry["artifact"]}
        row.update(entry.get("metrics", {}) or {})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature schemas (example defaults from the raw data; never used for training)
# --------------------------------------------------------------------------- #
@_cache_data
def raw_defaults_frame(track: str) -> pd.DataFrame:
    """Raw, correctly-named example rows used to derive input defaults."""
    ensure_src_on_path()
    if track == "classification":
        from src.preprocessing import load_fraud_clean

        return load_fraud_clean(FRAUD_RAW_PATH)
    if track == "regression":
        from src.preprocessing import load_superconductivity_clean, prepare_regression

        # Reuse the notebook's deterministic cleaning/engineering to obtain a
        # frame with the trained feature columns (no model fitting involved).
        return prepare_regression(SUPERCONDUCTIVITY_RAW_PATH).X_train_raw
    raise ValueError(f"No schema defaults available for track {track!r}")


@_cache_data
def cached_schema(track: str) -> FeatureSchema:
    return build_schema(track, raw_defaults_frame(track))


# --------------------------------------------------------------------------- #
# Model artifacts
# --------------------------------------------------------------------------- #
@_cache_resource
def load_pipeline(track: str, artifact: str):
    """Deserialize one fitted inference pipeline (cached across reruns)."""
    path = MODELS_DIR_BY_TRACK[track] / artifact
    if not path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {relative_to_repo(path)}")
    return joblib.load(path)


@_cache_resource
def load_pipelines(track: str) -> dict[str, Any]:
    """Deserialize every persisted model for a track, keyed by display name."""
    return {
        entry["name"]: load_pipeline(track, entry["artifact"])
        for entry in model_entries(track)
    }


def artifacts_present(track: str) -> bool:
    try:
        entries = model_entries(track)
    except Exception:
        return False
    return bool(entries) and all(
        (MODELS_DIR_BY_TRACK[track] / e["artifact"]).is_file() for e in entries
    )


# --------------------------------------------------------------------------- #
# Evaluation results
# --------------------------------------------------------------------------- #
def list_result_files(track: str) -> list[Path]:
    directory = RESULTS_DIR_BY_TRACK[track]
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and not p.name.startswith("_")),
        key=lambda p: p.name,
    )


def result_files_by_kind(track: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {"images": [], "tables": [], "text": []}
    for path in list_result_files(track):
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            grouped["images"].append(path)
        elif suffix in TABLE_SUFFIXES:
            grouped["tables"].append(path)
        elif suffix in TEXT_SUFFIXES:
            grouped["text"].append(path)
    return grouped


@_cache_data
def load_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


@_cache_data
def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


@_cache_data
def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@_cache_data
def load_test_frame(track: str) -> tuple[pd.DataFrame, pd.Series]:
    """Hold-out split persisted by the notebooks (already preprocessed)."""
    directory = PROCESSED_DIR_BY_TRACK[track]
    X = pd.read_csv(directory / "X_test.csv")
    y = pd.read_csv(directory / "y_test.csv").iloc[:, 0]
    return X, y


# --------------------------------------------------------------------------- #
# Clustering (read-only results)
# --------------------------------------------------------------------------- #
@_cache_data
def clustering_X() -> pd.DataFrame:
    return pd.read_csv(CLUSTERING_PROCESSED_DIR / "X.csv")


@_cache_data
def clustering_assignments() -> pd.Series:
    return pd.read_csv(CLUSTERING_PROCESSED_DIR / "cluster_assignments.csv").iloc[:, 0]


@_cache_data
def clustering_satisfaction() -> pd.Series:
    return pd.read_csv(CLUSTERING_PROCESSED_DIR / "satisfaction.csv").iloc[:, 0]


__all__ = [
    "cached_metadata",
    "model_entries",
    "model_entry",
    "metrics_table",
    "raw_defaults_frame",
    "cached_schema",
    "load_pipeline",
    "load_pipelines",
    "artifacts_present",
    "list_result_files",
    "result_files_by_kind",
    "load_table",
    "load_text",
    "load_json",
    "load_test_frame",
    "clustering_X",
    "clustering_assignments",
    "clustering_satisfaction",
]
