"""Reading and validating the persisted model metadata.

Metadata lives next to the artifacts in ``models/<track>/metadata.json`` and is
the single source of truth for which models exist, what metrics they scored on
the hold-out test set, and how their preprocessing works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import MODELS_DIR_BY_TRACK

TRACKS = ("classification", "regression")


class MetadataError(RuntimeError):
    """Raised when a metadata file is missing or malformed."""


def metadata_path(track: str) -> Path:
    if track not in MODELS_DIR_BY_TRACK:
        raise MetadataError(f"Unknown track: {track!r}")
    return MODELS_DIR_BY_TRACK[track] / "metadata.json"


def load_metadata(track: str) -> dict[str, Any]:
    """Load and validate ``models/<track>/metadata.json``."""
    path = metadata_path(track)
    if not path.is_file():
        raise MetadataError(
            f"Missing metadata for {track!r} at {path}. "
            "Run the matching export script in scripts/ first."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise MetadataError(f"Invalid JSON in {path}: {exc}") from exc
    validate_metadata(data, track=track)
    return data


def validate_metadata(metadata: dict[str, Any], track: str | None = None) -> None:
    """Check the fields the GUI depends on, raising ``MetadataError`` if absent."""
    if track is not None and metadata.get("track") != track:
        raise MetadataError(
            f"Metadata track mismatch: expected {track!r}, got {metadata.get('track')!r}"
        )
    for key in ("track", "target", "feature_names", "models"):
        if key not in metadata:
            raise MetadataError(f"Metadata is missing required key: {key!r}")
    if not isinstance(metadata["feature_names"], list) or not metadata["feature_names"]:
        raise MetadataError("Metadata 'feature_names' must be a non-empty list")
    models = metadata["models"]
    if not isinstance(models, list) or not models:
        raise MetadataError("Metadata 'models' must be a non-empty list")
    for entry in models:
        for key in ("name", "artifact"):
            if key not in entry:
                raise MetadataError(f"Model entry missing {key!r}: {entry!r}")


def model_entries(track: str) -> list[dict[str, Any]]:
    """Return the metadata entries for every persisted model in a track."""
    return list(load_metadata(track)["models"])


def model_names(track: str) -> list[str]:
    return [entry["name"] for entry in model_entries(track)]


def metrics_frame(track: str):
    """Return a tidy DataFrame of test-set metrics for every model in a track.

    Imported lazily so the metadata module stays usable without pandas.
    """
    import pandas as pd

    rows = []
    for entry in model_entries(track):
        metrics = entry.get("metrics", {}) or {}
        row = {
            "Model": entry["name"],
            "Artifact": entry["artifact"],
            "Estimator": entry.get("estimator_class", ""),
            "Probability": entry.get("probability_kind", "n/a"),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "TRACKS",
    "MetadataError",
    "metadata_path",
    "load_metadata",
    "validate_metadata",
    "model_entries",
    "model_names",
    "metrics_frame",
]
