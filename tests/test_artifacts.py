"""Artifacts load, metadata is valid, and every model can actually infer."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from core.loaders import cached_schema, raw_defaults_frame
from core.metadata import MetadataError, load_metadata, validate_metadata
from core.paths import MODELS_DIR_BY_TRACK
from core.schemas import build_schema, feature_matrix

TRACKS = ("classification", "regression")


@pytest.mark.parametrize("track", TRACKS)
def test_metadata_exists_and_is_valid(track):
    metadata = load_metadata(track)
    validate_metadata(metadata, track=track)
    assert metadata["track"] == track
    assert metadata["models"], "metadata lists no models"
    assert metadata["feature_names"], "metadata lists no features"


@pytest.mark.parametrize("track", TRACKS)
def test_every_artifact_file_exists(track):
    directory = MODELS_DIR_BY_TRACK[track]
    for entry in load_metadata(track)["models"]:
        path = directory / entry["artifact"]
        assert path.is_file(), f"missing artifact {path}"


@pytest.mark.parametrize("track", TRACKS)
def test_every_artifact_loads_and_predicts(track):
    """Each persisted pipeline must load and return finite predictions."""
    metadata = load_metadata(track)
    defaults = raw_defaults_frame(track)
    frame = defaults.loc[:, metadata["feature_names"]].head(5)

    for entry in metadata["models"]:
        pipeline = joblib.load(MODELS_DIR_BY_TRACK[track] / entry["artifact"])
        preds = np.asarray(pipeline.predict(frame)).ravel()
        assert preds.shape[0] == len(frame), entry["name"]
        assert np.all(np.isfinite(preds)), f"{entry['name']} produced non-finite output"


def test_classification_probabilities_only_where_supported(classification_metadata):
    """Native-probability models must return a valid probability; others must not."""
    defaults = raw_defaults_frame("classification")
    frame = defaults.loc[:, classification_metadata["feature_names"]].head(3)
    for entry in classification_metadata["models"]:
        pipeline = joblib.load(
            MODELS_DIR_BY_TRACK["classification"] / entry["artifact"]
        )
        kind = entry.get("probability_kind")
        if kind == "predict_proba":
            proba = pipeline.predict_proba(frame)[:, 1]
            assert np.all((proba >= 0) & (proba <= 1)), entry["name"]
        else:
            assert not hasattr(pipeline, "predict_proba") or kind in (
                "decision_function_normalized",
                "none",
            )


def test_regression_predictions_are_non_negative(regression_metadata):
    """The notebook's >= 0 K clip must be baked into the artifacts."""
    defaults = raw_defaults_frame("regression")
    frame = defaults.loc[:, regression_metadata["feature_names"]].head(25)
    for entry in regression_metadata["models"]:
        pipeline = joblib.load(MODELS_DIR_BY_TRACK["regression"] / entry["artifact"])
        preds = np.asarray(pipeline.predict(frame)).ravel()
        assert np.all(preds >= 0), f"{entry['name']} produced a negative Kelvin value"


@pytest.mark.parametrize("track", TRACKS)
def test_metrics_match_reported_values(track):
    """Re-running the persisted pipelines must reproduce the stored test metrics.

    This is the key guard against train/inference preprocessing mismatch.
    """
    from sklearn.metrics import accuracy_score, r2_score

    metadata = load_metadata(track)

    # The processed test CSVs are already scaled, so rebuild the raw split the
    # same way the export scripts did.
    from src.preprocessing import prepare_classification, prepare_regression
    from core.paths import FRAUD_RAW_PATH, SUPERCONDUCTIVITY_RAW_PATH

    if track == "classification":
        data = prepare_classification(FRAUD_RAW_PATH)
    else:
        data = prepare_regression(SUPERCONDUCTIVITY_RAW_PATH)

    for entry in metadata["models"]:
        pipeline = joblib.load(MODELS_DIR_BY_TRACK[track] / entry["artifact"])
        preds = pipeline.predict(data.X_test_raw)
        if track == "classification":
            actual = accuracy_score(data.y_test, preds)
            expected = entry["metrics"]["accuracy"]
        else:
            actual = r2_score(data.y_test, preds)
            expected = entry["metrics"]["r2"]
        assert abs(actual - expected) < 1e-6, (
            f"{track}/{entry['name']}: recomputed {actual:.6f} != stored {expected:.6f}"
        )


def test_schema_feature_order_matches_metadata():
    for track in TRACKS:
        schema = build_schema(track, raw_defaults_frame(track))
        assert schema.feature_names == load_metadata(track)["feature_names"]


def test_metadata_error_on_unknown_track():
    with pytest.raises(MetadataError):
        load_metadata("does-not-exist")
