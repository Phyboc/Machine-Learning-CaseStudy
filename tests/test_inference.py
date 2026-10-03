"""Inference layer: schemas, validation, target rejection, CSV handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.inference import (
    batch_prediction_matrix,
    classification_comparison_table,
    predict_classification,
    predict_regression,
    regression_comparison_table,
)
from core.loaders import cached_schema, model_entries, raw_defaults_frame
from core.metadata import load_metadata
from core.schemas import feature_matrix

TRACKS = ("classification", "regression")


@pytest.mark.parametrize("track", TRACKS)
def test_schema_has_expected_structure(track):
    schema = cached_schema(track)
    metadata = load_metadata(track)
    assert schema.target == metadata["target"]
    assert schema.feature_names == metadata["feature_names"]
    assert len(schema.features) == metadata["feature_count"]
    assert all(f.name for f in schema.features)
    # Every feature has a usable default.
    assert all(isinstance(f.default, (int, float)) for f in schema.features)


@pytest.mark.parametrize("track", TRACKS)
def test_schema_rejects_target_column(track):
    schema = cached_schema(track)
    frame = schema.frame()
    frame[schema.target] = 0
    problems = schema.validate(frame)
    assert any("Target column" in p for p in problems), problems
    with pytest.raises(ValueError):
        feature_matrix(frame, schema)


@pytest.mark.parametrize("track", TRACKS)
def test_schema_rejects_missing_columns(track):
    schema = cached_schema(track)
    frame = schema.frame().drop(columns=[schema.feature_names[0]])
    problems = schema.validate(frame)
    assert any("Missing required column" in p for p in problems)
    with pytest.raises(ValueError):
        feature_matrix(frame, schema)


@pytest.mark.parametrize("track", TRACKS)
def test_schema_reorders_and_warns_on_extra_columns(track):
    schema = cached_schema(track)
    frame = schema.frame()
    shuffled = frame.loc[:, list(reversed(schema.feature_names))]
    shuffled["surprise_extra"] = 1.0

    problems = schema.validate(shuffled)
    assert any("Unexpected column" in p for p in problems)
    assert not [p for p in problems if not p.startswith("Unexpected")]

    ordered = feature_matrix(shuffled, schema)
    assert list(ordered.columns) == schema.feature_names


@pytest.mark.parametrize("track", TRACKS)
def test_schema_flags_non_numeric_and_blank_values(track):
    schema = cached_schema(track)
    frame = schema.frame()
    frame[schema.feature_names[0]] = "not-a-number"
    problems = schema.validate(frame)
    assert any("Non-numeric" in p for p in problems)

    frame = schema.frame()
    frame[schema.feature_names[0]] = np.nan
    problems = schema.validate(frame)
    assert any("Missing/blank" in p for p in problems)


def test_schema_defaults_are_dataset_medians():
    """Defaults must be real dataset statistics, not invented placeholders.

    The fraud features are genuinely median-zero for many columns, so the
    meaningful invariant is that each default equals the training column's
    median and lies inside its observed range.
    """
    for track in TRACKS:
        schema = cached_schema(track)
        source = raw_defaults_frame(track)
        defaults = pd.Series(schema.defaults())
        assert defaults.notna().all()

        for spec in schema.features:
            expected = float(pd.to_numeric(source[spec.name], errors="coerce").median())
            assert spec.default == pytest.approx(round(expected, 6), abs=1e-6), spec.name
            if spec.minimum is not None and spec.maximum is not None:
                assert spec.minimum <= spec.default <= spec.maximum, spec.name


# --------------------------------------------------------------------------- #
# End-to-end inference
# --------------------------------------------------------------------------- #
def test_classification_runs_same_input_through_all_models():
    schema = cached_schema("classification")
    frame = schema.frame()
    predictions = predict_classification(frame, schema)

    expected = len(model_entries("classification"))
    assert len(predictions) == expected, "not every model was run"

    for p in predictions:
        assert p.error is None, f"{p.name} failed: {p.error}"
        assert p.predictions.size == 1
        assert int(p.predictions[0]) in (0, 1)

    table = classification_comparison_table(predictions)
    assert len(table) == expected
    assert {"Model", "Prediction", "Probability"}.issubset(table.columns)

    # Probability only where genuinely supported.
    for p in predictions:
        if p.supports_probability:
            assert p.probability is not None
            assert 0.0 <= float(p.probability[0]) <= 1.0
        else:
            assert p.probability is None


def test_classification_batch_matrix_has_one_column_per_model():
    schema = cached_schema("classification")
    frame = schema.frame(n_rows=4)
    predictions = predict_classification(frame, schema)
    matrix = batch_prediction_matrix(predictions)
    assert matrix.shape[0] == 4
    assert matrix.shape[1] == len(model_entries("classification"))


def test_regression_runs_same_input_through_all_models():
    schema = cached_schema("regression")
    frame = schema.frame()
    predictions = predict_regression(frame, schema)

    expected = len(model_entries("regression"))
    assert len(predictions) == expected

    for p in predictions:
        assert p.error is None, f"{p.name} failed: {p.error}"
        assert p.predictions.size == 1
        assert np.isfinite(p.predictions[0])
        assert p.predictions[0] >= 0.0  # >= 0 K clip

    table = regression_comparison_table(predictions)
    assert len(table) == expected
    assert "Predicted critical_temp (K)" in table.columns


def test_inference_matches_real_test_rows():
    """Smoke-test against real rows from the project's processed test split."""
    from src.preprocessing import prepare_classification, prepare_regression
    from core.paths import FRAUD_RAW_PATH, SUPERCONDUCTIVITY_RAW_PATH

    clf_schema = cached_schema("classification")
    data = prepare_classification(FRAUD_RAW_PATH)
    raw_rows = data.X_test_raw.head(3)
    predictions = predict_classification(raw_rows, clf_schema)
    assert all(p.error is None for p in predictions)

    reg_schema = cached_schema("regression")
    reg = prepare_regression(SUPERCONDUCTIVITY_RAW_PATH)
    reg_rows = reg.X_test_raw.head(3)
    reg_predictions = predict_regression(reg_rows, reg_schema)
    assert all(p.error is None for p in reg_predictions)
    assert all(np.isfinite(p.predictions).all() for p in reg_predictions)


def test_csv_round_trip_through_feature_matrix(tmp_path):
    """A CSV written from the schema validates and infers unchanged."""
    for track in TRACKS:
        schema = cached_schema(track)
        source = schema.frame(n_rows=3)
        csv_path = tmp_path / f"{track}.csv"
        source.to_csv(csv_path, index=False)

        loaded = pd.read_csv(csv_path)
        problems = schema.validate(loaded)
        assert not [p for p in problems if not p.startswith("Unexpected")], problems

        matrix = feature_matrix(loaded, schema)
        assert list(matrix.columns) == schema.feature_names
        assert matrix.shape == (3, len(schema.features))
