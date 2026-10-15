"""Preprocessing builders shared by the notebooks and the model-export scripts.

This module is the single, importable source of truth for *how* each track is
prepared.  It reproduces the original (now-removed) preprocessing notebook and
the regression notebook exactly, but packages the steps as fitted
scikit-learn objects so they can be serialized into model artifacts.

It never changes the methodology:

* Classification (fraud):  drop row-id -> drop constant cols -> drop exact
  duplicate cols -> drop duplicate rows -> 80/20 stratified split ->
  log1p skewed columns -> StandardScaler (fit on train).
* Regression (superconductivity): drop duplicate rows -> drop features with
  |r| > 0.95 -> add ``thermal_to_mass_ratio`` -> 80/20 split ->
  StandardScaler (fit on train).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .transforms import ColumnKeeper, Log1pColumns, ThermalRatioAdder, select_log1p_columns

__all__ = [
    "RANDOM_STATE",
    "TEST_SIZE",
    "ClassificationData",
    "RegressionData",
    "load_fraud_clean",
    "prepare_classification",
    "classification_preprocessor",
    "load_superconductivity_clean",
    "prepare_regression",
    "regression_preprocessor",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

CLASSIFICATION_TARGET = "targets"
REGRESSION_TARGET = "critical_temp"
CORRELATION_THRESHOLD = 0.95


# --------------------------------------------------------------------------- #
# Classification (fraud detection)
# --------------------------------------------------------------------------- #
@dataclass
class ClassificationData:
    """Container for the prepared classification matrices."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    log1p_columns: list[str]
    dropped_constant: list[str]
    dropped_duplicate: list[str]
    dropped_rows: int


def load_fraud_clean(raw_path) -> pd.DataFrame:
    """Load and clean the raw fraud dataset (drop id / constant / dup cols / dup rows)."""
    df = pd.read_csv(raw_path)
    df = df.drop(columns=[df.columns[0]])  # unnamed row-id

    constant_cols = df.columns[df.nunique() == 1].tolist()
    df = df.drop(columns=constant_cols)

    feature_cols = [c for c in df.columns if c != CLASSIFICATION_TARGET]
    dup_mask = df[feature_cols].T.duplicated(keep="first")
    dup_cols = [c for c, is_dup in zip(feature_cols, dup_mask) if is_dup]
    df = df.drop(columns=dup_cols)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df.attrs["dropped_constant"] = constant_cols
    df.attrs["dropped_duplicate"] = dup_cols
    df.attrs["dropped_rows"] = before - len(df)
    return df


def prepare_classification(raw_path) -> ClassificationData:
    """Reproduce the exact classification split + preprocessing selection.

    The returned raw (unscaled, un-logged) train/test frames are the inputs the
    pipeline will be fitted on, so that a serialized pipeline applies identical
    transforms at inference time.
    """
    df = load_fraud_clean(raw_path)

    X = df.drop(columns=[CLASSIFICATION_TARGET])
    y = df[CLASSIFICATION_TARGET]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    log_cols = select_log1p_columns(X_train_raw)

    return ClassificationData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
        log1p_columns=log_cols,
        dropped_constant=df.attrs.get("dropped_constant", []),
        dropped_duplicate=df.attrs.get("dropped_duplicate", []),
        dropped_rows=df.attrs.get("dropped_rows", 0),
    )


def classification_preprocessor(log1p_columns: list[str]) -> Pipeline:
    """Fitted-on-demand preprocessing pipeline for the classification track."""
    return Pipeline(
        steps=[
            ("log1p", Log1pColumns(columns=list(log1p_columns))),
            ("scaler", StandardScaler()),
        ]
    )


# --------------------------------------------------------------------------- #
# Regression (superconductivity)
# --------------------------------------------------------------------------- #
@dataclass
class RegressionData:
    """Container for the prepared regression matrices."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    kept_features: list[str]
    dropped_correlated: list[str]
    dropped_rows: int


def load_superconductivity_clean(raw_path) -> pd.DataFrame:
    """Load the superconductivity dataset and drop duplicate rows."""
    df = pd.read_csv(raw_path)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df.attrs["dropped_rows"] = before - len(df)
    return df


def _correlated_features(X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> list[str]:
    """Features to drop because they correlate above ``threshold`` with another feature."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col] > threshold)]


def prepare_regression(raw_path) -> RegressionData:
    """Reproduce the exact regression split, correlation filter and engineering."""
    df = load_superconductivity_clean(raw_path)

    X = df.drop(columns=[REGRESSION_TARGET])
    y = df[REGRESSION_TARGET]

    to_drop = _correlated_features(X)
    kept = [c for c in X.columns if c not in set(to_drop)]

    X_filtered = X.loc[:, kept].copy()
    X_filtered[ThermalRatioAdder.OUTPUT] = X_filtered[
        ThermalRatioAdder.NUMERATOR
    ] / (X_filtered[ThermalRatioAdder.DENOMINATOR] + ThermalRatioAdder.EPSILON)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_filtered, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return RegressionData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_filtered.columns),
        kept_features=list(kept),
        dropped_correlated=to_drop,
        dropped_rows=df.attrs.get("dropped_rows", 0),
    )


def regression_preprocessor(kept_features: list[str]) -> Pipeline:
    """Fitted-on-demand preprocessing pipeline for the regression track."""
    return Pipeline(
        steps=[
            ("keep", ColumnKeeper(columns=list(kept_features))),
            ("ratio", ThermalRatioAdder()),
            ("scaler", StandardScaler()),
        ]
    )
