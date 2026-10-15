"""Reproducible preprocessing for the Machine Learning Case Study.

Importable, notebook-independent implementation of the exact preprocessing used
by the classification and regression tracks, packaged so that fitted pipelines
can be serialized into ``models/`` and reused for inference.
"""

from .builders import (
    RANDOM_STATE,
    TEST_SIZE,
    ClassificationData,
    RegressionData,
    classification_preprocessor,
    load_fraud_clean,
    load_superconductivity_clean,
    prepare_classification,
    prepare_regression,
    regression_preprocessor,
)
from .transforms import (
    ClippedRegressor,
    ColumnKeeper,
    Log1pColumns,
    ThermalRatioAdder,
    select_log1p_columns,
)

__all__ = [
    "RANDOM_STATE",
    "TEST_SIZE",
    "ClassificationData",
    "RegressionData",
    "classification_preprocessor",
    "load_fraud_clean",
    "load_superconductivity_clean",
    "prepare_classification",
    "prepare_regression",
    "regression_preprocessor",
    "ClippedRegressor",
    "ColumnKeeper",
    "Log1pColumns",
    "ThermalRatioAdder",
    "select_log1p_columns",
]
