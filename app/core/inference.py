"""Inference layer: run the same input through every persisted model.

The pipelines loaded here already contain their fitted preprocessing, so this
module only has to feed raw feature values in the trained column order and
collect predictions.  Nothing is ever fitted at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .loaders import load_pipeline, model_entries
from .schemas import FeatureSchema

# Probability kinds recorded by the export scripts.
PROBA_NATIVE = "predict_proba"
PROBA_DECISION = "decision_function_normalized"
PROBA_NONE = "none"


@dataclass
class ModelPrediction:
    """The outcome of one model for one or more input rows."""

    name: str
    artifact: str
    predictions: np.ndarray
    probability: np.ndarray | None = None
    decision_score: np.ndarray | None = None
    probability_kind: str = PROBA_NONE
    error: str | None = None

    @property
    def supports_probability(self) -> bool:
        return self.probability is not None and self.probability_kind == PROBA_NATIVE


def _positive_class_probability(estimator, X) -> np.ndarray | None:
    """Real class-1 probabilities when the estimator genuinely supports them."""
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = np.asarray(estimator.predict_proba(X))
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


def _decision_scores(estimator, X) -> np.ndarray | None:
    """Raw decision-function scores (NOT probabilities)."""
    if not hasattr(estimator, "decision_function"):
        return None
    scores = np.asarray(estimator.decision_function(X))
    return scores.ravel()


def predict_classification(
    frame: pd.DataFrame,
    schema: FeatureSchema,
    models: Iterable[dict[str, Any]] | None = None,
) -> list[ModelPrediction]:
    """Run every classification model over ``frame`` (already in feature order)."""
    entries = list(models) if models is not None else model_entries("classification")
    results: list[ModelPrediction] = []
    for entry in entries:
        name, artifact = entry["name"], entry["artifact"]
        try:
            pipeline = load_pipeline("classification", artifact)
            y_pred = np.asarray(pipeline.predict(frame)).ravel()
            kind = entry.get("probability_kind", PROBA_NONE)
            probability = None
            decision = None
            if kind == PROBA_NATIVE:
                probability = _positive_class_probability(pipeline, frame)
            elif kind == PROBA_DECISION:
                decision = _decision_scores(pipeline, frame)
            results.append(
                ModelPrediction(
                    name=name,
                    artifact=artifact,
                    predictions=y_pred,
                    probability=probability,
                    decision_score=decision,
                    probability_kind=kind,
                )
            )
        except Exception as exc:  # surface per-model failures instead of aborting
            results.append(
                ModelPrediction(
                    name=name,
                    artifact=artifact,
                    predictions=np.array([]),
                    probability_kind=entry.get("probability_kind", PROBA_NONE),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def predict_regression(
    frame: pd.DataFrame,
    schema: FeatureSchema,
    models: Iterable[dict[str, Any]] | None = None,
) -> list[ModelPrediction]:
    """Run every regression model over ``frame``.

    The persisted pipelines bake in the notebook's ``np.clip(pred, 0, None)``
    physical constraint, so returned values are already on the Kelvin scale and
    never negative.
    """
    entries = list(models) if models is not None else model_entries("regression")
    results: list[ModelPrediction] = []
    for entry in entries:
        name, artifact = entry["name"], entry["artifact"]
        try:
            pipeline = load_pipeline("regression", artifact)
            y_pred = np.asarray(pipeline.predict(frame)).ravel()
            results.append(
                ModelPrediction(
                    name=name,
                    artifact=artifact,
                    predictions=y_pred,
                    probability_kind=PROBA_NONE,
                )
            )
        except Exception as exc:
            results.append(
                ModelPrediction(
                    name=name,
                    artifact=artifact,
                    predictions=np.array([]),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def classification_comparison_table(predictions: list[ModelPrediction]) -> pd.DataFrame:
    """One row per model for a single input sample."""
    labels = {0: "Genuine", 1: "Fraud"}
    rows = []
    for p in predictions:
        row: dict[str, Any] = {"Model": p.name}
        if p.error:
            row.update({"Prediction": "—", "Label": "error", "Probability": None,
                        "Decision score": None, "Note": p.error})
        else:
            value = int(p.predictions[0]) if p.predictions.size else None
            row.update(
                {
                    "Prediction": value,
                    "Label": labels.get(value, str(value)),
                    "Probability": (
                        float(p.probability[0]) if p.supports_probability else None
                    ),
                    "Decision score": (
                        float(p.decision_score[0])
                        if p.decision_score is not None and p.decision_score.size
                        else None
                    ),
                    "Note": (
                        "no probability available"
                        if not p.supports_probability and p.probability_kind != PROBA_NATIVE
                        else ""
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def regression_comparison_table(predictions: list[ModelPrediction]) -> pd.DataFrame:
    """One row per model for a single input sample (predictions in Kelvin)."""
    rows = []
    for p in predictions:
        row: dict[str, Any] = {"Model": p.name}
        if p.error:
            row.update({"Predicted critical_temp (K)": None, "Note": p.error})
        else:
            row.update(
                {
                    "Predicted critical_temp (K)": (
                        float(p.predictions[0]) if p.predictions.size else None
                    ),
                    "Note": "",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def batch_prediction_matrix(predictions: list[ModelPrediction], index=None) -> pd.DataFrame:
    """All models' predictions for a batch of rows, one column per model."""
    data = {}
    for p in predictions:
        if p.error or not p.predictions.size:
            continue
        data[p.name] = p.predictions
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data, index=index)


__all__ = [
    "PROBA_NATIVE",
    "PROBA_DECISION",
    "PROBA_NONE",
    "ModelPrediction",
    "predict_classification",
    "predict_regression",
    "classification_comparison_table",
    "regression_comparison_table",
    "batch_prediction_matrix",
]
