"""Persist the regression notebook's required plots as result images.

The regression notebook rendered its predicted-vs-actual, residual, feature
importance and model-comparison plots inline, so nothing was written to
``results/regression/``.  This script recreates exactly those figures **from the
already-fitted artifacts** — it loads the persisted pipelines and predicts on the
same hold-out split; it never trains, tunes or refits a model.

Run from the repository root:

    python scripts/export_regression_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import prepare_regression  # noqa: E402

RAW_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
MODELS_DIR = REPO_ROOT / "models" / "regression"
RESULTS_DIR = REPO_ROOT / "results" / "regression"

plt.style.use("seaborn-v0_8-whitegrid")


def _feature_importance(pipeline, feature_names: list[str]) -> pd.Series | None:
    """Extract importances from the tree-based model inside an artifact."""
    model = pipeline.named_steps.get("model")
    estimator = getattr(model, "estimator_", model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))

    data = prepare_regression(RAW_PATH)
    X_test_raw, y_test = data.X_test_raw, data.y_test
    feature_names = list(data.kept_features) + ["thermal_to_mass_ratio"]

    predictions: dict[str, np.ndarray] = {}
    for entry in metadata["models"]:
        pipeline = joblib.load(MODELS_DIR / entry["artifact"])
        predictions[entry["name"]] = np.asarray(pipeline.predict(X_test_raw)).ravel()

    best_name = metadata["highest_test_r2_model"]
    best_pred = predictions[best_name]
    residuals = np.asarray(y_test) - best_pred

    # 1. Actual vs predicted ------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, best_pred, alpha=0.3, s=18, color="#1f77b4", edgecolors="none")
    lo, hi = float(np.min(y_test)), float(np.max(y_test))
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit (y = ŷ)")
    ax.set_title(f"Actual vs Predicted — {best_name}", fontweight="bold")
    ax.set_xlabel("Actual critical temperature (K)")
    ax.set_ylabel("Predicted critical temperature (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "regression_predicted_vs_actual.png", dpi=130)
    plt.close(fig)

    # 2. Residuals ----------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(best_pred, residuals, alpha=0.3, s=18, color="#9467bd", edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=2, label="Zero residual")
    ax.set_title(f"Residuals — {best_name}", fontweight="bold")
    ax.set_xlabel("Predicted critical temperature (K)")
    ax.set_ylabel("Residual (actual − predicted, K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "regression_residuals.png", dpi=130)
    plt.close(fig)

    # 3. Feature importance (tree-based) ------------------------------------ #
    rf_pipeline = joblib.load(MODELS_DIR / "random_forest.joblib")
    importance = _feature_importance(rf_pipeline, feature_names)
    if importance is not None:
        top = importance.head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top.index[::-1], top.values[::-1], color="#2a9d8f")
        ax.set_title("Random Forest — Top 15 Feature Importances", fontweight="bold")
        ax.set_xlabel("Relative importance")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "regression_feature_importance.png", dpi=130)
        plt.close(fig)

    # 4. Model comparison ---------------------------------------------------- #
    table = pd.DataFrame(
        {
            "Model": list(predictions),
            "R2": [entry["metrics"]["r2"] for entry in metadata["models"]],
            "RMSE": [entry["metrics"]["rmse"] for entry in metadata["models"]],
            "MAE": [entry["metrics"]["mae"] for entry in metadata["models"]],
            "CV R2": [entry["metrics"]["cv_r2"] for entry in metadata["models"]],
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, ascending) in zip(
        axes.ravel(), [("R2", False), ("CV R2", False), ("RMSE", True), ("MAE", True)]
    ):
        ordered = table.sort_values(col, ascending=ascending)
        ax.barh(ordered["Model"], ordered[col], color="#264653")
        ax.invert_yaxis()
        ax.set_title(col, fontweight="bold")
        for y, value in enumerate(ordered[col]):
            ax.annotate(f"{value:.3f}", (value, y), va="center", xytext=(4, 0),
                        textcoords="offset points", fontsize=8)
        ax.margins(x=0.15)
    fig.suptitle("Regression — test-set model comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "regression_model_comparison.png", dpi=130)
    plt.close(fig)

    written = sorted(p.name for p in RESULTS_DIR.glob("*.png"))
    print("Wrote figures:", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
