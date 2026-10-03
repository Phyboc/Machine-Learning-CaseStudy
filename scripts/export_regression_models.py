"""Export fitted regression pipelines + metadata to ``models/regression/``.

Reproduces the *exact* methodology of ``src/regression/regression1.ipynb``:
superconductivity dataset, duplicate removal, |r| > 0.95 correlation filter,
``thermal_to_mass_ratio`` engineering, 80/20 split (``random_state=42``),
``StandardScaler`` fitted on train, prediction clipped at 0 K, and the same ten
estimators/hyperparameters (including the SVR and KNN tuning loops).

The predictions are clipped to >= 0 K exactly as the notebook's
``evaluate_model`` does.  Run from the repository root:

    python scripts/export_regression_models.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import (  # noqa: E402
    RANDOM_STATE,
    ClippedRegressor,
    prepare_regression,
    regression_preprocessor,
)

RAW_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
MODELS_DIR = REPO_ROOT / "models" / "regression"
RESULTS_DIR = REPO_ROOT / "results" / "regression"
NOTEBOOK = "src/regression/regression1.ipynb"

TARGET = "critical_temp"


def jsonable(value):
    """Recursively convert estimator params into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return f"<{type(value).__name__}>"


def evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test, n_splits=5):
    """Fit + score exactly like the notebook (clip predictions at 0 K).

    Metrics are computed on the *already-scaled* matrices, matching the
    notebook's ``evaluate_model`` (which receives ``X_train_scaled`` /
    ``X_test_scaled``), so the reported numbers are identical to the notebook.
    """
    model.fit(X_train_scaled, y_train)
    y_pred = np.clip(model.predict(X_test_scaled), 0, None)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="r2", n_jobs=-1)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "cv_r2": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
    }


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_regression(RAW_PATH)
    X_train, y_train = data.X_train_raw, data.y_train
    X_test, y_test = data.X_test_raw, data.y_test

    print("=" * 78)
    print("REGRESSION MODEL EXPORT (superconductivity)")
    print("=" * 78)
    print(f"train {X_train.shape} | test {X_test.shape} | features {len(data.feature_names)}")
    print(f"dropped {len(data.dropped_correlated)} correlated features | engineered: thermal_to_mass_ratio")

    def pipe(estimator):
        return Pipeline([("prep", regression_preprocessor(data.kept_features)), ("model", estimator)])

    # Fit the shared preprocessor once; tuning and metrics mirror the notebook,
    # which operates on the already-scaled matrices.
    prep = regression_preprocessor(data.kept_features).fit(X_train)
    X_train_scaled = pd.DataFrame(prep.transform(X_train), columns=data.feature_names)
    X_test_scaled = pd.DataFrame(prep.transform(X_test), columns=data.feature_names)
    print(f"scaled train {X_train_scaled.shape} | scaled test {X_test_scaled.shape}")

    specs: dict[str, tuple[str, object]] = {}
    tuned_params: dict[str, dict] = {}

    def tune(name, estimator, grid, cv=5, data_x=None, data_y=None):
        print(f"  tuning {name} ...", flush=True)
        search = GridSearchCV(estimator, grid, cv=cv, scoring="r2", n_jobs=-1)
        search.fit(X_train_scaled if data_x is None else data_x, y_train if data_y is None else data_y)
        return search

    # 1 Linear Regression
    specs["linear_regression"] = ("Linear Regression", LinearRegression())
    # 2 Ridge
    ridge = tune("Ridge", Ridge(), {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]})
    specs["ridge"] = ("Ridge", ridge.best_estimator_)
    tuned_params["ridge"] = {"alpha": ridge.best_params_["alpha"]}
    # 3 Lasso
    lasso = tune("Lasso", Lasso(max_iter=10000, random_state=RANDOM_STATE), {"alpha": [0.01, 0.1, 1.0, 10.0]})
    specs["lasso"] = ("Lasso", lasso.best_estimator_)
    tuned_params["lasso"] = {"alpha": lasso.best_params_["alpha"]}
    # 4 ElasticNet
    enet = tune("ElasticNet", ElasticNet(max_iter=10000, random_state=RANDOM_STATE),
                {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8, 0.9]})
    specs["elasticnet"] = ("ElasticNet", enet.best_estimator_)
    tuned_params["elasticnet"] = dict(enet.best_params_)
    # 5 Polynomial (degree 2 on the scaled matrix)
    specs["polynomial_regression"] = (
        "Polynomial Regression (Degree 2)",
        Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("model", LinearRegression())]),
    )
    # 6 Decision Tree
    dt = tune("Decision Tree", DecisionTreeRegressor(random_state=RANDOM_STATE),
              {"max_depth": [6, 8, 10, 12, 15, 20, None]})
    specs["decision_tree"] = ("Decision Tree", dt.best_estimator_)
    tuned_params["decision_tree"] = {"max_depth": dt.best_params_["max_depth"]}
    # 7 Random Forest (n_estimators tuned 50 vs 100)
    print("  tuning Random Forest ...", flush=True)
    best_n, best_score = 50, -np.inf
    for n in (50, 100):
        candidate = RandomForestRegressor(n_estimators=n, max_depth=15, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
        score = cross_val_score(candidate, X_train_scaled, y_train, cv=3, scoring="r2", n_jobs=-1).mean()
        if score > best_score:
            best_n, best_score = n, score
    specs["random_forest"] = ("Random Forest", RandomForestRegressor(n_estimators=best_n, max_depth=15, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1))
    tuned_params["random_forest"] = {"n_estimators": best_n, "max_depth": 15, "max_features": "sqrt"}
    # 8 Gradient Boosting (HistGradientBoosting, learning_rate tuned)
    print("  tuning Gradient Boosting ...", flush=True)
    best_lr, best_score = 0.1, -np.inf
    for lr in (0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25):
        candidate = HistGradientBoostingRegressor(learning_rate=lr, max_iter=150, min_samples_leaf=20, random_state=RANDOM_STATE)
        score = cross_val_score(candidate, X_train_scaled, y_train, cv=5, scoring="r2").mean()
        if score > best_score:
            best_lr, best_score = lr, score
    specs["gradient_boosting"] = ("Gradient Boosting", HistGradientBoostingRegressor(learning_rate=best_lr, max_iter=150, min_samples_leaf=20, random_state=RANDOM_STATE))
    tuned_params["gradient_boosting"] = {"learning_rate": best_lr, "max_iter": 150, "min_samples_leaf": 20}
    # 9 SVR (tuned on a 3000-sample subset, as in the notebook)
    print("  tuning SVR ...", flush=True)
    svr_grid = [{"kernel": ["rbf"], "C": [1.0, 10.0, 100.0]}, {"kernel": ["linear"], "C": [1.0, 10.0]}]
    rng = np.random.RandomState(RANDOM_STATE)
    sub_idx = rng.choice(len(X_train_scaled), size=3000, replace=False)
    X_sub, y_sub = X_train_scaled.iloc[sub_idx], y_train.iloc[sub_idx]
    best_params, best_score = None, -np.inf
    for params in ParameterGrid(svr_grid):
        candidate = SVR(**params)
        score = cross_val_score(candidate, X_sub, y_sub, cv=3, scoring="r2", n_jobs=-1).mean()
        if score > best_score:
            best_params, best_score = params, score
    specs["svr"] = ("Support Vector Regressor", SVR(**best_params))
    tuned_params["svr"] = best_params
    # 10 KNN Regressor (k tuned on scaled features)
    print("  tuning KNN ...", flush=True)
    best_k, best_score = 5, -np.inf
    for k in (3, 5, 7, 9, 11):
        candidate = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        score = cross_val_score(candidate, X_train_scaled, y_train, cv=3, scoring="r2", n_jobs=-1).mean()
        if score > best_score:
            best_k, best_score = k, score
    specs["knn_regressor"] = ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=best_k, n_jobs=-1))
    tuned_params["knn_regressor"] = {"n_neighbors": best_k}

    artifacts: dict[str, dict] = {}
    results_rows = []
    for slug, (display, model) in specs.items():
        print(f"  evaluating {display} ...", flush=True)
        metrics = evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
        # Deployment artifact = *fitted* preprocessing + fitted model, with the
        # notebook's physical >= 0 K clip baked in so inference matches training.
        artifact = Pipeline([
            ("prep", clone(prep)),
            ("model", ClippedRegressor(estimator=clone(model), lower=0.0)),
        ])
        artifact.fit(X_train, y_train)
        joblib.dump(artifact, MODELS_DIR / f"{slug}.joblib")
        artifacts[slug] = {
            "name": display,
            "artifact": f"{slug}.joblib",
            "metrics": metrics,
            "params": jsonable(tuned_params.get(slug, getattr(model, "get_params", lambda: {})())),
            "estimator_class": type(model.named_steps["model"]).__name__ if isinstance(model, Pipeline) else type(model).__name__,
            "probability_kind": "n/a",
        }
        results_rows.append({
            "Model": display,
            "Test R2": round(metrics["r2"], 4),
            "RMSE": round(metrics["rmse"], 4),
            "MAE": round(metrics["mae"], 4),
            "5-Fold CV R2": round(metrics["cv_r2"], 4),
            "CV R2 Std": round(metrics["cv_r2_std"], 4),
        })
        print(f"  [{display:32}] R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}")

    results_df = pd.DataFrame(results_rows).sort_values("Test R2", ascending=False).reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    print(f"\nHighest test-set R2: {best_name}")

    metadata = {
        "track": "regression",
        "dataset": "UCI Superconductivity (train.csv)",
        "source_notebook": NOTEBOOK,
        "target": TARGET,
        "target_units": "Kelvin",
        "prediction_clip": {"lower": 0.0, "note": "predictions clipped at 0 K as in the notebook"},
        "feature_count": len(data.kept_features),
        "feature_names": data.kept_features,
        "engineered_features": ["thermal_to_mass_ratio"],
        "preprocessing": {
            "steps": [
                "drop duplicate rows",
                "drop features with |corr| > 0.95",
                "add thermal_to_mass_ratio",
                "StandardScaler (fitted on train)",
            ],
            "dropped_correlated_features": data.dropped_correlated,
            "dropped_duplicate_rows": data.dropped_rows,
        },
        "split": {"test_size": 0.2, "random_state": RANDOM_STATE,
                  "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "input_space": "raw feature values (the pipeline applies the correlation filter, engineering and scaling)",
        "trained_on": "raw feature values",
        "highest_test_r2_model": best_name,
        "prediction_clip_lower": 0.0,
        "prediction_pipeline": "prep (correlation filter + thermal_to_mass_ratio + StandardScaler) -> ClippedRegressor(estimator, lower=0)",
        "models": list(artifacts.values()),
        "package_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_version": 1,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote {len(artifacts)} artifacts + metadata.json to {MODELS_DIR.relative_to(REPO_ROOT)}")
    print(results_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
