"""Export fitted classification pipelines + metadata to ``models/classification/``.

This script reproduces the *exact* methodology of
``src/classification/classification.ipynb`` (same estimators, same
hyperparameters, same tuning rules, same metrics) and packages each fitted model
together with its preprocessing into a single serializable scikit-learn
``Pipeline``.

It does not change the methodology: the numbers printed here are the notebook's
numbers.  Run from the repository root:

    python scripts/export_classification_models.py
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
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import (  # noqa: E402
    RANDOM_STATE,
    classification_preprocessor,
    prepare_classification,
)

RAW_PATH = REPO_ROOT / "data" / "raw" / "fraud_detection_bank_dataset_raw.csv"
MODELS_DIR = REPO_ROOT / "models" / "classification"
RESULTS_DIR = REPO_ROOT / "results" / "classification"
NOTEBOOK = "src/classification/classification.ipynb"


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
    # Nested estimator (e.g. BaggingClassifier's base estimator): record its class.
    return f"<{type(value).__name__}>"


def score_model(model, X_test, y_test):
    """Compute the notebook's mandatory metrics for a fitted classifier."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        proba_kind = "predict_proba"
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        proba_kind = "decision_function_normalized"
    else:
        y_prob = y_pred
        proba_kind = "none"

    return (
        {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        },
        proba_kind,
    )


def build_estimators():
    """The ten baseline estimators exactly as defined in the notebook."""
    return {
        "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "gaussian_nb": ("Naive Bayes (Gaussian)", GaussianNB()),
        "decision_tree": ("Decision Tree", DecisionTreeClassifier(max_depth=7, random_state=RANDOM_STATE)),
        "svm": ("Support Vector Machine", SVC(C=1.0, kernel="rbf", probability=False, random_state=RANDOM_STATE)),
        "random_forest": ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)),
        "adaboost": ("AdaBoost", AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)),
        "gradient_boosting": ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE)),
        "bagging": ("Bagging (Decision Tree)", BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE), n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)),
        "mlp": ("MLP Neural Network", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=400, random_state=RANDOM_STATE)),
    }


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_classification(RAW_PATH)

    print("=" * 78)
    print("CLASSIFICATION MODEL EXPORT")
    print("=" * 78)
    print(f"train {data.X_train_raw.shape} | test {data.X_test_raw.shape} | features {len(data.feature_names)}")
    print(f"log1p columns: {len(data.log1p_columns)}")

    # ---- KNN tuning (k by test ROC-AUC, as in the notebook) ----
    knn_rows = []
    for k in [3, 5, 7, 9, 11]:
        pipe = Pipeline([("prep", classification_preprocessor(data.log1p_columns)),
                         ("model", KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2))])
        pipe.fit(data.X_train_raw, data.y_train)
        probs = pipe.predict_proba(data.X_test_raw)[:, 1]
        knn_rows.append({"k": k, "roc_auc": roc_auc_score(data.y_test, probs)})
    best_k = int(pd.DataFrame(knn_rows).loc[lambda d: d["roc_auc"].idxmax(), "k"])
    print(f"KNN best k = {best_k}")

    estimators = build_estimators()
    estimators["knn_classifier"] = ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=2))

    artifacts: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    metrics_table: dict[str, dict] = {}

    for slug, (display, est) in estimators.items():
        pipe = Pipeline([("prep", classification_preprocessor(data.log1p_columns)), ("model", est)])
        pipe.fit(data.X_train_raw, data.y_train)
        metrics, proba_kind = score_model(pipe, data.X_test_raw, data.y_test)
        fitted[display] = pipe
        metrics_table[display] = metrics
        artifacts[slug] = {
            "name": display,
            "artifact": f"{slug}.joblib",
            "metrics": metrics,
            "probability_kind": proba_kind,
            "estimator_class": type(est).__name__,
            "params": jsonable(est.get_params()),
        }
        print(f"  [{display:24}] acc={metrics['accuracy']:.4f} roc_auc={metrics['roc_auc']:.4f} proba={proba_kind}")

    # ---- Champion selection: top-2 by ROC-AUC -> GridSearchCV (as in the notebook) ----
    ranking = sorted(metrics_table.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)
    best_name, runner_up_name = ranking[0][0], ranking[1][0]
    print(f"\nTop performer: {best_name} | runner-up: {runner_up_name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name in (best_name, runner_up_name):
        scores = cross_val_score(fitted[name], data.X_train_raw, data.y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  5-fold CV ROC-AUC {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    if "Random Forest" in best_name:
        grid = {"n_estimators": [100, 200], "max_depth": [10, 15], "min_samples_split": [2, 5]}
        base = RandomForestClassifier(random_state=RANDOM_STATE)
    elif "Gradient Boosting" in best_name:
        grid = {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif "MLP" in best_name:
        grid = {"hidden_layer_sizes": [(64, 32), (100, 50)], "activation": ["relu", "tanh"]}
        base = MLPClassifier(max_iter=400, random_state=RANDOM_STATE)
    else:
        grid = {"n_estimators": [50, 100, 150], "learning_rate": [0.05, 0.1]}
        base = AdaBoostClassifier(random_state=RANDOM_STATE)

    champion_pipe = Pipeline([("prep", classification_preprocessor(data.log1p_columns)), ("model", base)])
    search = GridSearchCV(champion_pipe, {f"model__{k}": v for k, v in grid.items()}, cv=3, scoring="roc_auc", n_jobs=-1)
    search.fit(data.X_train_raw, data.y_train)
    champion_metrics, champion_proba = score_model(search.best_estimator_, data.X_test_raw, data.y_test)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"Champion tuned params: {best_params}")
    print(f"Champion metrics: acc={champion_metrics['accuracy']:.4f} roc_auc={champion_metrics['roc_auc']:.4f}")

    joblib.dump(search.best_estimator_, MODELS_DIR / "champion_gradient_boosting.joblib")
    artifacts["champion_gradient_boosting"] = {
        "name": f"{best_name} (Hyperparameter Tuned)",
        "artifact": "champion_gradient_boosting.joblib",
        "metrics": champion_metrics,
        "probability_kind": champion_proba,
        "estimator_class": type(search.best_estimator_.named_steps["model"]).__name__,
        "params": jsonable(best_params),
        "is_champion": True,
    }

    # ---- Persist every baseline artifact ----
    for slug, info in artifacts.items():
        if slug == "champion_gradient_boosting":
            continue
        joblib.dump(fitted[info["name"]], MODELS_DIR / info["artifact"])

    # ---- Benchmark CSV (same columns as the notebook's consolidated summary) ----
    rows = []
    for slug, info in artifacts.items():
        m = info["metrics"]
        rows.append({
            "Algorithm": info["name"],
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score (Weighted)": round(m["f1_weighted"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
        })
    benchmark = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    benchmark.to_csv(RESULTS_DIR / "gui_export_benchmark.csv", index=False)

    # ---- Metadata ----
    metadata = {
        "track": "classification",
        "dataset": "Fraud Detection Bank Dataset",
        "source_notebook": NOTEBOOK,
        "target": "targets",
        "target_labels": {"0": "Genuine", "1": "Fraud"},
        "feature_count": len(data.feature_names),
        "feature_names": data.feature_names,
        "preprocessing": {
            "steps": ["log1p (skewed non-negative non-binary columns, fitted on train)", "StandardScaler (fitted on train)"],
            "log1p_columns": data.log1p_columns,
            "dropped_constant_columns": data.dropped_constant,
            "dropped_duplicate_columns": data.dropped_duplicate,
            "dropped_duplicate_rows": data.dropped_rows,
        },
        "split": {"test_size": 0.2, "random_state": RANDOM_STATE, "stratified": True,
                  "train_rows": int(len(data.X_train_raw)), "test_rows": int(len(data.X_test_raw))},
        "input_space": "raw feature values (the pipeline applies log1p + scaling)",
        "trained_on": "raw feature values",
        "champion": best_name,
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
    print(benchmark.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
