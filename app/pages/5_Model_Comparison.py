"""Model comparison: every persisted model's test-set metrics, side by side."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _bootstrap  # noqa: F401,E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.figures import CLASSIFICATION_METRICS, REGRESSION_METRICS, metric_bar_chart  # noqa: E402
from core.loaders import cached_metadata, metrics_table, model_entries  # noqa: E402
from ui import repo_note, setup_page  # noqa: E402

setup_page("Model Comparison", "⚖️")
repo_note()

st.caption(
    "All metrics below are the notebooks' **hold-out test-set** results, read from "
    "`models/<track>/metadata.json`. Nothing is recomputed here."
)

tab_clf, tab_reg = st.tabs(["🚨 Classification", "🌡️ Regression"])

with tab_clf:
    try:
        meta = cached_metadata("classification")
        table = metrics_table("classification")
    except Exception as exc:
        st.error(f"Classification metadata unavailable: {exc}")
        table = None

    if table is not None:
        st.subheader("Test-set metrics")
        cols = [c for c in ["Model", "accuracy", "precision", "recall", "f1_weighted", "roc_auc"] if c in table]
        ordered = table[cols].sort_values("roc_auc", ascending=False)
        st.dataframe(
            ordered.style.format({c: "{:.4f}" for c in cols if c != "Model"}),
            use_container_width=True,
        )

        st.subheader("Rankings")
        for key, label, higher in CLASSIFICATION_METRICS:
            fig = metric_bar_chart(table, key, label, higher)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)

        champion = meta.get("champion")
        if champion:
            st.info(
                f"The notebook's champion (selected by ROC-AUC and then tuned) is "
                f"**{champion}**. This is the project's documented selection rule, "
                "not a new claim."
            )

        with st.expander("Probability support by model"):
            rows = [
                {
                    "Model": e["name"],
                    "Probability kind": e.get("probability_kind", "n/a"),
                    "Estimator": e.get("estimator_class", ""),
                }
                for e in model_entries("classification")
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(
                "Models marked `decision_function_normalized` do **not** provide real "
                "probabilities; the GUI shows their raw decision margin instead."
            )

with tab_reg:
    try:
        meta = cached_metadata("regression")
        table = metrics_table("regression")
    except Exception as exc:
        st.error(f"Regression metadata unavailable: {exc}")
        table = None

    if table is not None:
        st.subheader("Test-set metrics")
        cols = [c for c in ["Model", "r2", "rmse", "mae", "cv_r2", "cv_r2_std"] if c in table]
        ordered = table[cols].sort_values("r2", ascending=False)
        st.dataframe(
            ordered.style.format({c: "{:.4f}" for c in cols if c != "Model"}),
            use_container_width=True,
        )

        st.subheader("Rankings")
        for key, label, higher in REGRESSION_METRICS:
            fig = metric_bar_chart(table, key, label, higher)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)

        best = meta.get("highest_test_r2_model")
        if best:
            st.info(
                f"The notebook reports the highest test R² for **{best}**. The top two "
                "models were additionally validated with 5-fold cross-validation."
            )
