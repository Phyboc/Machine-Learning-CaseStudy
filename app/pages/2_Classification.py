"""Classification page: run one sample (or a CSV) through every persisted model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _bootstrap  # noqa: F401,E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.inference import (  # noqa: E402
    batch_prediction_matrix,
    classification_comparison_table,
    predict_classification,
)
from core.loaders import (  # noqa: E402
    artifacts_present,
    cached_schema,
    metrics_table,
    model_entries,
)
from core.schemas import feature_matrix  # noqa: E402
from ui import (  # noqa: E402
    csv_input_form,
    download_button,
    expected_columns_expander,
    manual_input_form,
    repo_note,
    setup_page,
    track_metrics_block,
)

setup_page("Classification — Fraud Detection", "🚨")
repo_note()

if not artifacts_present("classification"):
    st.error(
        "No classification artifacts found in `models/classification/`. "
        "Run `python scripts/export_classification_models.py` first."
    )
    st.stop()

schema = cached_schema("classification")

st.markdown(
    f"Enter one sample below, or upload a CSV of many. The **same input** is run "
    f"through all **{len(schema.features)}-feature** pipelines that the notebook trained."
)

tab_single, tab_batch, tab_perf = st.tabs(
    ["🔎 Single sample", "📄 Batch CSV", "📊 Test-set performance"]
)

# --------------------------------------------------------------------------- #
# Single sample
# --------------------------------------------------------------------------- #
with tab_single:
    expected_columns_expander(schema)
    values = manual_input_form(schema, key_prefix="clf")

    if st.button("Run all classification models", type="primary", key="clf_run"):
        frame = schema.frame(values)
        predictions = predict_classification(frame, schema)
        table = classification_comparison_table(predictions)

        st.subheader("Current input — prediction from each model")
        st.caption(
            "These are predictions for **your input**, not test-set metrics. "
            "Probability is shown only where the estimator natively supports "
            "`predict_proba`."
        )

        display = table.copy()
        if "Probability" in display:
            display["Probability"] = display["Probability"].apply(
                lambda v: "—" if pd.isna(v) else f"{v:.4f}"
            )
        if "Decision score" in display:
            display["Decision score"] = display["Decision score"].apply(
                lambda v: "—" if pd.isna(v) else f"{v:.4f}"
            )
        st.dataframe(display, use_container_width=True)

        valid = table[table["Prediction"].isin([0, 1])]
        if not valid.empty:
            fraud_votes = int((valid["Prediction"] == 1).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Models run", len(table))
            c2.metric("Flagged as fraud", f"{fraud_votes} / {len(valid)}")
            proba = valid["Probability"].dropna()
            c3.metric(
                "Mean probability (where available)",
                f"{proba.mean():.4f}" if not proba.empty else "—",
            )

        with st.expander("Input values sent to the models"):
            st.dataframe(frame.T.rename(columns={0: "value"}), use_container_width=True)

# --------------------------------------------------------------------------- #
# Batch CSV
# --------------------------------------------------------------------------- #
with tab_batch:
    st.caption(
        "Columns may be in any order; the app re-orders them to the trained feature "
        "order and rejects target columns."
    )
    expected_columns_expander(schema)
    uploaded = csv_input_form(schema, key_prefix="clf")

    if uploaded is not None and st.button(
        "Predict uploaded rows", type="primary", key="clf_batch_run"
    ):
        try:
            frame = feature_matrix(uploaded, schema)
        except ValueError as exc:
            st.error(str(exc))
        else:
            predictions = predict_classification(frame, schema)
            matrix = batch_prediction_matrix(predictions, index=frame.index)
            st.subheader(f"Predictions for {len(frame)} row(s)")
            st.dataframe(matrix, use_container_width=True)

            failures = [p for p in predictions if p.error]
            if failures:
                st.warning(
                    "Some models failed on this input: "
                    + "; ".join(f"{p.name} ({p.error})" for p in failures)
                )

            export = matrix.copy()
            for p in predictions:
                if p.supports_probability:
                    export[f"{p.name} · P(fraud)"] = p.probability
            download_button(
                export, "Download predictions (CSV)", "classification_predictions.csv",
                key="clf_download",
            )

# --------------------------------------------------------------------------- #
# Persisted test-set metrics
# --------------------------------------------------------------------------- #
with tab_perf:
    st.subheader("Test-set performance (persisted, not recomputed)")
    st.caption(
        "These metrics were produced by the notebook's hold-out evaluation and are "
        "stored in `models/classification/metadata.json`. They describe the models, "
        "**not** your input above."
    )
    from core.figures import CLASSIFICATION_METRICS

    table = metrics_table("classification")
    cols = [c for c in ["Model", "accuracy", "precision", "recall", "f1_weighted", "roc_auc"] if c in table]
    track_metrics_block(table[cols], CLASSIFICATION_METRICS, n_cols=3)

    with st.expander("Model artifacts and estimators"):
        detail = table[["Model", "Artifact"]].copy()
        detail["Estimator"] = [
            e.get("estimator_class", "") for e in model_entries("classification")
        ]
        st.dataframe(detail, use_container_width=True)
