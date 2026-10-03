"""Regression page: predict critical temperature with every persisted model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _bootstrap  # noqa: F401,E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.inference import (  # noqa: E402
    batch_prediction_matrix,
    predict_regression,
    regression_comparison_table,
)
from core.loaders import (  # noqa: E402
    artifacts_present,
    cached_metadata,
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

setup_page("Regression — Critical Temperature", "🌡️")
repo_note()

if not artifacts_present("regression"):
    st.error(
        "No regression artifacts found in `models/regression/`. "
        "Run `python scripts/export_regression_models.py` first."
    )
    st.stop()

schema = cached_schema("regression")
meta = cached_metadata("regression")
target = meta["target"]

st.markdown(
    f"Predict **`{target}`** (Kelvin) for a superconductor composition. The same input "
    f"is run through all **{len(schema.features)}-feature** regressors. Predictions carry "
    "the notebook's physical `>= 0 K` clip."
)

tab_single, tab_batch, tab_perf = st.tabs(
    ["🔎 Single sample", "📄 Batch CSV", "📊 Test-set performance"]
)

# --------------------------------------------------------------------------- #
# Single sample
# --------------------------------------------------------------------------- #
with tab_single:
    expected_columns_expander(schema)
    values = manual_input_form(schema, key_prefix="reg")

    if st.button("Run all regression models", type="primary", key="reg_run"):
        frame = schema.frame(values)
        predictions = predict_regression(frame, schema)
        table = regression_comparison_table(predictions)

        st.subheader("Current input — predicted critical temperature")
        st.caption(
            "These are predictions for **your input**, not test-set metrics."
        )
        display = table.copy()
        display["Predicted critical_temp (K)"] = display[
            "Predicted critical_temp (K)"
        ].apply(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
        st.dataframe(display, use_container_width=True)

        valid = table["Predicted critical_temp (K)"].dropna()
        if not valid.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Models run", len(table))
            c2.metric("Mean prediction", f"{valid.mean():.2f} K")
            c3.metric("Spread (min–max)", f"{valid.min():.2f} – {valid.max():.2f} K")
            st.caption(
                "A wide spread across models is expected: the linear models and the "
                "tree ensembles disagree, which is why the notebook compares them."
            )

        with st.expander("Input values sent to the models"):
            st.dataframe(frame.T.rename(columns={0: "value"}), use_container_width=True)

# --------------------------------------------------------------------------- #
# Batch CSV
# --------------------------------------------------------------------------- #
with tab_batch:
    st.caption(
        "Columns may be in any order; the app re-orders them to the trained feature "
        "order and rejects the target column."
    )
    expected_columns_expander(schema)
    uploaded = csv_input_form(schema, key_prefix="reg")

    if uploaded is not None and st.button(
        "Predict uploaded rows", type="primary", key="reg_batch_run"
    ):
        try:
            frame = feature_matrix(uploaded, schema)
        except ValueError as exc:
            st.error(str(exc))
        else:
            predictions = predict_regression(frame, schema)
            matrix = batch_prediction_matrix(predictions, index=frame.index)
            matrix = matrix.rename(columns=lambda c: f"{c} (K)")
            st.subheader(f"Predictions for {len(frame)} row(s)")
            st.dataframe(matrix, use_container_width=True)

            failures = [p for p in predictions if p.error]
            if failures:
                st.warning(
                    "Some models failed on this input: "
                    + "; ".join(f"{p.name} ({p.error})" for p in failures)
                )

            download_button(
                matrix, "Download predictions (CSV)", "regression_predictions.csv",
                key="reg_download",
            )

# --------------------------------------------------------------------------- #
# Persisted test-set metrics
# --------------------------------------------------------------------------- #
with tab_perf:
    st.subheader("Test-set performance (persisted, not recomputed)")
    st.caption(
        "Produced by the notebook's hold-out evaluation and stored in "
        "`models/regression/metadata.json`. They describe the models, **not** your "
        "input above."
    )
    from core.figures import REGRESSION_METRICS

    table = metrics_table("regression")
    cols = [c for c in ["Model", "r2", "rmse", "mae", "cv_r2", "cv_r2_std"] if c in table]
    track_metrics_block(table[cols], REGRESSION_METRICS, n_cols=2)

    with st.expander("Model artifacts and estimators"):
        detail = table[["Model", "Artifact"]].copy()
        detail["Estimator"] = [
            e.get("estimator_class", "") for e in model_entries("regression")
        ]
        st.dataframe(detail, use_container_width=True)
