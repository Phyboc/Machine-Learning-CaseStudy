"""Streamlit entry point for the Machine Learning Case Study GUI.

Run from the repository root:

    streamlit run app/main.py

The application only *loads* artifacts and results produced by the notebooks —
it never trains, tunes or re-fits a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# Make ``app/core`` and ``src`` importable regardless of the launch directory.
for path in (str(APP_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.paths import (  # noqa: E402
    CLASSIFICATION_MODELS_DIR,
    CLUSTERING_RESULTS_DIR,
    REGRESSION_MODELS_DIR,
    ensure_src_on_path,
)

ensure_src_on_path()

st.set_page_config(
    page_title="ML Case Study — GUI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import repo_note  # noqa: E402

st.title("🧪 Machine Learning Case Study — Interactive GUI")
repo_note()

st.markdown(
    """
This application is an **inference and exploration layer** on top of the models
the project's notebooks already trained. It reuses the saved pipelines, the
persisted evaluation metrics and the existing result figures, and adds an
interactive way to run inputs through every model at once.

**Use the sidebar to navigate:**

| Page | What it does |
|---|---|
| **Overview** | Project, datasets, tracks and headline results |
| **Classification** | Enter a sample → run all fraud-detection models → compare |
| **Regression** | Enter a sample → predict critical temperature across all models |
| **Clustering** | Explore the existing customer-segmentation analysis |
| **Model Comparison** | Side-by-side test-set metrics for every model |
| **Visualizations** | Gallery of the existing notebook result figures |
"""
)

st.divider()
st.subheader("Environment status")

col1, col2, col3 = st.columns(3)

try:
    from core.loaders import model_entries

    with col1:
        try:
            entries = model_entries("classification")
            st.metric("Classification models", len(entries))
            st.caption(f"Artifacts in `{CLASSIFICATION_MODELS_DIR.name}/`")
        except Exception as exc:
            st.metric("Classification models", "unavailable")
            st.caption(f"_{exc}_")

    with col2:
        try:
            entries = model_entries("regression")
            st.metric("Regression models", len(entries))
            st.caption(f"Artifacts in `{REGRESSION_MODELS_DIR.name}/`")
        except Exception as exc:
            st.metric("Regression models", "unavailable")
            st.caption(f"_{exc}_")

    with col3:
        n_images = len(list(CLUSTERING_RESULTS_DIR.glob("*.png")))
        st.metric("Clustering figures", n_images)
        st.caption("Read-only exploration of the saved analysis.")
except Exception as exc:  # pragma: no cover - defensive
    st.error(f"Could not initialise the application core: {exc}")

with st.expander("Where things live"):
    st.code(
        f"repo root              : {REPO_ROOT.as_posix()}\n"
        f"classification models  : {CLASSIFICATION_MODELS_DIR.as_posix()}\n"
        f"regression models      : {REGRESSION_MODELS_DIR.as_posix()}",
        language="text",
    )

with st.expander("Scope and limitations"):
    st.markdown(
        """
- The app performs **no training**. Every number under "test-set performance" comes
  from the notebooks' hold-out evaluation, persisted in `models/<track>/metadata.json`.
- Predictions made from your input are **separate** from those test-set metrics.
- Classification models only report a probability when the estimator genuinely
  supports `predict_proba`; margins from `decision_function` are never shown as
  probabilities.
- Regression predictions are in **Kelvin** and carry the notebook's physical
  `>= 0 K` clip.
"""
    )
