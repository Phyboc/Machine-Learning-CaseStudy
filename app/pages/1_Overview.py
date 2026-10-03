"""Project overview: datasets, tracks, models and headline results."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _bootstrap  # noqa: F401,E402
import streamlit as st  # noqa: E402

from core.loaders import cached_metadata, model_entries  # noqa: E402
from ui import repo_note, setup_page  # noqa: E402

setup_page("Project Overview", "📋")
repo_note()

st.markdown(
    """
A single repository covering **three machine-learning tracks**, each developed in
its own notebook, with the trained models persisted so this GUI can run real
inference on top of them.
"""
)

# --------------------------------------------------------------------------- #
# Track cards
# --------------------------------------------------------------------------- #
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("① Classification")
    st.caption("Fraud Detection Bank Dataset")
    try:
        meta = cached_metadata("classification")
        st.metric("Models persisted", len(meta["models"]))
        st.write(f"**Target:** `{meta['target']}` (binary)")
        st.write(f"**Features:** {meta['feature_count']}")
        st.write(f"**Trained on:** {meta['split']['train_rows']:,} rows")
    except Exception as exc:
        st.warning(f"Metadata unavailable: {exc}")

with c2:
    st.subheader("② Regression")
    st.caption("UCI Superconductivity (`train.csv`)")
    try:
        meta = cached_metadata("regression")
        st.metric("Models persisted", len(meta["models"]))
        st.write(f"**Target:** `{meta['target']}` ({meta.get('target_units', 'K')})")
        st.write(f"**Features:** {meta['feature_count']}")
        st.write(f"**Trained on:** {meta['split']['train_rows']:,} rows")
    except Exception as exc:
        st.warning(f"Metadata unavailable: {exc}")

with c3:
    st.subheader("③ Clustering")
    st.caption("Airplane Customer Satisfaction")
    st.metric("Algorithms", "K-Means · Agglomerative")
    st.write("**Selected k (K-Means):** 3")
    st.write("**Features (post-encoding):** 23")
    st.write("**Method:** unsupervised segmentation, satisfaction used only post-hoc")

st.divider()

# --------------------------------------------------------------------------- #
# Headline results
# --------------------------------------------------------------------------- #
st.subheader("Headline results (test set, from the notebooks)")

left, right = st.columns(2)

with left:
    st.markdown("**Classification — top model by ROC-AUC**")
    try:
        from core.loaders import metrics_table

        table = metrics_table("classification")
        top = table.sort_values("roc_auc", ascending=False).iloc[0]
        st.metric(top["Model"], f"{top['roc_auc']:.4f} ROC-AUC", f"{top['accuracy']:.4f} accuracy")
        st.caption(
            "Ranking reflects the notebooks' own comparison; "
            "'top' here is simply the highest persisted ROC-AUC, not a claim of general superiority."
        )
    except Exception as exc:
        st.warning(f"Unavailable: {exc}")

with right:
    st.markdown("**Regression — top model by test R²**")
    try:
        from core.loaders import metrics_table

        table = metrics_table("regression")
        top = table.sort_values("r2", ascending=False).iloc[0]
        st.metric(top["Model"], f"R² {top['r2']:.4f}", f"RMSE {top['rmse']:.2f} K")
        st.caption("Same caveat: the highest persisted test R², nothing more.")
    except Exception as exc:
        st.warning(f"Unavailable: {exc}")

st.divider()

# --------------------------------------------------------------------------- #
# Methodology
# --------------------------------------------------------------------------- #
st.subheader("Methodology summary")

with st.expander("Classification pipeline", expanded=False):
    try:
        meta = cached_metadata("classification")
        st.markdown(
            f"""
**Dataset:** {meta['dataset']} · **source notebook:** `{meta['source_notebook']}`

1. Drop the unnamed row-id column, constant columns and exact duplicate columns,
   then drop duplicate rows ({meta['preprocessing']['dropped_duplicate_rows']} removed).
2. Stratified 80/20 train/test split (`random_state={meta['split']['random_state']}`).
3. `log1p` on skewed, non-negative, non-binary columns (fitted on train only).
4. `StandardScaler` (fitted on train only).
5. Compare {len(meta['models'])} models; the strongest by ROC-AUC is additionally
   hyperparameter-tuned and saved as the *champion*.

The target column `{meta['target']}` is never used as an input feature.
"""
        )
    except Exception as exc:
        st.warning(f"Unavailable: {exc}")

with st.expander("Regression pipeline", expanded=False):
    try:
        meta = cached_metadata("regression")
        st.markdown(
            f"""
**Dataset:** {meta['dataset']} · **source notebook:** `{meta['source_notebook']}`

1. Drop duplicate rows ({meta['preprocessing']['dropped_duplicate_rows']} removed).
2. Drop features with |Pearson r| > 0.95 ({len(meta['preprocessing']['dropped_correlated_features'])} removed).
3. Engineer `thermal_to_mass_ratio` = mean thermal conductivity / mean atomic mass.
4. 80/20 train/test split (`random_state={meta['split']['random_state']}`).
5. `StandardScaler` (fitted on train only).
6. Compare {len(meta['models'])} regressors; predictions are clipped at
   **0 K** because critical temperature cannot be negative.

The target column `{meta['target']}` is never used as an input feature.
"""
        )
    except Exception as exc:
        st.warning(f"Unavailable: {exc}")

with st.expander("Clustering methodology", expanded=False):
    st.markdown(
        """
**Dataset:** Airplane Customer Satisfaction · **source notebook:** `src/clustering/clustering.ipynb`

1. Encode categorical variables → 23 features; standardise.
2. Evaluate K-Means for k = 2..10 using **inertia, silhouette, Davies-Bouldin and
   Calinski-Harabasz**, all computed on the same fixed evaluation sample.
3. Select **k = 3** by the elbow with a near-optimal silhouette and balanced sizes.
4. Agglomerative clustering with a linkage comparison and a cluster-size guard
   (Ward, k = 5 under that guard).
5. PCA / t-SNE projections, dendrograms, cluster profiles.

**Satisfaction is deliberately excluded** from fitting and from model selection —
it is used only for post-hoc interpretation. The analysis documents that cluster
separation is **weak** under this representation; that limitation is preserved here
rather than overstated.
"""
    )

st.divider()

# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
st.subheader("Navigate")
st.markdown(
    """
- **Classification** — enter one sample (or upload a CSV) and run it through every
  persisted classifier at once.
- **Regression** — the same, for critical-temperature prediction in Kelvin.
- **Clustering** — read-only exploration of the saved segmentation analysis.
- **Model Comparison** — test-set metrics side by side for every model.
- **Visualizations** — gallery of the result figures the notebooks already produced.
"""
)

st.info(
    "Every prediction shown in this app comes from the **saved pipelines** — the app "
    "never retrains, retunes or rebuilds the dataset."
)
