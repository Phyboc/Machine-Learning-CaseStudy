"""Clustering page: read-only exploration of the saved segmentation analysis.

This is deliberately **not** a prediction form. Clustering is unsupervised, so
there is no "correct" cluster to predict for a new sample; the page surfaces the
analysis the notebook already produced instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _bootstrap  # noqa: F401,E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.loaders import (  # noqa: E402
    clustering_assignments,
    clustering_satisfaction,
    load_table,
    load_text,
    result_files_by_kind,
)
from core.paths import CLUSTERING_RESULTS_DIR  # noqa: E402
from ui import repo_note, result_gallery, setup_page  # noqa: E402

setup_page("Clustering — Customer Segmentation", "🧩")
repo_note()

st.info(
    "Clustering is **unsupervised**: this page explores the existing segmentation "
    "analysis. It does not pretend to predict a cluster for a new customer, and it "
    "does not re-run the clustering algorithms — everything shown is already saved "
    "in `results/clustering/`."
)

files = result_files_by_kind("clustering")

tab_summary, tab_selection, tab_agglom, tab_profiles, tab_sat, tab_figs = st.tabs(
    [
        "📝 Report",
        "📐 K-Means selection",
        "🔗 Agglomerative",
        "🧬 Cluster profiles",
        "🎯 Satisfaction (post-hoc)",
        "🖼️ Figures",
    ]
)

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
with tab_summary:
    report = CLUSTERING_RESULTS_DIR / "clustering_report.txt"
    if report.is_file():
        st.code(load_text(report), language="text")
    else:
        st.warning("`clustering_report.txt` not found.")
    st.caption(
        "The report documents that separation is relatively **weak** under the current "
        "representation — that limitation is the project's own finding and is "
        "reproduced here unchanged."
    )

# --------------------------------------------------------------------------- #
# K-Means selection
# --------------------------------------------------------------------------- #
with tab_selection:
    st.subheader("K-Means model-selection sweep")
    table_path = CLUSTERING_RESULTS_DIR / "benchmark_summary.csv"
    if table_path.is_file():
        table = load_table(table_path)
        st.dataframe(table, use_container_width=True)
        selected = table[table["Selected Model"] == True]  # noqa: E712
        if not selected.empty:
            row = selected.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Selected k", int(row["k"]))
            c2.metric("Silhouette", f"{row['Silhouette Score']:.4f}")
            c3.metric("Davies-Bouldin", f"{row['Davies-Bouldin Index']:.4f}")
            c4.metric("Calinski-Harabasz", f"{row['Calinski-Harabasz Index']:.1f}")
        st.caption(
            "The three internal indices do not necessarily agree; the notebook selects "
            "k = 3 as the elbow with a near-optimal silhouette and balanced cluster sizes."
        )
    else:
        st.warning("`benchmark_summary.csv` not found.")

    for name in ("kmeans_selection_curves.png", "kmeans_marginal_inertia.png",
                 "kmeans_cluster_distribution.png"):
        path = CLUSTERING_RESULTS_DIR / name
        if path.is_file():
            st.image(str(path), caption=path.stem.replace("_", " "), use_container_width=True)

# --------------------------------------------------------------------------- #
# Agglomerative
# --------------------------------------------------------------------------- #
with tab_agglom:
    st.subheader("Agglomerative clustering & linkage comparison")
    for name in ("agglomerative_linkage_comparison.csv",
                 "agglomerative_degeneracy_sensitivity.csv"):
        path = CLUSTERING_RESULTS_DIR / name
        if path.is_file():
            st.markdown(f"**{name}**")
            st.dataframe(load_table(path), use_container_width=True)
    report = CLUSTERING_RESULTS_DIR / "agglomerative_clustering_report.txt"
    if report.is_file():
        st.code(load_text(report), language="text")

# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
with tab_profiles:
    st.subheader("Cluster profiles (standardised feature means)")
    for name in ("cluster_profile.png", "kmeans_cluster_profile_heatmap.png",
                 "agglomerative_cluster_profile_heatmap.png",
                 "agglomerative_cluster_profile.csv"):
        path = CLUSTERING_RESULTS_DIR / name
        if path.is_file():
            if path.suffix == ".csv":
                st.markdown(f"**{name}**")
                st.dataframe(load_table(path), use_container_width=True)
            else:
                st.image(str(path), caption=path.stem.replace("_", " "), use_container_width=True)

# --------------------------------------------------------------------------- #
# Post-hoc satisfaction
# --------------------------------------------------------------------------- #
with tab_sat:
    st.warning(
        "Satisfaction was **excluded** from clustering and from model selection. "
        "Everything below is post-hoc interpretation only and introduced no leakage."
    )
    try:
        assignments = clustering_assignments()
        satisfaction = clustering_satisfaction()
        frame = pd.DataFrame({"cluster": assignments, "satisfaction": satisfaction})
        crosstab = pd.crosstab(frame["cluster"], frame["satisfaction"], normalize="index")
        st.markdown("**Satisfaction rate by cluster**")
        st.dataframe(crosstab.style.format("{:.4f}"), use_container_width=True)
        counts = frame["cluster"].value_counts().sort_index()
        st.markdown("**Cluster sizes (full-data labels)**")
        st.dataframe(
            pd.DataFrame({"cluster": counts.index, "customers": counts.values,
                          "share": (counts.values / counts.sum()).round(4)}),
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Could not load cluster assignments: {exc}")

    for name in ("cluster_satisfaction_posthoc.png",
                 "agglomerative_cluster_satisfaction_posthoc.png"):
        path = CLUSTERING_RESULTS_DIR / name
        if path.is_file():
            st.image(str(path), caption=path.stem.replace("_", " "), use_container_width=True)

# --------------------------------------------------------------------------- #
# Figure gallery
# --------------------------------------------------------------------------- #
with tab_figs:
    result_gallery(files["images"], columns=2)
