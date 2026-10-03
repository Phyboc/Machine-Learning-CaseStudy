"""Figure helpers.

Two kinds of visuals are offered:

* **Existing result images** produced by the notebooks (displayed as-is, never
  regenerated) — see ``render_result_image``.
* **Lightweight metric charts** built from the persisted test-set metrics so the
  GUI can compare models without rerunning anything — see ``metric_bar_chart``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .paths import relative_to_repo

CLASSIFICATION_METRICS = [
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1_weighted", "F1 (weighted)", True),
    ("roc_auc", "ROC-AUC", True),
]

REGRESSION_METRICS = [
    ("r2", "R² (higher is better)", True),
    ("rmse", "RMSE (lower is better)", False),
    ("mae", "MAE (lower is better)", False),
    ("cv_r2", "5-fold CV R² (higher is better)", True),
]


def metric_bar_chart(
    table: pd.DataFrame,
    metric_key: str,
    title: str,
    higher_is_better: bool = True,
):
    """Return a matplotlib figure ranking models by one persisted metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metric_key not in table.columns:
        return None

    frame = table[["Model", metric_key]].dropna(subset=[metric_key])
    if frame.empty:
        return None
    frame = frame.sort_values(metric_key, ascending=not higher_is_better)

    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.42 * len(frame) + 1.2)))
    colors = ["#2a9d8f" if higher_is_better else "#e76f51"] * len(frame)
    ax.barh(frame["Model"], frame[metric_key], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.set_title(title, fontweight="bold")
    for y, value in enumerate(frame[metric_key]):
        ax.annotate(f"{value:.4f}", (value, y), va="center", xytext=(4, 0),
                    textcoords="offset points", fontsize=8)
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


def render_result_image(path: Path, caption: str | None = None) -> None:
    """Display an existing result image from the notebooks."""
    st.image(str(path), caption=caption or path.name, use_container_width=True)


def render_table(path: Path, max_rows: int | None = None) -> None:
    """Display an existing result CSV."""
    frame = pd.read_csv(path)
    if max_rows is not None:
        frame = frame.head(max_rows)
    st.dataframe(frame, use_container_width=True)


def metric_summary_row(entry: dict) -> dict:
    """Flatten one metadata model entry into a display-friendly row."""
    metrics = entry.get("metrics", {}) or {}
    row = {
        "Model": entry["name"],
        "Artifact": entry["artifact"],
        "Estimator": entry.get("estimator_class", ""),
    }
    row.update(metrics)
    return row


def gallery_caption(path: Path) -> str:
    """Turn a result filename into a readable caption."""
    stem = path.stem.replace("_", " ").strip()
    return stem[:1].upper() + stem[1:]


__all__ = [
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "metric_bar_chart",
    "render_result_image",
    "render_table",
    "metric_summary_row",
    "gallery_caption",
    "relative_to_repo",
]
