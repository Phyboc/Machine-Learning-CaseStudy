"""Shared Streamlit UI helpers.

Kept deliberately small: page chrome, an input builder that groups the many
features into tidy sections, CSV upload/validation, and metric display blocks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.figures import metric_bar_chart
from core.schemas import FeatureSchema

MAX_UPLOAD_ROWS = 5000


def setup_page(title: str, icon: str = "🧪", layout: str = "wide") -> None:
    st.set_page_config(page_title=f"{title} · ML Case Study", page_icon=icon, layout=layout)
    st.title(f"{icon} {title}")


def repo_note() -> None:
    st.caption(
        "Interactive inference over the models already trained by the project's "
        "notebooks. No model is trained, tuned or re-fitted by this app."
    )


def track_metrics_block(table: pd.DataFrame, metrics: list[tuple], n_cols: int = 4) -> None:
    """Render the persisted test-set metrics as small charts + a table."""
    st.markdown("#### Test-set performance (persisted from the notebooks)")
    available = [(k, label, hib) for k, label, hib in metrics if k in table.columns]
    for start in range(0, len(available), n_cols):
        cols = st.columns(n_cols)
        for col, (key, label, higher) in zip(cols, available[start:start + n_cols]):
            with col:
                fig = metric_bar_chart(table, key, label, higher_is_better=higher)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
    st.dataframe(table, use_container_width=True)


# --------------------------------------------------------------------------- #
# Input construction
# --------------------------------------------------------------------------- #
def _widget_for(spec, key_prefix: str):
    """Render one feature input, returning its value."""
    label = spec.name
    key = f"{key_prefix}__{spec.name}"
    if spec.binary:
        return st.selectbox(
            label, options=[0, 1], index=1 if spec.default >= 0.5 else 0, key=key
        )
    step = 1.0 if spec.dtype == "int" else max(abs(spec.default) * 0.01, 1e-6)
    return st.number_input(
        label, value=float(spec.default), step=float(step), format="%.6g", key=key
    )


def manual_input_form(schema: FeatureSchema, key_prefix: str) -> dict:
    """Grouped feature inputs with reset-to-defaults, returning a value dict."""
    st.caption(
        "Defaults are **example values** derived from the dataset (feature medians) "
        "purely to make the form usable — they are not recommended or optimal inputs. "
        "Edit any field, then submit below."
    )

    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("Reset to defaults", key=f"{key_prefix}_reset"):
            for spec in schema.features:
                st.session_state.pop(f"{key_prefix}__{spec.name}", None)
            st.rerun()
    with col_b:
        st.caption(f"{len(schema.features)} features · grouped into {len(schema.sections())} sections")

    values: dict[str, float] = {}
    sections = schema.sections()
    for section, specs in sections.items():
        with st.expander(f"{section}  ({len(specs)} features)", expanded=len(sections) <= 3):
            cols = st.columns(3)
            for i, spec in enumerate(specs):
                with cols[i % 3]:
                    values[spec.name] = _widget_for(spec, key_prefix)
    return values


def csv_input_form(schema: FeatureSchema, key_prefix: str) -> pd.DataFrame | None:
    """Upload + validate a CSV of samples. Returns a validated frame or None."""
    st.caption(
        "Upload a CSV whose columns are the model's input features. "
        f"Required: {len(schema.features)} feature columns in any order. "
        "Target columns are rejected."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{key_prefix}_upload")
    if uploaded is None:
        return None
    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return None

    if len(frame) > MAX_UPLOAD_ROWS:
        st.warning(
            f"File has {len(frame):,} rows; showing/predicting on the first "
            f"{MAX_UPLOAD_ROWS:,}."
        )
        frame = frame.head(MAX_UPLOAD_ROWS)

    problems = schema.validate(frame)
    blocking = [p for p in problems if not p.startswith("Unexpected column")]
    for problem in problems:
        (st.warning if problem.startswith("Unexpected column") else st.error)(problem)
    if blocking:
        st.info(
            "Tip: an example file with the exact expected columns is shown in the "
            "‘Expected columns’ expander."
        )
        return None

    st.success(f"CSV validated — {len(frame)} row(s) × {len(schema.features)} features.")
    return frame


def expected_columns_expander(schema: FeatureSchema) -> None:
    with st.expander("Expected columns"):
        st.code("\n".join(schema.features[i].name for i in range(len(schema.features))), language="text")
        st.caption(
            f"Target column to avoid: `{schema.target}`"
            + (
                " · also forbidden: " + ", ".join(schema.extra.get("forbidden_columns", []))
                if schema.extra.get("forbidden_columns")
                else ""
            )
        )


def download_button(frame: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def result_gallery(paths: list[Path], columns: int = 2, key_prefix: str = "gallery") -> None:
    """Display existing result images in a responsive grid."""
    if not paths:
        st.info("No result images found for this track yet.")
        return
    for start in range(0, len(paths), columns):
        cols = st.columns(columns)
        for col, path in zip(cols, paths[start:start + columns]):
            with col:
                st.image(str(path), caption=path.stem.replace("_", " "), use_container_width=True)


__all__ = [
    "setup_page",
    "repo_note",
    "track_metrics_block",
    "manual_input_form",
    "csv_input_form",
    "expected_columns_expander",
    "download_button",
    "result_gallery",
    "MAX_UPLOAD_ROWS",
]
