"""Feature schemas for the classification and regression inference inputs.

A schema is derived from the *actual* training metadata (feature names + order)
and from representative rows of the raw dataset (used only for example defaults,
never for training).  It is the single place the GUI asks: "what does a valid
input look like for this track?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .metadata import load_metadata

# Statistic prefixes used by the superconductivity feature names.  Stripping
# them yields the underlying physical property, which makes tidy input groups.
_STAT_PREFIXES = (
    "wtd_gmean_",
    "wtd_entropy_",
    "wtd_range_",
    "wtd_std_",
    "wtd_mean_",
    "gmean_",
    "entropy_",
    "range_",
    "std_",
    "mean_",
)


@dataclass
class FeatureSpec:
    """One model input feature."""

    name: str
    dtype: str = "float"
    default: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    binary: bool = False
    section: str = "Features"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "default": self.default,
            "min": self.minimum,
            "max": self.maximum,
            "binary": self.binary,
            "section": self.section,
        }


@dataclass
class FeatureSchema:
    """The ordered feature contract for one track."""

    track: str
    target: str
    features: list[FeatureSpec] = field(default_factory=list)
    target_labels: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------- #
    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def target_columns(self) -> list[str]:
        """Column names that must never appear in an input frame."""
        names = {self.target}
        # The fraud dataset's regression target is a legitimate column, but the
        # classification target must never leak in as a feature.
        names.update(self.extra.get("forbidden_columns", []))
        return sorted(names)

    def sections(self) -> dict[str, list[FeatureSpec]]:
        grouped: dict[str, list[FeatureSpec]] = {}
        for spec in self.features:
            grouped.setdefault(spec.section, []).append(spec)
        return grouped

    def defaults(self) -> dict[str, float]:
        return {f.name: f.default for f in self.features}

    def frame(self, values: dict[str, Any] | None = None, n_rows: int = 1) -> pd.DataFrame:
        """Build a correctly ordered, validated input frame.

        ``values`` overrides individual features; anything missing falls back to
        the example default.
        """
        values = values or {}
        unknown = set(values) - set(self.feature_names)
        if unknown:
            raise ValueError(f"Unknown feature(s) for {self.track}: {sorted(unknown)}")
        row = {name: values.get(name, self.defaults()[name]) for name in self.feature_names}
        return pd.DataFrame([row] * n_rows, columns=self.feature_names)

    # -- validation -------------------------------------------------------- #
    def validate(self, df: pd.DataFrame) -> list[str]:
        """Return a list of human-readable problems; empty means valid."""
        problems: list[str] = []
        if not isinstance(df, pd.DataFrame):
            return ["Input is not a table."]
        if df.empty:
            problems.append("Input table has no rows.")

        forbidden = [c for c in self.target_columns if c in df.columns]
        if forbidden:
            problems.append(
                "Target column(s) must not be used as input: " + ", ".join(forbidden)
            )

        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            shown = ", ".join(missing[:10])
            suffix = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            problems.append(f"Missing required column(s): {shown}{suffix}")

        extra = [c for c in df.columns if c not in self.feature_names]
        if extra:
            shown = ", ".join(map(str, extra[:10]))
            suffix = f" (+{len(extra) - 10} more)" if len(extra) > 10 else ""
            problems.append(f"Unexpected column(s) will be ignored: {shown}{suffix}")

        present = [c for c in self.feature_names if c in df.columns]
        if present:
            numeric = df[present].apply(pd.to_numeric, errors="coerce")
            non_numeric = [c for c in present if numeric[c].isna().all() and df[c].notna().any()]
            if non_numeric:
                problems.append(
                    "Non-numeric value(s) in: " + ", ".join(map(str, non_numeric[:10]))
                )
            nan_cols = [c for c in present if numeric[c].isna().any()]
            if nan_cols:
                problems.append(
                    "Missing/blank value(s) in: " + ", ".join(map(str, nan_cols[:10]))
                )
        return problems

    def coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the feature columns in the trained order, coerced to float."""
        ordered = df.loc[:, self.feature_names].copy()
        return ordered.apply(pd.to_numeric, errors="coerce").astype(float)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _section_for_regression(name: str) -> str:
    if name == "thermal_to_mass_ratio":
        return "Engineered feature"
    for prefix in _STAT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):].replace("_", " ")
    return "Other properties"


def _section_for_classification(name: str, index: int, block: int = 8) -> str:
    start = (index // block) * block
    end = min(start + block - 1, 10**9)
    return f"{name.split('_')[0]}_{start} – {name.split('_')[0]}_{end}"


def _spec_from_series(name: str, series: pd.Series, section: str) -> FeatureSpec:
    numeric = pd.to_numeric(series, errors="coerce")
    unique = numeric.dropna().unique()
    is_binary = len(unique) <= 2 and set(np.round(unique, 6)).issubset({0.0, 1.0})
    median = float(numeric.median()) if numeric.notna().any() else 0.0
    default = round(float(round(median)), 6) if is_binary else float(median)
    return FeatureSpec(
        name=name,
        dtype="int" if is_binary else "float",
        default=default,
        minimum=float(numeric.min()) if numeric.notna().any() else None,
        maximum=float(numeric.max()) if numeric.notna().any() else None,
        mean=float(numeric.mean()) if numeric.notna().any() else None,
        std=float(numeric.std()) if numeric.notna().any() else None,
        binary=bool(is_binary),
        section=section,
    )


def build_schema(track: str, defaults_source: pd.DataFrame) -> FeatureSchema:
    """Build a schema from metadata feature order + a frame of raw example rows.

    ``defaults_source`` should contain raw (unscaled) feature values in the
    training column names; only summary statistics are used.
    """
    metadata = load_metadata(track)
    names = list(metadata["feature_names"])
    missing = [c for c in names if c not in defaults_source.columns]
    if missing:
        raise ValueError(
            f"Defaults source is missing {len(missing)} feature(s) for {track}: {missing[:10]}"
        )

    features: list[FeatureSpec] = []
    for index, name in enumerate(names):
        section = (
            _section_for_regression(name)
            if track == "regression"
            else _section_for_classification(name, index)
        )
        features.append(_spec_from_series(name, defaults_source[name], section))

    return FeatureSchema(
        track=track,
        target=metadata["target"],
        features=features,
        target_labels=metadata.get("target_labels", {}),
        extra={
            "feature_count": len(names),
            "dataset": metadata.get("dataset"),
            "forbidden_columns": metadata.get("forbidden_columns", []),
        },
    )


def feature_matrix(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Validate then coerce a user frame into the exact trained feature order."""
    problems = schema.validate(frame)
    blocking = [p for p in problems if not p.startswith("Unexpected column")]
    if blocking:
        raise ValueError("; ".join(blocking))
    return schema.coerce(frame)


def coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Coerce ``columns`` to float, raising on unparseable values."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


__all__ = [
    "FeatureSpec",
    "FeatureSchema",
    "build_schema",
    "feature_matrix",
    "coerce_numeric",
]
