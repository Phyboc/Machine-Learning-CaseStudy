"""Picklable scikit-learn transformers used by the case-study model pipelines.

These classes exist so that the *exact* preprocessing used during training can be
serialized inside a scikit-learn ``Pipeline`` (via joblib) and reused verbatim at
inference time.  They are deliberately dependency-light and contain no notebook
state.

The two tracks use different, dataset-specific preprocessing:

Classification (fraud detection)
    ``Log1pColumns``  -> log1p-compress the skewed, non-negative, non-binary
                         columns identified on the training split.
    ``StandardScaler`` -> z-score the compressed matrix (fit on train only).

Regression (superconductivity)
    ``ColumnKeeper``     -> keep the features that survived the |r| > 0.95
                            correlation filter (fitted on train only).
    ``ThermalRatioAdder``-> append the engineered feature
                            ``thermal_to_mass_ratio``.
    ``StandardScaler``   -> z-score the filtered/engineered matrix.

Nothing here changes the methodology of the existing notebooks: it merely
packages the same sequence of operations into reusable, serializable objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "Log1pColumns",
    "ColumnKeeper",
    "ThermalRatioAdder",
    "select_log1p_columns",
]


def select_log1p_columns(
    df: pd.DataFrame, skew_threshold: float = 1.0
) -> list[str]:
    """Return the columns eligible for ``log1p`` compression.

    Eligibility is decided from the *training* data only and requires a column to
    be (a) non-binary (more than two unique values), (b) non-negative, and
    (c) right-skewed beyond ``skew_threshold``.

    This is the identical rule used by the original preprocessing notebook, so
    the selection is reproducible from the raw data alone.
    """
    eligible: list[str] = []
    for col in df.columns:
        series = df[col]
        if series.nunique() <= 2:
            continue
        if series.min() < 0:
            continue
        if skew(series.dropna()) > skew_threshold:
            eligible.append(col)
    return eligible


class Log1pColumns(BaseEstimator, TransformerMixin):
    """Apply ``np.log1p`` to a fixed set of columns; pass the rest through.

    Parameters
    ----------
    columns:
        The columns to compress.  Supplied explicitly (rather than re-derived at
        transform time) so that train and inference use an identical selection.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        X = self._as_frame(X)
        if self.columns is None:
            self.columns_ = select_log1p_columns(X)
        else:
            self.columns_ = [c for c in self.columns if c in X.columns]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        X = self._as_frame(X).copy()
        cols = getattr(self, "columns_", self.columns or [])
        present = [c for c in cols if c in X.columns]
        if present:
            X[present] = np.log1p(X[present].astype(float))
        return X

    def get_feature_names_out(self, input_features=None):  # noqa: N803
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class ColumnKeeper(BaseEstimator, TransformerMixin):
    """Keep (and order) a fixed subset of columns.

    Used for the regression correlation filter: the list of surviving columns is
    computed once on the training split and then frozen into the artifact.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        X = self._as_frame(X)
        if self.columns is None:
            self.columns_ = list(X.columns)
        else:
            self.columns_ = list(self.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        cols = list(getattr(self, "columns_", self.columns or []))
        reference = list(getattr(self, "feature_names_in_", cols))
        X = self._as_frame(X, reference=reference)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(
                "ColumnKeeper: missing required feature columns: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
        return X.loc[:, cols]

    def get_feature_names_out(self, input_features=None):  # noqa: N803
        return np.asarray(
            getattr(self, "columns_", self.columns or []), dtype=object
        )

    @staticmethod
    def _as_frame(X, reference=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if reference is not None and arr.shape[1] == len(reference):
            return pd.DataFrame(arr, columns=list(reference))
        return pd.DataFrame(arr)


class ThermalRatioAdder(BaseEstimator, TransformerMixin):
    """Append the engineered ``thermal_to_mass_ratio`` feature.

    ``thermal_to_mass_ratio = mean_ThermalConductivity / (mean_atomic_mass + 1e-5)``

    This mirrors the single feature-engineering step of the regression notebook.
    The required source columns must be present; if they are not, the frame is
    returned unchanged so the transformer degrades gracefully.
    """

    NUMERATOR = "mean_ThermalConductivity"
    DENOMINATOR = "mean_atomic_mass"
    OUTPUT = "thermal_to_mass_ratio"
    EPSILON = 1e-5

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.available_ = (
            self.NUMERATOR in X.columns and self.DENOMINATOR in X.columns
        )
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        X = self._as_frame(X).copy()
        if not getattr(self, "available_", True):
            return X
        if self.NUMERATOR not in X.columns or self.DENOMINATOR not in X.columns:
            return X
        X[self.OUTPUT] = X[self.NUMERATOR].astype(float) / (
            X[self.DENOMINATOR].astype(float) + self.EPSILON
        )
        return X

    def get_feature_names_out(self, input_features=None):  # noqa: N803
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        names = list(input_features) if input_features is not None else []
        if getattr(self, "available_", True) and self.OUTPUT not in names:
            names.append(self.OUTPUT)
        return np.asarray(names, dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
