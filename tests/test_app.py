"""App integrity: modules import, results load, and no training happens."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"

TRACKS = ("classification", "regression", "clustering")

APP_MODULES = [
    "core.paths",
    "core.metadata",
    "core.schemas",
    "core.loaders",
    "core.inference",
    "core.figures",
]


@pytest.mark.parametrize("module", APP_MODULES)
def test_app_core_modules_import(module):
    __import__(module)


def test_app_main_and_pages_exist():
    assert (APP_DIR / "main.py").is_file()
    pages = sorted((APP_DIR / "pages").glob("*.py"))
    assert len(pages) >= 6, f"expected at least 6 pages, found {len(pages)}"


def test_pages_are_syntactically_valid():
    for path in sorted((APP_DIR / "pages").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_main_is_syntactically_valid():
    ast.parse((APP_DIR / "main.py").read_text(encoding="utf-8"), filename="main.py")


@pytest.mark.parametrize("track", TRACKS)
def test_result_files_load(track):
    import pandas as pd

    from core.loaders import result_files_by_kind

    files = result_files_by_kind(track)
    assert files["images"], f"no result images for {track}"

    for path in files["tables"]:
        frame = pd.read_csv(path)
        assert not frame.empty, f"{path.name} is empty"
    for path in files["text"]:
        assert path.read_text(encoding="utf-8", errors="replace").strip()


@pytest.mark.parametrize("track", TRACKS)
def test_result_images_are_non_empty_png(track):
    """Every result image must be a real, non-trivial file."""
    from core.loaders import result_files_by_kind

    for path in result_files_by_kind(track)["images"]:
        assert path.suffix == ".png", path.name
        assert path.stat().st_size > 1000, f"{path.name} looks truncated"


def test_clustering_results_load():
    from core.loaders import clustering_assignments, clustering_satisfaction

    assignments = clustering_assignments()
    satisfaction = clustering_satisfaction()
    assert len(assignments) == len(satisfaction)
    assert assignments.nunique() >= 2


def test_no_model_fitting_anywhere_in_app():
    """The GUI must never call fit()/GridSearchCV/notebook execution."""
    forbidden = ("GridSearchCV", "cross_val_score", ".fit(", "fit_transform(")
    offenders = []
    for path in list(APP_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {token!r}")
    assert not offenders, "Training-like calls found in the app:\n" + "\n".join(offenders)


def test_no_notebook_execution_in_app():
    text = "".join(p.read_text(encoding="utf-8") for p in APP_DIR.rglob("*.py"))
    for token in ("nbconvert", "nbformat", "subprocess", "os.system"):
        assert token not in text, f"app must not execute notebooks: found {token!r}"


def test_importing_core_does_not_fit_models(monkeypatch):
    """Importing the app core must not fit any estimator.

    ``BaseEstimator`` does not define ``fit`` itself, so the guard is installed
    on every concrete estimator class reachable from the app's imports.
    """
    import inspect

    from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

    fitted = {"count": 0}
    targets = []
    for cls in list(BaseEstimator.__subclasses__()):
        if issubclass(cls, (RegressorMixin, TransformerMixin)) or hasattr(cls, "fit"):
            if "fit" in cls.__dict__:
                targets.append(cls)

    originals = {cls: cls.fit for cls in targets}

    def counting_fit(self, *args, **kwargs):  # pragma: no cover - guard only
        fitted["count"] += 1
        return originals[type(self)].__get__(self, type(self))(*args, **kwargs)

    for cls in targets:
        monkeypatch.setattr(cls, "fit", counting_fit, raising=False)

    for module in APP_MODULES:
        sys.modules.pop(module, None)
    for module in APP_MODULES:
        __import__(module)

    assert fitted["count"] == 0, "importing the app core fitted a model"
    assert inspect.isclass(BaseEstimator)
