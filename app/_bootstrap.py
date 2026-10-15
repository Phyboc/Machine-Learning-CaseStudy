"""Import bootstrap shared by every page.

Streamlit executes each page as a standalone script, so each one must be able to
resolve ``core`` (in ``app/``) and ``src`` (at the repository root) on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

for _path in (str(APP_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.paths import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

__all__ = ["APP_DIR", "REPO_ROOT"]
