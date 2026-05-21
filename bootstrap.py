"""Plug-and-play bootstrap: ensure venv + dependencies before the script runs.

Each `Run_*.py` calls :func:`ensure_dependencies` at the top of the file so
the user can simply double-click a script (or hit Run in VS Code) without
having to manually create a virtual environment, activate it, or `pip install`.

What it does (in order, all idempotent):

1.  If the current interpreter is **not** running inside the project-local
    ``.venv``, create the venv (if missing) and re-execute the calling script
    using ``.venv\\Scripts\\python.exe`` so the user never sees a system-Python
    install path.
2.  Probe each required top-level package via :func:`importlib.util.find_spec`.
    If any are missing it runs ``pip install -r requirements.txt`` once.
3.  Returns silently on subsequent runs.

Set environment variable ``DLS_SKIP_BOOTSTRAP=1`` to disable everything (useful
in CI where the environment is already provisioned).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Top-level module names that must be importable for the tool to work.
# Optional packages (pywin32, tqdm) are intentionally NOT in this list — their
# absence is handled gracefully at runtime.
_REQUIRED_MODULES = ("pandas", "numpy", "matplotlib", "scipy")

# Modules that are nice-to-have. Reported only when missing.
_OPTIONAL_MODULES = {"pyarrow": "fast parquet I/O",
                     "fastparquet": "alternative parquet engine",
                     "tqdm": "progress bars",
                     "pptx": "cross-platform PowerPoint export"}

_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_DIR = _PROJECT_ROOT / ".venv"
if os.name == "nt":
    _VENV_PYTHON = _VENV_DIR / "Scripts" / "python.exe"
else:
    _VENV_PYTHON = _VENV_DIR / "bin" / "python"


def _running_in_venv() -> bool:
    try:
        current = Path(sys.executable).resolve()
        return current == _VENV_PYTHON.resolve()
    except OSError:
        return False


def _create_venv() -> None:
    if _VENV_DIR.exists():
        return
    print(f"[bootstrap] Creating virtual environment at {_VENV_DIR} ...")
    subprocess.check_call([sys.executable, "-m", "venv", str(_VENV_DIR)])


def _missing_required() -> list[str]:
    return [m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def _pip_install_requirements() -> None:
    req = _PROJECT_ROOT / "requirements.txt"
    if not req.exists():
        return
    print("[bootstrap] Installing dependencies from requirements.txt ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "-r", str(req)]
    )


def ensure_dependencies(*, prefer_venv: bool = True) -> None:
    """Make sure the script can run.

    Re-execs into ``.venv\\Scripts\\python.exe`` on the first call when the user
    launched a system Python. After that, ensures all required packages are
    importable, installing them via pip if needed.
    """
    if os.environ.get("DLS_SKIP_BOOTSTRAP") == "1":
        return

    # Step 1: re-exec into venv if we're not already inside it.
    if prefer_venv and not _running_in_venv():
        try:
            _create_venv()
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"[bootstrap] WARN: could not create venv ({exc}); "
                  f"continuing with current interpreter.")
        else:
            if _VENV_PYTHON.exists():
                # Only re-exec the *top-level* script, never an imported module.
                main = sys.modules.get("__main__")
                main_file = getattr(main, "__file__", None)
                if main_file:
                    print(f"[bootstrap] Relaunching under {_VENV_PYTHON}")
                    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), str(main_file), *sys.argv[1:]])

    # Step 2: ensure required packages are installed.
    missing = _missing_required()
    if missing:
        print(f"[bootstrap] Missing packages: {', '.join(missing)}")
        try:
            _pip_install_requirements()
        except subprocess.CalledProcessError as exc:
            print(f"[bootstrap] ERROR: pip install failed: {exc}")
            raise SystemExit(1)
        # Re-check; if still missing, give up clearly.
        still_missing = _missing_required()
        if still_missing:
            print(f"[bootstrap] ERROR: still missing after install: "
                  f"{', '.join(still_missing)}")
            raise SystemExit(1)


__all__ = ("ensure_dependencies",)
