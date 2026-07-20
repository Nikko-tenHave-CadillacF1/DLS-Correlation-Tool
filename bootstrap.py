"""Advisory dependency check for Run_*.py entrypoints.

Default behaviour (as of 2026-07): probe that the required top-level packages
are importable. If any are missing, print a single-line install hint to stderr
and exit(1). If all present, return silently.

Environment overrides:

* ``DLS_SKIP_BOOTSTRAP=1`` — no-op (used by CI where env is already provisioned).
* ``DLS_ENABLE_AUTO_VENV=1`` — opt IN to the legacy auto-venv + auto-pip
  re-exec behaviour (:func:`_legacy_auto_venv_bootstrap`). Preserved so
  double-click / first-run workflows can be re-enabled without reverting.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_REQUIRED_MODULES = ("pandas", "numpy", "matplotlib", "scipy")
_INSTALL_HINT = (
    "[bootstrap] Missing required packages: {missing}. "
    "Install with `python -m pip install -e .` "
    "(or `python -m pip install -r requirements.txt`)."
)


def ensure_dependencies(*, prefer_venv: bool = True) -> None:
    """Verify required packages are importable; exit(1) with a hint if not."""
    if os.environ.get("DLS_SKIP_BOOTSTRAP") == "1":
        return
    if os.environ.get("DLS_ENABLE_AUTO_VENV") == "1":
        _legacy_auto_venv_bootstrap(prefer_venv=prefer_venv)
        return
    missing = [m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        print(_INSTALL_HINT.format(missing=", ".join(missing)), file=sys.stderr)
        raise SystemExit(1)


# --- Legacy auto-venv path (opt-in via DLS_ENABLE_AUTO_VENV=1) --------------
# Retained verbatim from the pre-2026-07 bootstrap so it can be re-enabled
# without a revert. Not exercised in the default path.

_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_DIR = _PROJECT_ROOT / ".venv"
if os.name == "nt":
    _VENV_PYTHON = _VENV_DIR / "Scripts" / "python.exe"
else:
    _VENV_PYTHON = _VENV_DIR / "bin" / "python"


def _running_in_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == _VENV_PYTHON.resolve()
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
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)]
    )


def _legacy_auto_venv_bootstrap(*, prefer_venv: bool = True) -> None:
    if prefer_venv and not _running_in_venv():
        try:
            _create_venv()
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"[bootstrap] WARN: could not create venv ({exc}); continuing with current interpreter.")
        else:
            if _VENV_PYTHON.exists():
                main = sys.modules.get("__main__")
                main_file = getattr(main, "__file__", None)
                if main_file:
                    print(f"[bootstrap] Relaunching under {_VENV_PYTHON}")
                    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), str(main_file), *sys.argv[1:]])

    missing = _missing_required()
    if missing:
        print(f"[bootstrap] Missing packages: {', '.join(missing)}")
        try:
            _pip_install_requirements()
        except subprocess.CalledProcessError as exc:
            print(f"[bootstrap] ERROR: pip install failed: {exc}")
            raise SystemExit(1)
        still_missing = _missing_required()
        if still_missing:
            print(f"[bootstrap] ERROR: still missing after install: {', '.join(still_missing)}")
            raise SystemExit(1)


__all__ = ("ensure_dependencies",)
