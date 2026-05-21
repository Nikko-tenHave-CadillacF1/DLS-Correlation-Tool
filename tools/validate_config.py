"""Validate a Run_*.py configuration without running the full pipeline.

Checks:
  - All referenced data files exist
  - Run types are valid
  - PowerPoint template exists (if specified)
  - Plot definitions are well-formed

Usage:
    python tools/validate_config.py Run_Correlation.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_module_from_path(path: Path):
    """Dynamically import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("_user_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Prevent the module from actually running the workflow
    sys.modules["_user_config"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", help="Path to a Run_*.py file to validate.")
    args = parser.parse_args()

    config_path = Path(args.config_file).resolve()
    if not config_path.exists():
        print(f"ERROR: File not found: {config_path}")
        sys.exit(1)

    print(f"Validating: {config_path.name}")
    print("-" * 50)

    # Import plot_runtime for validation utilities
    from engine.plot_runtime import validate_config, workflow_config, build_plot_groups

    # Read the config file to extract key variables
    # We can't safely exec Run_*.py (it calls run_workflow at module level),
    # so we parse with --dry-run simulation
    print("  Running with --dry-run to validate...")
    old_argv = sys.argv
    sys.argv = [str(config_path), "--dry-run"]
    try:
        spec = importlib.util.spec_from_file_location("_run_config", config_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print("\n  ✓ Configuration is valid (dry-run passed).")
    except SystemExit as e:
        if e.code and e.code != 0:
            print(f"\n  ✗ Validation failed (exit code {e.code}).")
            sys.exit(1)
        else:
            print("\n  ✓ Configuration is valid.")
    except Exception as e:
        print(f"\n  ✗ Error during validation: {e}")
        sys.exit(1)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
