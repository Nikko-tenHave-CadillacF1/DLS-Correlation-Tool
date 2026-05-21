"""Organise raw data files into event subfolders.

Scans Data/inputs/<workflow>/ for files matching the event pattern
(e.g. 26R03SUZ, 26T01BCN) and moves them into per-event subdirectories.

Usage:
    python tools/organise_data.py [--workflow correlation] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# Regex to extract the event code from standard filenames.
EVENT_PATTERN = re.compile(r"(26[RT]\d{2}[A-Z]{3})")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def detect_event(filename: str) -> str | None:
    """Extract event code from a filename, or None if not detected."""
    m = EVENT_PATTERN.search(filename)
    return m.group(1) if m else None


def organise(workflow: str, dry_run: bool = False) -> dict[str, list[Path]]:
    """Move files in Data/inputs/<workflow>/ into event subfolders.

    Returns a dict mapping event code → list of moved files.
    """
    input_dir = PROJECT_ROOT / "Data" / "inputs" / workflow
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    moved: dict[str, list[Path]] = {}

    for file in sorted(input_dir.iterdir()):
        if not file.is_file():
            continue
        event = detect_event(file.name)
        if event is None:
            continue

        target_dir = input_dir / event
        target = target_dir / file.name

        if target.exists():
            continue  # already organised

        moved.setdefault(event, []).append(file)

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(target))

    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default="correlation", help="Workflow subfolder name.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving files.")
    args = parser.parse_args()

    moved = organise(args.workflow, dry_run=args.dry_run)

    if not moved:
        print("Nothing to move — all files already organised or no event codes detected.")
        return

    prefix = "[DRY RUN] " if args.dry_run else ""
    for event, files in sorted(moved.items()):
        print(f"{prefix}{event}: {len(files)} file(s)")
        for f in files:
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
