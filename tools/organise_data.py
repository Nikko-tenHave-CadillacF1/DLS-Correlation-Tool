from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

EVENT_PATTERN = re.compile(r"(26[RT]\d{2}[A-Z]{3})")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def detect_event(filename: str) -> str | None:
    m = EVENT_PATTERN.search(filename)
    return m.group(1) if m else None


def organise(workflow: str, dry_run: bool = False) -> dict[str, list[Path]]:
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
            continue
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
