"""Golden-output regression harness for DLS-Correlation-Tool workflows.

Records SHA-256 hashes of every plot PNG (pixel-array) and PowerPoint (raw
bytes) produced by a ``Run_*.py`` runner and compares subsequent runs against
the committed snapshot.

Usage
-----
    python tools/golden_regression.py snapshot Run_Correlation
    python tools/golden_regression.py verify   Run_Correlation
    python tools/golden_regression.py list

The snapshot file lives at ``tests/golden/<runner>.json`` and is committed
to git — regeneration is an explicit human action that must show up in a
diff for review.

Design notes
------------
* PNGs are hashed via ``np.array(PIL.Image.open(p))`` -> ``sha256`` on the
  raw pixel bytes. Hashing the file bytes directly would drift with every
  ``matplotlib`` version because PNG metadata includes a build-time
  timestamp.
* PPTX files are hashed via raw bytes. python-pptx is deterministic for
  identical inputs on a given python-pptx version.
* Runners are invoked in an isolated subprocess with a pre-loaded wrapper
  that neutralises interactive matplotlib calls (``matplotlib.use`` is
  replaced with a no-op and ``plt.show`` closes the figure). This lets
  headless CI process workflows whose tails call ``plt.show(block=True)``
  (e.g. Run_Correlation's debug scatter3d step) without hanging.
* No modifications are made to ``engine/``, ``Run_*.py``, or any existing
  test — the harness is a pure observer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_DIR = _REPO_ROOT / "tests" / "golden"


# ─── Runner discovery ────────────────────────────────────────────────────────


def _normalise_runner_name(name: str) -> str:
    """Accept 'Run_Correlation', 'correlation', 'Run_Correlation.py' -> canonical."""
    stem = Path(name).stem
    if not stem.lower().startswith("run_"):
        stem = "Run_" + stem
    # Preserve original casing for the tail (matches file on disk).
    for candidate in _REPO_ROOT.glob("Run_*.py"):
        if candidate.stem.lower() == stem.lower():
            return candidate.stem
    raise SystemExit(
        f"ERROR: no Run_*.py runner matches {name!r}. Available: {sorted(p.stem for p in _REPO_ROOT.glob('Run_*.py'))}"
    )


def _read_runner_globals(module_name: str) -> dict:
    """Import Run_<name>.py in-process to read WORKFLOW_NAME and EVENT.

    Importing the module (as opposed to running it as __main__) executes
    module-level code only — RUNS / plot definitions get built, but the
    ``if __name__ == "__main__":`` block is skipped, so no plots are
    generated as a side effect.
    """
    os.environ.setdefault("DLS_SKIP_BOOTSTRAP", "1")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    # Force fresh import so we don't get stale globals from a previous call.
    sys.modules.pop(module_name, None)
    mod = importlib.import_module(module_name)
    return dict(vars(mod))


def _resolve_output_dir(module_globals: dict) -> tuple[str, str, Path]:
    """Return (workflow_name, event, output_dir) for a runner's globals."""
    workflow_name = module_globals.get("WORKFLOW_NAME")
    event = module_globals.get("EVENT")
    if not workflow_name:
        raise SystemExit("ERROR: runner has no WORKFLOW_NAME constant.")
    from channel_config import get_workflow_dirs

    _, out_dir = get_workflow_dirs(workflow_name, event)
    return workflow_name, event or "", Path(out_dir)


def _resolve_expected_pptx(module_globals: dict) -> list[Path]:
    """Enumerate the canonical .pptx output paths a runner will produce.

    Reads `POWERPOINT_OUTPUT` (single) and `POWERPOINT_EXPORTS` (list of
    tuples) from the runner's globals. Filtering hashes by this list is
    what makes snapshots stable across runs — stray timestamped archives
    (`Correlation_Report_20260716_104936.pptx` produced by the win32com
    fallback when PowerPoint is holding a lock) get correctly ignored.
    """
    expected: list[Path] = []
    export_flag = bool(module_globals.get("EXPORT_TO_POWERPOINT", True))
    if not export_flag:
        return expected
    single = module_globals.get("POWERPOINT_OUTPUT")
    if single:
        expected.append(Path(single))
    exports = module_globals.get("POWERPOINT_EXPORTS") or []
    for entry in exports:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            expected.append(Path(entry[1]))
    return expected


# ─── Headless subprocess invocation ──────────────────────────────────────────

_SUBPROCESS_WRAPPER = r"""
import os, sys, runpy, matplotlib
# Force Agg BEFORE any Run_*.py code runs, then neutralise later use() calls
# (Run_Correlation forces TkAgg for its debug scatter3d tail; blocking there
# would hang the harness).
matplotlib.use('Agg', force=True)
matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close('all')
# --no-open suppresses os.startfile on the plots folder and the pptx.
sys.argv = [{module_name!r} + '.py', '--no-open']
runpy.run_module({module_name!r}, run_name='__main__')
"""


def _invoke_runner(module_name: str) -> None:
    """Run the runner headlessly in a fresh Python subprocess."""
    env = os.environ.copy()
    env["DLS_SKIP_BOOTSTRAP"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-c", _SUBPROCESS_WRAPPER.format(module_name=module_name)]
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            f"ERROR: {module_name} exited with code {proc.returncode}. "
            f"Aborting — do not trust snapshot output from a failed run."
        )


# ─── Hashing ─────────────────────────────────────────────────────────────────


def _hash_png_pixels(path: Path) -> str:
    """SHA-256 of the pixel array — ignores PNG metadata (creation time,
    tEXt chunks) which matplotlib varies per version.
    """
    try:
        from PIL import Image  # Pillow ships transitively with matplotlib
    except ImportError as exc:
        raise SystemExit(f"ERROR: Pillow is required (should be present via matplotlib). Import failed: {exc}") from exc
    import numpy as np

    with Image.open(path) as im:
        # Ensure a canonical mode so a mode change alone is caught.
        arr = np.asarray(im.convert("RGBA"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hash_file_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# PPTX/OOXML parts whose bytes vary between saves without reflecting any
# real content change. `docProps/core.xml` embeds `dcterms:modified`
# (wall-clock time of save) and `docProps/app.xml` embeds an edit-count.
# Excluding both from the content hash gives an actually-deterministic
# fingerprint of the presentation.
_PPTX_VOLATILE_PARTS = frozenset(
    {
        "docProps/core.xml",
        "docProps/app.xml",
    }
)


def _hash_pptx_content(path: Path) -> str:
    """Deterministic hash of a .pptx: unzip, hash each ZIP entry's bytes,
    then hash the sorted `{name: sha256}` table. Skips wall-clock and
    edit-count metadata parts that python-pptx rewrites on every save.
    """
    import zipfile

    per_part: dict[str, str] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name in _PPTX_VOLATILE_PARTS:
                continue
            with zf.open(info) as fh:
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                per_part[name] = h.hexdigest()
    outer = hashlib.sha256()
    for name in sorted(per_part):
        outer.update(name.encode("utf-8"))
        outer.update(b"\x00")
        outer.update(per_part[name].encode("ascii"))
        outer.update(b"\n")
    return outer.hexdigest()


# ─── Snapshot / verify payload ───────────────────────────────────────────────


def _collect_hashes(
    output_dir: Path,
    expected_pptx: list[Path],
) -> tuple[dict[str, str], dict[str, str]]:
    plots_dir = output_dir / "plots"
    png_hashes: dict[str, str] = {}
    if plots_dir.is_dir():
        for png in sorted(plots_dir.rglob("*.png")):
            rel = png.relative_to(plots_dir).as_posix()
            png_hashes[rel] = _hash_png_pixels(png)
    pptx_hashes: dict[str, str] = {}
    # Hash ONLY the canonical .pptx outputs declared by the runner. This
    # deliberately excludes:
    #   - Windows Office lock files (`~$<name>.pptx`)
    #   - Timestamped fallback archives (`<name>_YYYYMMDD_HHMMSS.pptx`)
    #     produced by the win32com fallback path when python-pptx cannot
    #     overwrite an open file.
    for expected in expected_pptx:
        if expected.is_file() and not expected.name.startswith("~$"):
            pptx_hashes[expected.name] = _hash_pptx_content(expected)
    return png_hashes, pptx_hashes


def _tool_versions() -> dict[str, str | None]:
    def _v(mod_name: str) -> str | None:
        try:
            m = importlib.import_module(mod_name)
        except ImportError:
            return None
        return getattr(m, "__version__", None) or "unknown"

    return {
        "python": ".".join(str(i) for i in sys.version_info[:3]),
        "matplotlib": _v("matplotlib"),
        "pillow": _v("PIL"),
        "python_pptx": _v("pptx"),
    }


def _snapshot_path(module_name: str) -> Path:
    return _GOLDEN_DIR / f"{module_name}.json"


# ─── Subcommands ─────────────────────────────────────────────────────────────


def _cmd_snapshot(args: argparse.Namespace) -> int:
    module_name = _normalise_runner_name(args.workflow)
    print(f"[snapshot] {module_name}: reading configuration...", flush=True)
    globals_dict = _read_runner_globals(module_name)
    workflow_name, event, output_dir = _resolve_output_dir(globals_dict)
    expected_pptx = _resolve_expected_pptx(globals_dict)
    print(f"[snapshot] {module_name}: output_dir = {output_dir}", flush=True)
    print(f"[snapshot] {module_name}: invoking runner headlessly...", flush=True)
    _invoke_runner(module_name)
    print(f"[snapshot] {module_name}: hashing outputs...", flush=True)
    png_hashes, pptx_hashes = _collect_hashes(output_dir, expected_pptx)
    if not png_hashes and not pptx_hashes:
        raise SystemExit(f"ERROR: no PNGs or PPTX found under {output_dir}. Refusing to save an empty snapshot.")
    payload = {
        "workflow_module": module_name,
        "workflow_name": workflow_name,
        "event": event,
        "plots_dir": (output_dir / "plots").relative_to(_REPO_ROOT).as_posix(),
        "output_dir": output_dir.relative_to(_REPO_ROOT).as_posix(),
        "versions": _tool_versions(),
        "png_hashes": png_hashes,
        "pptx_hashes": pptx_hashes,
    }
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    dest = _snapshot_path(module_name)
    # Deterministic serialisation: sorted keys, LF line endings, trailing NL,
    # so two consecutive `snapshot` invocations produce byte-identical files.
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    dest.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"[snapshot] {module_name}: wrote {dest.relative_to(_REPO_ROOT)} "
        f"({len(png_hashes)} PNG(s), {len(pptx_hashes)} PPTX)."
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    module_name = _normalise_runner_name(args.workflow)
    snap_path = _snapshot_path(module_name)
    if not snap_path.is_file():
        raise SystemExit(
            f"ERROR: no snapshot at {snap_path.relative_to(_REPO_ROOT)}. "
            f"Run `golden_regression.py snapshot {module_name}` first."
        )
    print(f"[verify]   {module_name}: reading snapshot...", flush=True)
    expected = json.loads(snap_path.read_text(encoding="utf-8"))
    globals_dict = _read_runner_globals(module_name)
    _, _, output_dir = _resolve_output_dir(globals_dict)
    expected_pptx = _resolve_expected_pptx(globals_dict)
    print(f"[verify]   {module_name}: invoking runner headlessly...", flush=True)
    _invoke_runner(module_name)
    print(f"[verify]   {module_name}: hashing outputs...", flush=True)
    actual_png, actual_pptx = _collect_hashes(output_dir, expected_pptx)
    problems = _diff_hashes("PNG", expected.get("png_hashes", {}), actual_png)
    problems += _diff_hashes("PPTX", expected.get("pptx_hashes", {}), actual_pptx)
    if not problems:
        n = len(actual_png) + len(actual_pptx)
        print(f"[verify]   {module_name}: OK - {n} hash(es) match.")
        return 0
    print(f"[verify]   {module_name}: FAILED", file=sys.stderr)
    for line in problems:
        print("  " + line, file=sys.stderr)
    print(
        f"\n  {len(problems)} regression(s) detected. Snapshot lives at {snap_path.relative_to(_REPO_ROOT)}.",
        file=sys.stderr,
    )
    return 1


def _diff_hashes(kind: str, expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for rel in sorted(set(expected) | set(actual)):
        exp = expected.get(rel)
        act = actual.get(rel)
        if exp is None:
            problems.append(f"[NEW]     {kind}: {rel}  (not in snapshot)")
        elif act is None:
            problems.append(f"[MISSING] {kind}: {rel}  (present in snapshot, absent on disk)")
        elif exp != act:
            problems.append(f"[CHANGED] {kind}: {rel}  expected={exp[:12]}... actual={act[:12]}...")
    return problems


def _cmd_list(args: argparse.Namespace) -> int:
    if not _GOLDEN_DIR.is_dir():
        print("(no snapshots)")
        return 0
    snaps = sorted(_GOLDEN_DIR.glob("*.json"))
    if not snaps:
        print("(no snapshots)")
        return 0
    print(f"{'workflow':<30} {'PNGs':>5} {'PPTX':>5}  path")
    print("-" * 70)
    for snap in snaps:
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
            n_png = len(data.get("png_hashes", {}))
            n_pptx = len(data.get("pptx_hashes", {}))
            print(
                f"{data.get('workflow_module', snap.stem):<30} {n_png:>5} {n_pptx:>5}  {snap.relative_to(_REPO_ROOT)}"
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  {snap.name}: unreadable ({exc})", file=sys.stderr)
    return 0


# ─── argparse ────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="golden_regression",
        description="Snapshot and verify plot-output hashes for Run_*.py workflows.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot", help="record hashes to tests/golden/<runner>.json")
    p_snap.add_argument("workflow", help="Runner name, e.g. 'Run_Correlation' or 'correlation'.")
    p_snap.set_defaults(func=_cmd_snapshot)

    p_verify = sub.add_parser("verify", help="re-run and diff hashes against snapshot")
    p_verify.add_argument("workflow", help="Runner name, e.g. 'Run_Correlation'.")
    p_verify.set_defaults(func=_cmd_verify)

    p_list = sub.add_parser("list", help="list all committed snapshots")
    p_list.set_defaults(func=_cmd_list)

    return p


def main() -> int:
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
