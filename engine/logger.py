"""Centralised logging and diagnostic collection for the DLS Correlation Tool.

Usage in any module:
    from logger import log, DiagnosticCollector

    log.info("Loaded %d rows", n)
    log.warning("Channel '%s' missing", ch)

The logger is configured once on import. Verbosity is controlled by calling
``configure(verbose=True)`` early in the pipeline (done by plot_runtime).

DiagnosticCollector accumulates warnings/errors during a run and can flush
a summary at the end or feed the data-quality report.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

_LOG_NAME = "dls_correlation"
log = logging.getLogger(_LOG_NAME)

# Make stderr safe for Unicode characters even when the underlying console
# uses cp1252 (Windows) or stderr is redirected to a file.
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

# Default handler — writes to stderr so it doesn't mix with pipeline stdout.
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)  # default; configure() may lower to DEBUG
# Don't bubble to root (avoids duplicate prints if user code calls basicConfig).
log.propagate = False


def configure(*, verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Set logging verbosity and optionally add a file handler.

    Called once by plot_runtime at the start of a workflow.
    """
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    _handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Diagnostic:
    """A single diagnostic entry."""
    level: str          # "warning" | "error" | "info"
    source: str         # module/component name
    message: str


@dataclass
class DiagnosticCollector:
    """Accumulates diagnostics during a pipeline run.

    Use as a context-aware collector that generators write to. At the end of
    the run, flush a summary to the console or feed the data-quality report.
    """
    entries: list = field(default_factory=list)

    def warning(self, source: str, message: str) -> None:
        self.entries.append(Diagnostic("warning", source, message))
        log.warning("[%s] %s", source, message)

    def error(self, source: str, message: str) -> None:
        self.entries.append(Diagnostic("error", source, message))
        log.error("[%s] %s", source, message)

    def info(self, source: str, message: str) -> None:
        self.entries.append(Diagnostic("info", source, message))
        log.info("[%s] %s", source, message)

    @property
    def warnings(self) -> list:
        return [e for e in self.entries if e.level == "warning"]

    @property
    def errors(self) -> list:
        return [e for e in self.entries if e.level == "error"]

    def has_errors(self) -> bool:
        return any(e.level == "error" for e in self.entries)

    def summary(self) -> str:
        """Return a compact multi-line summary of collected diagnostics."""
        if not self.entries:
            return "No diagnostics."
        lines = []
        for entry in self.entries:
            prefix = {"warning": "!", "error": "X", "info": "-"}.get(entry.level, "?")
            lines.append(f"  {prefix} [{entry.source}] {entry.message}")
        counts = f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return f"Diagnostics ({counts}):\n" + "\n".join(lines)

    def clear(self) -> None:
        self.entries.clear()


# Module-level singleton — importable by any module that needs to collect diagnostics.
diagnostics = DiagnosticCollector()
