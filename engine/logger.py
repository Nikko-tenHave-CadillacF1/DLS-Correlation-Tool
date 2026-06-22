
from __future__ import annotations

import logging
import sys
from typing import Optional

_LOG_NAME = "dls_correlation"
log = logging.getLogger(_LOG_NAME)

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

_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False

def configure(*, verbose: bool = False, log_file: Optional[str] = None) -> None:
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    _handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)
