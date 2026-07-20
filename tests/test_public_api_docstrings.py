"""Ensure every symbol re-exported by :mod:`engine.__init__.__all__` has a
non-empty docstring.

Companion assertion to Prompt 11: docstrings are enforced by test so future
public additions can't ship undocumented.
"""

from __future__ import annotations

import engine


def test_public_api_docstrings():
    missing = []
    for name in engine.__all__:
        obj = getattr(engine, name)
        doc = getattr(obj, "__doc__", None)
        if not (doc and doc.strip()):
            missing.append(name)
    assert not missing, f"engine.__all__ symbols missing docstrings: {missing}"
