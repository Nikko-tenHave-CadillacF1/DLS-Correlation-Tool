"""Back-compat shim; behaviour lives in :mod:`engine.plotting.generate_waveform`.

Kept only so downstream code that imports :class:`WaveformMixin` or
``generate_waveform_plots`` from this path keeps resolving. ``DataPlotter``
no longer inherits from :class:`WaveformMixin` (Prompt 12 Phase 3, 2026-07);
external subclasses trigger a :class:`DeprecationWarning`.
"""

import warnings

from .plotting.generate_waveform import generate_waveform_plots  # noqa: F401


class WaveformMixin:
    """Deprecated empty shell. Call the module functions directly:
    ``engine.plotting.generate_waveform.generate_waveform_plots(plotter)``.
    """

    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            f"{cls.__name__} inherits from WaveformMixin, which became an "
            "empty deprecation shell in Prompt 12 Phase 3 (2026-07). Call "
            "engine.plotting.generate_waveform.generate_waveform_plots "
            "directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
