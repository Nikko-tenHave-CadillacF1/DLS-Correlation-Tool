"""Back-compat shim; behaviour lives in :mod:`engine.plotting.generate_bar_box`.

See :mod:`engine.plot_generators_waveform` for the shim rationale. Prompt 12
Phase 3 (2026-07): ``DataPlotter`` no longer inherits from
:class:`BarBoxMixin`; the class is now an empty deprecation shell.
"""

import warnings

from .plotting.generate_bar_box import (  # noqa: F401
    generate_bar_plots,
    generate_box_plots,
)


class BarBoxMixin:
    """Deprecated empty shell. Call the module functions directly:
    ``engine.plotting.generate_bar_box.generate_bar_plots(plotter)`` /
    ``.generate_box_plots(plotter)``.
    """

    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            f"{cls.__name__} inherits from BarBoxMixin, which became an "
            "empty deprecation shell in Prompt 12 Phase 3 (2026-07). Call "
            "engine.plotting.generate_bar_box.generate_bar_plots / "
            ".generate_box_plots directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
