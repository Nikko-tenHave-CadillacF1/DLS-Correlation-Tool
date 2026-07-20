"""Back-compat shim for the two mixins formerly hosted here.

Real behaviour lives in :mod:`engine.plotting.generate_psd_hist` (for
:class:`PsdHistMixin`) and :mod:`engine.plotting.generate_heatmap` (for
:class:`HeatmapMixin`). Prompt 12 Phase 3 (2026-07): ``DataPlotter`` no
longer inherits from either mixin; both are now empty deprecation shells.
"""

import warnings

from .plotting.generate_heatmap import generate_heatmap_plots  # noqa: F401
from .plotting.generate_psd_hist import (  # noqa: F401
    generate_histogram_plots,
    generate_psd_plots,
)


class PsdHistMixin:
    """Deprecated empty shell. Call the module functions directly:
    ``engine.plotting.generate_psd_hist.generate_psd_plots(plotter)`` /
    ``.generate_histogram_plots(plotter)``.
    """

    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            f"{cls.__name__} inherits from PsdHistMixin, which became an "
            "empty deprecation shell in Prompt 12 Phase 3 (2026-07). Call "
            "engine.plotting.generate_psd_hist.generate_psd_plots / "
            ".generate_histogram_plots directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)


class HeatmapMixin:
    """Deprecated empty shell. Call the module function directly:
    ``engine.plotting.generate_heatmap.generate_heatmap_plots(plotter)``.
    """

    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            f"{cls.__name__} inherits from HeatmapMixin, which became an "
            "empty deprecation shell in Prompt 12 Phase 3 (2026-07). Call "
            "engine.plotting.generate_heatmap.generate_heatmap_plots "
            "directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
