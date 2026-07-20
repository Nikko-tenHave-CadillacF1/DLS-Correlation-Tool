"""Back-compat shim; behaviour lives in :mod:`engine.plotting.generate_scatter`.

See :mod:`engine.plot_generators_waveform` for the shim rationale. Prompt 12
Phase 3 (2026-07): ``DataPlotter`` no longer inherits from
:class:`ScatterMixin`; the class is now an empty deprecation shell.
"""

import warnings

from .plotting.generate_scatter import (  # noqa: F401
    _resolve_scatter_style,
    generate_scatter3d_plots,
    generate_scatter_plots,
)


class ScatterMixin:
    """Deprecated empty shell. Call the module functions directly:
    ``engine.plotting.generate_scatter.generate_scatter_plots(plotter)``.
    """

    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            f"{cls.__name__} inherits from ScatterMixin, which became an "
            "empty deprecation shell in Prompt 12 Phase 3 (2026-07). Call "
            "engine.plotting.generate_scatter.generate_scatter_plots "
            "directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
