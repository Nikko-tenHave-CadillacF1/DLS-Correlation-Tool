from __future__ import annotations

_INFO_CORNER_XY = {
    ("left", "top"): (0.02, 0.98),
    ("right", "top"): (0.98, 0.98),
    ("left", "bottom"): (0.02, 0.02),
    ("right", "bottom"): (0.98, 0.02),
    ("center", "top"): (0.50, 0.98),
    ("center", "bottom"): (0.50, 0.02),
    ("left", "center"): (0.02, 0.50),
    ("right", "center"): (0.98, 0.50),
}
_CORNER_TO_LOC = {
    ("left", "top"): "upper left",
    ("right", "top"): "upper right",
    ("left", "bottom"): "lower left",
    ("right", "bottom"): "lower right",
    ("center", "top"): "upper center",
    ("center", "bottom"): "lower center",
    ("left", "center"): "center left",
    ("right", "center"): "center right",
}
_LOC_TO_CORNER = {
    "upper right": ("right", "top"),
    "upper left": ("left", "top"),
    "lower right": ("right", "bottom"),
    "lower left": ("left", "bottom"),
    "upper center": ("center", "top"),
    "lower center": ("center", "bottom"),
    "center left": ("left", "center"),
    "center right": ("right", "center"),
}


def _legend_corner_from_bbox(bbox):
    """Return the closest ``_INFO_CORNER_XY`` corner for a bbox in axes coords.

    Standalone rewrite of the former ``DataPlotter._legend_corner_from_bbox``:
    the class-method used to take a legend and call ``self._legend_axes_bbox``
    to derive the bbox; that responsibility now sits with the caller so this
    helper is a pure function of the (x0, y0, x1, y1) tuple.
    """
    if bbox is None:
        return None
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    halign = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    valign = "bottom" if cy < 1 / 3 else ("top" if cy > 2 / 3 else "center")
    corner = (halign, valign)
    if corner in _INFO_CORNER_XY:
        return corner
    return min(
        _INFO_CORNER_XY.keys(),
        key=lambda c: (_INFO_CORNER_XY[c][0] - cx) ** 2 + (_INFO_CORNER_XY[c][1] - cy) ** 2,
    )
