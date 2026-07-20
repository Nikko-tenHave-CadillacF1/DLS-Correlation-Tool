from __future__ import annotations

_FOLDER_RUN_COLOR_PALETTE = (
    "#FF8000",
    "#2000BF",
    "#D70000",
    "#008CFF",
    "#00CC88",
    "#CC0066",
    "#FFD700",
    "#4C00BF",
)
_TYPE_COLORMAPS = {
    "CAR": "Oranges",
    "DLS": "Blues",
    "DIL": "Greens",
    "OC": "Purples",
    "FMIOpt": "Reds",
}


def _shades_from_cmap(
    cmap_name: str,
    n: int,
    offset: int = 0,
    low: float = 0.45,
    high: float = 0.95,
) -> list[str]:
    import matplotlib.cm as _cm
    from matplotlib.colors import to_hex

    cmap = _cm.get_cmap(cmap_name)
    total = max(n + offset, 2)
    span = high - low
    pts = [low + span * (i / max(total - 1, 1)) for i in range(offset, offset + n)]
    return [to_hex(cmap(p)) for p in pts]


def _interpolate_two_colors(start: str, end: str, n: int) -> list[str]:
    """Return ``n`` colours interpolated through HSV space between two
    user-supplied endpoints (hex strings, named colours, or any
    matplotlib-recognised colour spec).

    Hue is interpolated linearly without shortest-arc wrap correction so
    the path traces the rainbow when the endpoints span the spectrum
    (e.g. red ``#FF0000`` to blue-violet ``#4800FF`` passes through
    orange, yellow, green, cyan, blue). Saturation and value are also
    interpolated linearly so the endpoints reproduce the user's colours
    exactly.

    For ``n == 1`` returns the midpoint; for ``n >= 2`` the first
    colour is ``start`` and the last is ``end``.
    """
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb

    rgb1 = to_rgb(start)
    rgb2 = to_rgb(end)
    h1, s1, v1 = rgb_to_hsv(rgb1)
    h2, s2, v2 = rgb_to_hsv(rgb2)
    if n <= 0:
        return []
    if n == 1:
        mid_hsv = (
            ((h1 + h2) / 2) % 1.0,
            (s1 + s2) / 2,
            (v1 + v2) / 2,
        )
        return [to_hex(hsv_to_rgb(mid_hsv))]
    out: list[str] = []
    for i in range(n):
        t = i / (n - 1)
        h = (h1 + (h2 - h1) * t) % 1.0
        s = s1 + (s2 - s1) * t
        v = v1 + (v2 - v1) * t
        out.append(to_hex(hsv_to_rgb((h, s, v))))
    return out
