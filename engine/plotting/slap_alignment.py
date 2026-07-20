from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_slap_vcar_series(df):
    if "sLap" not in df.columns or "vCar" not in df.columns:
        return None, None
    s = pd.to_numeric(df["sLap"], errors="coerce")
    v = pd.to_numeric(df["vCar"], errors="coerce")
    tmp = pd.DataFrame({"s": s, "v": v}).dropna()
    if tmp.empty:
        return None, None
    tmp = tmp[tmp["s"] >= 0].sort_values("s")
    if tmp.empty:
        return None, None
    tmp = tmp.groupby("s", as_index=False)["v"].mean()
    if len(tmp) < 50:
        return None, None
    return tmp["s"].to_numpy(dtype=float), tmp["v"].to_numpy(dtype=float)


def _score_slap_alignment(ref_s, ref_v, oth_s, oth_v, scale, offset):
    transformed = oth_s * scale + offset
    lo = max(ref_s.min(), transformed.min())
    hi = min(ref_s.max(), transformed.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.arange(np.ceil(lo), np.floor(hi) + 1.0, 5.0)
    if grid.size < 100:
        return None
    ref_interp = np.interp(grid, ref_s, ref_v)
    oth_interp = np.interp(grid, transformed, oth_v)
    if np.std(ref_interp) < 1e-9 or np.std(oth_interp) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(ref_interp, oth_interp)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    mae = float(np.mean(np.abs(ref_interp - oth_interp)))
    return corr, mae, grid.size
