"""Generate a front-ARB (FARB) lookup JSON with a pseudo-deadband about zero twist.

Plug and play: edit SETTINGS below if needed, then run this file (no arguments).

The FARB JSON is a 2D table ``M [Nm] = f(aARBFL, aARBFR)`` with breakpoints in
rad (row 0 / column 0). The linear bar ``M = k * (aARBFL + aARBFR)`` is replaced
by a two-gradient curve of the twist ``s = aARBFL + aARBFR``::

    |s| <= s_db :  M = k_low * s
    |s| >  s_db :  M = sign(s) * (k_low * s_db + k_high * (|s| - s_db))

k_low, k_high and s_db are fitted jointly to every CAR run in ARB_CSV that used
that FARB, on the xDamperDeltaF vs FPRodDeltaF plot. Each run keeps its own
sensor-zero offsets and heave-crosstalk coefficient. A DLS run on the template
table converts plot units (N/mm, mm) to table units (Nm/rad, rad). One table is
made per FARB value in ARB_CSV that has CAR data on disk.

Outputs: Data/outputs/farb_deadband/FARB_<k>_deadband.json, plots/FARB_<k>_deadband.png
and plots/FARB_summary.png
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUTS_DIR = PROJECT_ROOT / "Data" / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "outputs" / "farb_deadband"

# ─── SETTINGS ────────────────────────────────────────────────────────────────
TEMPLATE_JSON = TOOLS_DIR / "FARB_8000.json"     # linear table: format/extent, and the table DLS_FILE ran on
ARB_CSV = TOOLS_DIR / "ARBS with no TB.csv"      # Race, Driver, Session, FARB, RARB
FARB_VALUES: list[float] = []                    # empty = every FARB in ARB_CSV
CAR_FILES: list[str] = []                        # empty = auto from ARB_CSV; else paths under Data/inputs (single FARB)
DLS_FILE = "correlation/26R16SEP/SEP PAG R15 P2R2 CORR_- Design COG_DLS.parquet"  # DLS run on TEMPLATE_JSON

# None = fitted; a value (plot units) fixes it.
K_LOW_N_PER_MM = None            # gradient inside the deadband [N/mm]
K_HIGH_N_PER_MM = None           # gradient outside the deadband [N/mm]
DEADBAND_HALF_WIDTH_MM = None    # half-width in xDamperDeltaF [mm]

MIN_VCAR_KPH = 100.0             # drop pit lane / slow running
MAX_ABS_VDAMPER_DELTA = 10.0     # [mm/s] drop damper-force hysteresis and kerb strikes
HEAVE_COMPENSATION = True        # remove FPRodAvgF crosstalk from CAR FPRodDeltaF
OUTLIER_SIGMA = 3.0              # robust-sigma trim before the final fit; None = off
GRID_POINTS = 7                  # table breakpoints per axis (7x7 = 49 nodes)
FINE_DEADBAND_RATIO = 1.0        # table fine band = ratio x deadband half-width (puts a node on the kink)
# ─────────────────────────────────────────────────────────────────────────────

X_CH, Y_CH = "xDamperDeltaF", "FPRodDeltaF"
GATE_CHANNELS = ("vCar", "vDamperDeltaF", "FPRodAvgF", "gLat")
ANCHOR_GLAT = 0.2                # |gLat| [g] treated as zero ARB torque when locating each run's deadband
DLS_NLAP = 1
MAX_POINTS_PER_RUN = 8000        # equal weighting between short and long runs
# Used when DLS_FILE is unavailable (26R16SEP DLS 'Design COG' on FARB_8000.json).
FALLBACK_MM_PER_RAD, FALLBACK_PLOT_SCALE, FALLBACK_FINE_DEG = 46.12, 1.116, 5.0


# ─── TABLE ───────────────────────────────────────────────────────────────────

def read_table(path: Path):
    raw = np.asarray(json.loads(path.read_text()), dtype=float)
    cols, rows, values = raw[0, 1:], raw[1:, 0], raw[1:, 1:]
    s = rows[:, None] + cols[None, :]
    k = float(np.sum(s * values) / np.sum(s * s))
    return k, float(max(np.abs(cols).max(), np.abs(rows).max())), raw[0, 0]


def write_table(path: Path, bp: np.ndarray, values: np.ndarray, corner: float) -> None:
    def fmt(v, nd):
        r = round(float(v), nd)
        return str(int(r)) if r == int(r) else f"{r:.{nd}f}".rstrip("0")

    lines = ["\t[ " + ", ".join([fmt(corner, 0)] + [fmt(b, 8) for b in bp]) + " ]"]
    lines += ["\t[ " + ", ".join([fmt(b, 8)] + [fmt(v, 3) for v in row]) + " ]" for b, row in zip(bp, values)]
    path.write_text("[\n" + ",\n".join(lines) + "\n]\n")


def deadband_moment(s, k_low, k_high, s_db):
    s = np.asarray(s, dtype=float)
    a = np.abs(s)
    return np.where(a <= s_db, k_low * s, np.sign(s) * (k_low * s_db + k_high * (a - s_db)))


def make_interpolator(bp: np.ndarray, values: np.ndarray):
    from scipy.interpolate import RegularGridInterpolator

    rgi = RegularGridInterpolator((bp, bp), values, bounds_error=False, fill_value=None)
    return lambda a, b: rgi(np.column_stack([np.ravel(a), np.ravel(b)])).reshape(np.shape(a))


def max_interp_error(bp: np.ndarray, values: np.ndarray, k_lo: float, k_hi: float, s_db: float,
                     extent: float, fine: float) -> float:
    table = make_interpolator(bp, values)
    pts = np.union1d(np.linspace(-extent, extent, 801), np.linspace(-fine, fine, 401))
    ga, gb = np.meshgrid(pts, pts)
    return float(np.max(np.abs(table(ga, gb) - deadband_moment(ga + gb, k_lo, k_hi, s_db))))


def bilinear_error(path: Path, k_lo: float, k_hi: float, s_db: float, extent: float, fine: float) -> float:
    raw = np.asarray(json.loads(path.read_text()), dtype=float)
    return max_interp_error(raw[0, 1:], raw[1:, 1:], k_lo, k_hi, s_db, extent, fine)


# ─── CSV / FILE LOOKUP ───────────────────────────────────────────────────────

_FILE_RE = re.compile(r"^(26[RT]\d{2}[A-Z]{3})_\d+_[^_]+_([A-Z]{3})_([A-Z]+\d?)_R(\d+)")
_CSV_EVENT_RE = re.compile(r"^26([RT]?)(\d{2})([A-Z]{3})$")
_CSV_SESSION_RE = re.compile(r"^F?(P[1-3]|SQ|SR|Q|GP)\d?(?:R(\d+))?")


def load_arb_csv(path: Path) -> list[tuple]:
    """(event, driver, session, run or None, farb) per CSV row."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for rec in list(csv.reader(fh))[1:]:
            if len(rec) < 4:
                continue
            ev = _CSV_EVENT_RE.match(rec[0].strip().upper())
            ss = _CSV_SESSION_RE.match(rec[2].strip().upper())
            try:
                farb = float(rec[3])
            except ValueError:
                continue
            if ev and ss:
                event = f"26{ev[1] or 'R'}{ev[2]}{ev[3]}"
                rows.append((event, rec[1].strip().upper(), ss[1], int(ss[2]) if ss[2] else None, farb))
    return rows


def farb_for_file(path: Path, arb_rows: list[tuple]) -> float | None:
    m = _FILE_RE.match(path.name)
    if m:
        for event, driver, session, run, farb in arb_rows:
            if (event, driver, session) == m.groups()[:3] and run in (None, int(m[4])):
                return farb
    return None


# ─── DATA (engine pipeline, identical to the Run_*.py scatter plots) ────────

def load_runs(runs: list[dict], channels: tuple[str, ...]):
    import channel_config as cc
    from engine import ScatterPlot, build_plot_groups
    from engine.dataplotter import DataPlotter
    from engine.logger import log

    # Column projection makes the engine warn about every unloaded transform/filter channel.
    log.setLevel(logging.ERROR)
    plotter = DataPlotter(
        root_folder=INPUTS_DIR,
        runs=runs,
        plot_definitions=build_plot_groups(scatters=[ScatterPlot(f"_{c}", X_CH, c) for c in (Y_CH, *channels)]),
        channel_mappings=cc.CHANNEL_MAPPINGS,
        channel_transforms=cc.CHANNEL_TRANSFORMS,
        calculated_channels=cc.CALCULATED_CHANNELS,
        filters=cc.FILTERS,
        units_map=cc.UNITS_MAP,
        resample_rate=cc.RESAMPLE_RATE,
        output_dir=OUTPUT_DIR,
    )
    return plotter.run_data, plotter.runs


def get_columns(df, names) -> list[np.ndarray] | None:
    """Float arrays of the named columns with NaN rows dropped, or None if any is missing."""
    if df is None or any(n not in df.columns for n in names):
        return None
    sub = df[list(names)].astype(float).dropna()
    return [sub[n].to_numpy() for n in names]


def calibrate(k_nom: float):
    """(mm_per_rad, plot_scale, fine_range_deg) from the DLS run on the nominal table."""
    path = INPUTS_DIR / DLS_FILE if DLS_FILE else None
    if path is None or not path.exists():
        print(f"[WARN] DLS file not found, using fallback calibration: {path}")
        return FALLBACK_MM_PER_RAD, FALLBACK_PLOT_SCALE, FALLBACK_FINE_DEG
    data, _ = load_runs([{"name": "DLS", "file": str(path), "type": "DLS", "nlap": DLS_NLAP}], ("aARBFL", "aARBFR"))
    cols = get_columns(data.get("dls"), (X_CH, Y_CH, "aARBFL", "aARBFR"))
    if cols is None:
        print("[WARN] DLS run lacks xDamperDeltaF/FPRodDeltaF/aARBFL/aARBFR, using fallback calibration.")
        return FALLBACK_MM_PER_RAD, FALLBACK_PLOT_SCALE, FALLBACK_FINE_DEG
    x, f, a_fl, a_fr = cols
    mm_per_rad = float(np.polyfit(np.deg2rad(a_fl + a_fr), x, 1)[0])
    plot_scale = float(np.polyfit(x, f, 1)[0]) / k_nom
    fine_deg = float(np.ceil((np.percentile(np.abs(np.r_[a_fl, a_fr]), 99.5) + 0.5) * 2) / 2)
    return mm_per_rad, plot_scale, fine_deg


def load_car_runs(farb: float, arb_rows: list[tuple]) -> list[dict]:
    """Filtered CAR runs: x, F, centred heave h and an all-True fit mask."""
    if CAR_FILES:
        files = [INPUTS_DIR / p for p in CAR_FILES]
    else:
        files = list({p.name: p for p in sorted(INPUTS_DIR.rglob("*.txt"))
                      if farb_for_file(p, arb_rows) == farb}.values())
    if not files:
        return []
    data, runs = load_runs([{"name": p.stem, "file": str(p), "type": "CAR"} for p in files], GATE_CHANNELS)
    out, seen = [], set()
    for run in runs:
        cols = get_columns(data.get(run["name"].lower()), (X_CH, Y_CH, *GATE_CHANNELS))
        if cols is None:
            print(f"[WARN] {run['name']}: missing {X_CH}/{Y_CH}/{GATE_CHANNELS}, skipped.")
            continue
        x, f, vcar, vdamp, favg, glat = cols
        # Same session exported twice (e.g. *_R01 and *_R01PARTIAL) differs only in rounding.
        key = (len(x), round(float(x.mean()), 3), round(float(f.mean())))
        if key in seen:
            print(f"[INFO] {run['name']}: duplicate data, skipped.")
            continue
        seen.add(key)
        keep = (vcar >= MIN_VCAR_KPH) & (np.abs(vdamp) <= MAX_ABS_VDAMPER_DELTA)
        if keep.sum() < 200:
            print(f"[WARN] {run['name']}: only {keep.sum()} samples after filtering, skipped.")
            continue
        x, f, favg, glat = x[keep], f[keep], favg[keep], glat[keep]
        zero = np.abs(glat) <= ANCHOR_GLAT
        h = favg - np.median(favg) if HEAVE_COMPENSATION else np.zeros(len(x))
        farb_run = farb_for_file(Path(run["file"]), arb_rows)
        out.append({"name": run["name"], "color": run["color"], "farb": f"{farb_run:g}" if farb_run else "?",
                    "x": x, "f": f, "h": h, "n_raw": len(keep), "mask": np.ones(len(x), bool),
                    "x_anchor": float(np.median(x[zero] if zero.sum() >= 50 else x))})
    return out


# ─── FITTING ─────────────────────────────────────────────────────────────────

def run_model(r: dict, p: dict, q, idx=slice(None)) -> np.ndarray:
    """Per-run model F = F0 + c*h + g(x - x0) with q = (x0, F0, c)."""
    return q[1] + q[2] * r["h"][idx] + deadband_moment(r["x"][idx] - q[0], p["K_lo"], p["K_hi"], p["x_db"])


def fit_deadband(runs: list[dict], fixed: dict):
    """Robust joint fit of shared (K_lo, K_hi, x_db) and per-run (x0, F0, c). Returns (params, per_run, sigma)."""
    from scipy.optimize import least_squares

    free = [n for n, v in fixed.items() if v is None]
    idxs, per0, slopes, lb_r, ub_r = [], [], [], [], []
    for r in runs:
        ii = np.flatnonzero(r["mask"])
        ii = ii[np.linspace(0, len(ii) - 1, min(len(ii), MAX_POINTS_PER_RUN)).astype(int)]
        x, f, h = r["x"][ii], r["f"][ii], r["h"][ii]
        xs = r["x_anchor"]
        a, c, _ = np.linalg.lstsq(np.column_stack([x, h, np.ones_like(x)]), f, rcond=None)[0]
        idxs.append(ii)
        slopes.append(a)
        per0.append([xs, 0.0, float(np.clip(c, -1.9, 1.9))])
        lb_r += [xs - 0.5, -np.inf, -2.0]
        ub_r += [xs + 0.5, np.inf, 2.0]
    k_hi0 = fixed["K_hi"] or 1.1 * float(np.median(slopes))
    init = {"K_lo": fixed["K_lo"] or 0.3 * k_hi0, "K_hi": k_hi0}
    for r, ii, q in zip(runs, idxs, per0):
        g = deadband_moment(r["x"][ii] - q[0], init["K_lo"], k_hi0, fixed["x_db"] or 0.2)
        q[1] = float(np.median(r["f"][ii] - g - q[2] * r["h"][ii]))

    def unpack(v):
        return {**fixed, **dict(zip(free, v[: len(free)]))}, v[len(free):].reshape(-1, 3)

    def resid(v):
        p, per = unpack(v)
        return np.concatenate([r["f"][ii] - run_model(r, p, q, ii) for r, ii, q in zip(runs, idxs, per)])

    bounds = {"K_lo": (0.0, np.inf, 1000.0), "K_hi": (1.0, np.inf, 1000.0), "x_db": (0.0, 2.0, 0.1)}
    lb = [bounds[n][0] for n in free] + lb_r
    ub = [bounds[n][1] for n in free] + ub_r
    x_scale = np.array([bounds[n][2] for n in free] + [0.1, 500.0, 0.1] * len(runs))
    best = None
    for db0 in (0.1, 0.25, 0.5) if "x_db" in free else (None,):
        v0 = np.array([{**init, "x_db": db0}[n] for n in free] + sum(per0, []), dtype=float)
        sol = least_squares(resid, v0, bounds=(lb, ub), x_scale=x_scale, loss="soft_l1", f_scale=300.0)
        if best is None or sol.cost < best.cost:
            best = sol
    p, per = unpack(best.x)
    res = np.concatenate([r["f"][r["mask"]] - run_model(r, p, q, r["mask"]) for r, q in zip(runs, per)])
    return p, per, float(1.4826 * np.median(np.abs(res - np.median(res))))


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main() -> None:
    k_tpl, outer, corner = read_table(TEMPLATE_JSON)
    cal = calibrate(k_tpl)
    arb_rows = load_arb_csv(ARB_CSV)
    farbs = FARB_VALUES or sorted({row[-1] for row in arb_rows})
    results = [res for farb in farbs if (res := process_farb(farb, outer, corner, cal, arb_rows))]

    print("\nSUMMARY")
    print(f"{'FARB':>6} {'runs':>4} {'K_lo N/mm':>10} {'K_hi N/mm':>10} {'db mm':>6} "
          f"{'k_lo Nm/rad':>12} {'k_hi Nm/rad':>12} {'db deg':>7} {'sigma N':>8}")
    for r in results:
        print(f"{r['farb']:6g} {r['n_runs']:4d} {r['K_lo']:10.0f} {r['K_hi']:10.0f} {r['x_db']:6.3f} "
              f"{r['k_lo']:12.0f} {r['k_hi']:12.0f} {np.rad2deg(r['s_db']):7.3f} {r['sigma']:8.0f}")
    skipped = [f"{f:g}" for f in farbs if f not in {r['farb'] for r in results}]
    if skipped:
        print(f"No usable CAR data for FARB {', '.join(skipped)}.")
    if results:
        print(f"Summary plot    : {plot_summary(OUTPUT_DIR / 'plots' / 'FARB_summary.png', results)}")


def process_farb(farb, outer, corner, cal, arb_rows) -> dict | None:
    """Fit and write the deadband table for one FARB value; None when there is no data to fit."""
    mm_per_rad, plot_scale, fine_deg = cal
    n_per_nm = plot_scale * mm_per_rad  # FPRodDeltaF per ARB moment [N/Nm]
    print(f"\n{'#' * 78}\nFARB {farb:g}")

    runs = load_car_runs(farb, arb_rows)
    fixed = {"K_lo": K_LOW_N_PER_MM, "K_hi": K_HIGH_N_PER_MM, "x_db": DEADBAND_HALF_WIDTH_MM}
    if not runs:
        if None in fixed.values():
            print(f"[SKIP] No CAR runs with FARB {farb:g} found.")
            return None
        p, sigma = fixed, float("nan")
    else:
        p, per, sigma = fit_deadband(runs, fixed)
        if OUTLIER_SIGMA:
            for r, q in zip(runs, per):
                r["mask"] = np.abs(r["f"] - run_model(r, p, q)) <= OUTLIER_SIGMA * sigma
            p, per, sigma = fit_deadband(runs, fixed)
        for r, q in zip(runs, per):
            r["x0"], r["f0"], r["c"] = q

    # Plot units -> table units, then build the table.
    k_lo, k_hi, s_db = p["K_lo"] / plot_scale, p["K_hi"] / plot_scale, p["x_db"] / mm_per_rad
    fine = np.deg2rad(fine_deg)
    inner = FINE_DEADBAND_RATIO * s_db
    bp = np.r_[-outer, np.linspace(-inner, inner, GRID_POINTS - 2), outer]
    values = deadband_moment(bp[:, None] + bp[None, :], k_lo, k_hi, s_db)
    table = make_interpolator(bp, values)

    stem = f"FARB_{farb:g}_deadband"
    out_json = OUTPUT_DIR / f"{stem}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_table(out_json, bp, values, corner)
    max_err = bilinear_error(out_json, k_lo, k_hi, s_db, outer, inner)

    def src(name):
        return "fixed" if fixed[name] is not None else "fitted"

    print("=" * 78)
    print(f"Nominal         : FARB {farb:g} Nm/rad (calibration DLS on {TEMPLATE_JSON.name})")
    print(f"Conversion      : {mm_per_rad:.3f} mm/rad, {plot_scale:.4f} (N/mm)/(Nm/rad), {n_per_nm:.2f} N/Nm")
    print(f"Filters         : vCar >= {MIN_VCAR_KPH:g} kph, |vDamperDeltaF| <= {MAX_ABS_VDAMPER_DELTA:g} mm/s, "
          f"heave comp {'on' if HEAVE_COMPENSATION else 'off'}, outlier trim {OUTLIER_SIGMA or 'off'} sigma")
    for r in runs:
        print(f"  {r['name']:<50} FARB {r['farb']:>6}  kept {r['mask'].sum():6d}/{r['n_raw']:<6d}"
              f"  x0={r['x0']:+.3f} mm  F0={r['f0']:+6.0f} N  heave c={r['c']:+.3f}")
    print(f"Low gradient    : {p['K_lo']:9.0f} N/mm  = {k_lo:9.1f} Nm/rad  ({src('K_lo')})")
    print(f"High gradient   : {p['K_hi']:9.0f} N/mm  = {k_hi:9.1f} Nm/rad  ({src('K_hi')})")
    print(f"Deadband (half) : {p['x_db']:9.3f} mm    = {np.rad2deg(s_db):.3f} deg twist  ({src('x_db')})")
    print(f"Fit residual    : robust sigma {sigma:.0f} N")
    print(f"Table           : {len(bp)} x {len(bp)}, fine band +/-{np.rad2deg(inner):.3f} deg, max bilinear error "
          f"{max_err:.2f} Nm (= {max_err * n_per_nm:.0f} N)")
    print("Nodes           : sampled on the deadband model")
    print(f"JSON written    : {out_json}")
    summary = {**p, "k_lo": k_lo, "k_hi": k_hi, "s_db": s_db, "k_nom": farb, "mm_per_rad": mm_per_rad,
               "n_per_nm": n_per_nm, "max_err": max_err, "sigma": sigma, "grid": len(bp), "fine": fine}
    print(f"Plot written    : {plot_validation(OUTPUT_DIR / 'plots' / f'{stem}.png', runs, table, summary)}")
    return {**summary, "farb": farb, "n_runs": len(runs)}


# ─── PLOT ────────────────────────────────────────────────────────────────────

def plot_validation(path: Path, runs: list[dict], table, s: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def json_curve(u):  # plot-space FPRodDeltaF of the JSON table at zero heave, u = xDamperDeltaF - x0
        tw = u / s["mm_per_rad"]
        return s["n_per_nm"] * table(tw / 2, tw / 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_raw, ax_zoom) = plt.subplots(1, 2, figsize=(18, 8))

    # Left: filtered xDamperDeltaF vs FPRodDeltaF, the JSON fit drawn from each run's x0, F0.
    for r in runs:
        x, f = r["x"][r["mask"]], r["f"][r["mask"]]
        ax_raw.scatter(x, f, s=3, color=r["color"], alpha=0.5, rasterized=True,
                       label=f"{r['name']} (FARB {r['farb']}, {len(x)}/{r['n_raw']} pts)")
        xs = np.linspace(x.min(), x.max(), 600)
        ax_raw.plot(xs, r["f0"] + json_curve(xs - r["x0"]), color="black", lw=1.5)
    ax_raw.plot([], [], color="black", lw=1.5, label="JSON fit (at each run's x0, F0)")
    ax_raw.set_title(f"{X_CH} vs {Y_CH} (filtered)")

    # Right: all runs centred on (x0, F0) and heave-compensated, with the JSON fit.
    half = max(4.0 * s["x_db"], 1.5)
    pu, pv = [], []
    for r in runs:
        m = r["mask"]
        u, v = r["x"][m] - r["x0"], r["f"][m] - r["f0"] - r["c"] * r["h"][m]
        w = np.abs(u) <= half
        ax_zoom.scatter(u[w], v[w], s=3, color=r["color"], alpha=0.25, rasterized=True)
        pu.append(u[w])
        pv.append(v[w])
    if runs:
        pu, pv = np.concatenate(pu), np.concatenate(pv)
        edges = np.arange(-half, half + 1e-9, 0.05)
        idx = np.digitize(pu, edges)
        bins = [(0.5 * (edges[i - 1] + edges[i]), np.median(pv[idx == i]))
                for i in range(1, len(edges)) if (idx == i).sum() >= 20]
        if bins:
            ax_zoom.plot(*zip(*bins), "o", color="white", mec="black", ms=5, zorder=5, label="Pooled CAR binned median")
    us = np.linspace(-half, half, 801)
    ax_zoom.plot(us, json_curve(us), color="black", lw=2, label="JSON fit (bilinear)")
    ax_zoom.set_xlim(-half, half)
    ax_zoom.set_ylim(-1.15 * s["K_hi"] * half, 1.15 * s["K_hi"] * half)
    ax_zoom.set_title("All runs centred on (x0, F0)" + (", heave-compensated" if HEAVE_COMPENSATION else ""))

    for ax, xl, yl, ms in ((ax_raw, f"{X_CH} (mm)", f"{Y_CH} (N)", 3),
                           (ax_zoom, f"{X_CH} - x0 (mm)", f"{Y_CH} - F0 (N)", 1)):
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(loc="lower right", fontsize=8, markerscale=ms)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_summary(path: Path, results: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(r):
        raw = np.asarray(json.loads((OUTPUT_DIR / f"FARB_{r['farb']:g}_deadband.json").read_text()), dtype=float)
        return make_interpolator(raw[0, 1:], raw[1:, 1:])

    def twist_curve(table, s):  # M at aARBFL = aARBFR = s / 2, so the twist is s
        return table(s / 2, s / 2)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(results)))
    s_wide = np.linspace(-2.0, 2.0, 801)  # table extent is +/-1 rad per ARB
    s_zoom = np.deg2rad(np.linspace(-3.0, 3.0, 601))
    s_slope = np.deg2rad(np.linspace(-6.0, 6.0, 1201))
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    ax_m, ax_zoom, ax_k = axes[0]
    for r, c in zip(results, colors):
        table = load(r)
        label = f"FARB {r['farb']:g}"
        ax_m.plot(np.rad2deg(s_wide), twist_curve(table, s_wide), color=c, label=label)
        ax_m.plot(np.rad2deg(s_wide), r["k_nom"] * s_wide, "--", color=c, lw=0.8)
        ax_zoom.plot(np.rad2deg(s_zoom), twist_curve(table, s_zoom), color=c, label=label)
        ax_zoom.plot(np.rad2deg(s_zoom), r["k_nom"] * s_zoom, "--", color=c, lw=0.8)
        ax_k.plot(np.rad2deg(s_slope), np.gradient(twist_curve(table, s_slope), s_slope), color=c, label=label)
    for ax, title, xl, yl in (
        (ax_m, "M vs twist (solid) with nominal linear k*twist (dashed)", "twist aARBFL + aARBFR [deg]", "M [Nm]"),
        (ax_zoom, "Zoom on the deadband", "twist [deg]", "M [Nm]"),
        (ax_k, "Local stiffness dM/dtwist [Nm/rad]", "twist [deg]", "dM/dtwist [Nm/rad]"),
    ):
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.legend(loc="upper left", fontsize=8)

    # Bottom row: table surfaces for the FARBs closest to the first, middle and last of the list.
    idx = np.unique(np.round(np.linspace(0, len(results) - 1, 3)).astype(int))
    for ax_h, i in zip(axes[1], idx):
        r = results[i]
        g = np.linspace(-r["fine"], r["fine"], 61)
        a_l, a_r = np.meshgrid(g, g, indexing="ij")
        z = load(r)(a_l, a_r)
        v = np.abs(z).max()
        mesh = ax_h.pcolormesh(np.rad2deg(a_r), np.rad2deg(a_l), z, cmap="RdBu_r", vmin=-v, vmax=v,
                               shading="nearest")
        ax_h.plot(np.rad2deg(g), np.rad2deg(g), "k--", lw=0.8)
        fig.colorbar(mesh, ax=ax_h, label="M [Nm]")
        ax_h.set_title(f"FARB {r['farb']:g} table  M(aARBFL, aARBFR) [Nm]")
        ax_h.set_xlabel("aARBFR [deg]")
        ax_h.set_ylabel("aARBFL [deg]")
        ax_h.set_aspect("equal")
    for ax_h in axes[1][len(idx):]:
        ax_h.axis("off")

    fig.suptitle("FARB deadband summary")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
