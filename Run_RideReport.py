"""Ride/DIL workflow — edit RUNS and plot definitions to configure your analysis.

This is the *complementary* report to the correlation report. Pipeline:

  1. RUNS expands via folder shorthand; the `consolidate` flag adds synthetic
     runs that concatenate source time-series with NaN gaps so that Welch sees
     one long signal with better PSD resolution and tighter confidence. Use
     `consolidate_by` to partition the merge — e.g. `consolidate_by="session"`
     emits one consolidated run per FP1/FP2/FP3/Q/GP session, since the car
     setup changes between sessions and a single weekend-wide merge would
     mix incompatible configurations.
  2. `VIBRATIONS_FIT` enables the automatic Lorentzian (or 4DOF body) modal
     fit per run inside `DataPlotter`. Modal parameters land as broadcast
     constant channels (`modal_heave_f0`, `modal_heave_zeta`, plus `_sigma`
     suffixes for both) so BarPlots can show them with error bars and the
     full fit results land on `plotter.modal_results` for the evolution plots.
  3. `POWERPOINT_EXPORTS` is a list of `(template, output, slides)` triples —
     a secondary modal-evolution deck is generated alongside the main ride
     deck.
  4. Set a `group` key on individual runs (`"RED"`, `"BLUE"`, event code, …)
     to split the modal-evolution lines by that key for car-to-car or
     event-to-event compare.
  5. After the workflow runs, `plot_modal_evolution(plotter, …)` is called
     to render the per-mode evolution figures (line+CI and bar+errorbar).
"""

from bootstrap import ensure_dependencies
ensure_dependencies()

from channel_config import get_workflow_dirs, resolve_template_path
from engine import (
    run_workflow, Slide,
    WaveformPlot, ScatterPlot, PsdPlot, BarPlot,
    plot_modal_evolution,
)

WORKFLOW_NAME = "ride_report"
EVENT = "26R07BCN"
_INPUT_DIR, _OUTPUT_DIR = get_workflow_dirs(WORKFLOW_NAME, EVENT)

# Single source of truth for Welch window length. 512 samples @ 100 Hz gives
# ?f ~0.2 Hz and many averages — better confidence on modal peaks than the
# default auto-sizing which favours frequency resolution over averaging.
NPERSEG = "auto"

# ─── RUNS ─────────────────────────────────────────────────────────────────────
# Folder expansion + consolidation. Modes for `consolidate`:
#   None  / omitted  → individual files only (default).
#   True             → individuals AND synthetic merged run(s).
#   "only"           → only the merged run(s) are kept in the active set.
#
# Partitioning via `consolidate_by` (default None → single all-in-one merge):
#   "session"        → preset extractor; groups files by the 2nd-to-last
#                      underscore token in the filename, so race-data files
#                      named <event>_<date>_<car>_<driver>_<session>_<run>
#                      yield one consolidated run per FP1/FP2/FP3/Q/GP.
#   <regex string>   → e.g. r"_(P\d|Q|GP)_" — first capture group = key.
#   <callable>       → fn(path: Path) -> str | None; None excludes the file.
#
# `consolidated_name` becomes a template when `consolidate_by` is set:
#   no `{group}` placeholder → suffix appended automatically
#                              (e.g. "26R07BCN" → "26R07BCN_P1", "26R07BCN_Q").
#   contains `{group}`       → formatted directly (e.g. "GP_{group}").
#
# Set `group` on an individual run (e.g. "RED"/"BLUE" car or event-code) to
# split the modal-evolution series — `plot_modal_evolution(group_by="group")`.

RUNS = [
    # One consolidated run per session (FP1, FP2, FP3, Q, GP) — car setup
    # changes between sessions so a single weekend-wide merge is not valid.
    # The car-tag prefix in `consolidated_name` guarantees RED and BLUE
    # consolidated runs have distinct names (otherwise BLUE would overwrite 
    # RED in `plotter.modal_results`, which keys by run name).

    {"folder": "RED", "filetype": ".txt", "type": "CAR",
     "consolidate": "only", "consolidate_by": "session",
     "consolidated_name": EVENT + "_RED", "group": "RED"},
    {"name": "DLS RED R", "type": "DLS", "file": r"RED/26R07BCN  11  Race_DLS.parquet", "group": "RED"},
    {"folder": "BLUE", "filetype": ".txt", "type": "CAR",
     "consolidate": "only", "consolidate_by": "session",
     "consolidated_name": EVENT + "_BLUE", "group": "BLUE"},
    {"name": "DLS BLUE R", "type": "DLS", "file": r"BLUE/26R07BCN  77  Race_DLS.parquet", "group": "BLUE"},
    # {"name": "BSL", "type": "DLS", "file": r"VPG Baselines  SPB  26R08SPB v1 LF_DLS.parquet", "color": "#FF0000"},
    # {"name": "500 HS", "type": "DLS", "file": r"VPG Baselines  SPB  26R08SPB v1 LF_HS500_DLS.parquet", "color": "#FF9900"},
    # {"name": "750 HS", "type": "DLS", "file": r"VPG Baselines  SPB  26R08SPB v1 LF_HS750_DLS.parquet", "color": "#8EB200"},
    # {"name": "1000 HS", "type": "DLS", "file": r"VPG Baselines  SPB  26R08SPB v1 LF_HS1000_DLS.parquet", "color": "#00A088"},
    # {"name": "7A HD", "type": "DLS", "file": r"VPG Baselines  SPB  26R08SPB v1 LF_7AFH_DLS.parquet", "color": "#0026FF"},
    # {"name": "7C HD", "type": "DLS", "file": r"VPG Baselines  SPB  26R08SPB v1 LF_7CFH_DLS.parquet", "color": "#6200FF"},
]

# ─── VIBRATIONS FIT ─────────────────
# When set, DataPlotter runs the modal fit per run after preprocessing and
# injects `modal_<mode>_f0/_zeta/_f0_sigma/_zeta_sigma` constant channels.
# Set to None to disable (workflow runs as before, no modal artefacts).

VIBRATIONS_FIT = {
    "method": "lorentzian_combined",   # or "body4dof"
    "fmin": 3.5,
    "fmax": 12.5,
    "nperseg": NPERSEG,                # shared with PSD plots below
    "displacement_mode": False,
    "expected_freqs": {
        "heave": (4.0, 6.5),
        "pitch": (7.0, 10.5),
        "roll":  (5.0, 7.5),
        "warp":  (8.5, 12.5),
    },
    "event": EVENT,
    "show_plots": True,               # auto-emits diagnosis plot per run
    # Segment block-bootstrap CIs on (fn, zeta, amp_front, amp_rear).
    # 400 draws x ~50 ms per fit adds ~20 s per run to the pipeline;
    # in return the modal_evolution bands reflect the true (asymmetric)
    # parameter uncertainty and honestly widen for unreliable modes.
    "bootstrap_ci": True,
    "bootstrap_n": 400,
    "bootstrap_seed": 0,
}

# ─── POWERPOINT ───────────────────────────────────────────────────────────────
EXPORT_TO_POWERPOINT  = False
POWERPOINT_TEMPLATE   = resolve_template_path("template.pptx")
POWERPOINT_OUTPUT     = _OUTPUT_DIR / "DIL_Ride_Report.pptx"
POWERPOINT_START_SLIDE = 4

# Secondary deck for modal evolution artefacts. The `powerpoint_exports` list
# accepts an unlimited number of `(template, output, export_map[, start])`
# tuples in addition to the legacy single-template fields.
MODAL_POWERPOINT_TEMPLATE = resolve_template_path("template.pptx")
MODAL_POWERPOINT_OUTPUT   = _OUTPUT_DIR / "Modal_Evolution_Report.pptx"

# ─── WAVEFORM PLOTS ───────────────────────────────────────────────────────────


WAVEFORM_PLOT_DEFINITIONS = [
    WaveformPlot(
        name="DIL TELEM",
        channels=('SM', 'gVert', 'PMGUK', ('vCar', 'NGear'), 'aSteerWheel', ('rThrottle', 'pBrakeF')),
        axis_limits=((-0.2, 1.2), (-3, 3), (-360, 360), ((60, 400), (-1, 9)), (-180, 180), ((None, None), (None, None))),
        reference_lines=(None, None, (-350, 0, 350), None, (0,), None),
        subplot_heights=(0.15, 0.2, 0.3, 0.5, 0.3, 0.3),
    ),
    # WaveformPlot(
    #     name="Prod Forces",
    #     channels=(('vCar', 'NGear'), 'FPushrodFL', 'FPushrodFR', 'FPushrodRL', 'FPushrodRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
    # WaveformPlot(
    #     name="Damper Displacements",
    #     channels=(('vCar', 'NGear'), 'xDamperFL', 'xDamperFR', 'xDamperRL', 'xDamperRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
    # WaveformPlot(
    #     name="Damper Variations",
    #     channels=(('vCar', 'NGear'), 'xDamperVarFL', 'xDamperVarFR', 'xDamperVarRL', 'xDamperVarRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
    # WaveformPlot(
    #     name="Prod Force Variations",
    #     channels=(('vCar', 'NGear'), 'FProdVarFL', 'FProdVarFR', 'FProdVarRL', 'FProdVarRR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None, None),
    #     reference_lines=(None, None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8, 0.8),
    # ),
    # WaveformPlot(
    #     name="Vertical Accelerations",
    #     channels=(('vCar', 'NGear'), 'gVert', 'gVertF', 'gVertR'),
    #     axis_limits=(((None, 400), (-1, 9)), None, None, None),
    #     reference_lines=(None, None, None, None),
    #     subplot_heights=(0.8, 0.8, 0.8, 0.8),
    # ),
]

# ─── SCATTER PLOTS ────────────────────────────────────────────────────────────


SCATTER_PLOT_DEFINITIONS = [
    ScatterPlot("Front Heave",             "xDamperAvgF",   "FPRodAvgF",
                best_fit=[('y', -7500, None), ('y', None, -9000)]),
    ScatterPlot("Front Roll",              "xDamperDeltaF", "FPRodDeltaF",          best_fit=[('x', None, None)]),
    ScatterPlot("Rear Heave",              "xDamperAvgR",   "FPRodAvgR",
                best_fit=[('y', None, 13000)]),
    ScatterPlot("Rear Roll",               "xDamperDeltaR", "FPRodDeltaR",          best_fit=[('x', None, None)]),
]

# ─── PSD PLOTS ────────────────────────────────────────────────────────────────
PSD_PLOT_DEFINITIONS = [
    # ── Ride modes from pushrod forces ────────────────────────────────────────

    PsdPlot("Heave Mode PSD - abs",  "FPRodHeave", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(4, 7),                 log_scale=False, nperseg=NPERSEG),
    PsdPlot("Pitch Mode PSD - abs",  "FPRodPitch", axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(7, 11),                log_scale=False, nperseg=NPERSEG),
    PsdPlot("Roll Mode PSD - abs",   "FPRodRoll",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=[(4, 7), (9, 12)],      log_scale=False, nperseg=NPERSEG),
    PsdPlot("Warp Mode PSD - abs",   "FPRodWarp",  axis_limits=[(0, 30), (1e4, None)], lorentz_fit=(20, 30),               log_scale=False, nperseg=NPERSEG),

    PsdPlot("FPushrod FL PSD - ungated",  "FPushrodFL", axis_limits=[(0, 20), (1e4, None)], annotate_at=(4, 9), log_scale=False, nperseg=NPERSEG),
    PsdPlot("FPushrod FR PSD - ungated",  "FPushrodFR", axis_limits=[(0, 20), (1e4, None)], annotate_at=(6),    log_scale=False, nperseg=NPERSEG),
    PsdPlot("FPushrod RL PSD - ungated",  "FPushrodRL", axis_limits=[(0, 20), (1e4, None)], annotate_at=(5, 9), log_scale=False, nperseg=NPERSEG),
    PsdPlot("FPushrod RR PSD - ungated",  "FPushrodRR", axis_limits=[(0, 20), (1e4, None)], annotate_at=(5, 9), log_scale=False, nperseg=NPERSEG),

    # ── Vertical chassis accelerations ─────────────────────────────────────────
    PsdPlot("Front Vertical Acceleration PSD", "gVertF", axis_limits=[(0, 30), (None, None)], lorentz_fit=[(7, 11)],                            nperseg=NPERSEG),
    PsdPlot("Rear Vertical Acceleration PSD",  "gVertR", axis_limits=[(0, 30), (None, None)], lorentz_fit=[(4, 7), (7, 11)],                    nperseg=NPERSEG),
    PsdPlot("Front Vertical Acceleration PSD - ABS", "gVertF", axis_limits=[(0, 30), (None, None)], lorentz_fit=[(7, 11), (13, 19)], log_scale=False, nperseg=NPERSEG),
    PsdPlot("Rear Vertical Acceleration PSD - ABS",  "gVertR", axis_limits=[(0, 30), (None, None)], lorentz_fit=[(4, 7), (7, 11)],   log_scale=False, nperseg=NPERSEG),

    PsdPlot("hRideF PSD", "hRideF (raw)", axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 9, 15), nperseg=NPERSEG),
    PsdPlot("hRideR PSD", "hRideR (raw)", axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 14.5),  nperseg=NPERSEG),

    PsdPlot("hRideF PSD - abs", "hRideF (high)", axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 9, 15), log_scale=False, nperseg=NPERSEG),
    PsdPlot("hRideR PSD - abs", "hRideR (high)", axis_limits=[(0, 30), (1e-4, None)], annotate_at=(5.5, 14.5),  log_scale=False, nperseg=NPERSEG),
]

# ─── BAR PLOTS (modal parameters, per-run with sigma errorbars) ───────────────
# One bar plot per (parameter, mode) — keeps every mode on its own axis so
# the per-session bars stay legible when RED and BLUE consolidated runs are
# rendered side-by-side.

BAR_PLOT_DEFINITIONS = [
    BarPlot(
        name="Modal Frequency (Heave)",
        metrics=("modal_heave_f0",),
        error_metrics=("modal_heave_f0_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Frequency (Pitch)",
        metrics=("modal_pitch_f0",),
        error_metrics=("modal_pitch_f0_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Frequency (Roll)",
        metrics=("modal_roll_f0",),
        error_metrics=("modal_roll_f0_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Frequency (Warp)",
        metrics=("modal_warp_f0",),
        error_metrics=("modal_warp_f0_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Damping Ratio (Heave)",
        metrics=("modal_heave_zeta",),
        error_metrics=("modal_heave_zeta_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Damping Ratio (Pitch)",
        metrics=("modal_pitch_zeta",),
        error_metrics=("modal_pitch_zeta_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Damping Ratio (Roll)",
        metrics=("modal_roll_zeta",),
        error_metrics=("modal_roll_zeta_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Damping Ratio (Warp)",
        metrics=("modal_warp_zeta",),
        error_metrics=("modal_warp_zeta_sigma",),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Amplitudes (Heave)",
        metrics=("modal_heave_amp_front", "modal_heave_amp_rear"),
        error_metrics=("modal_heave_amp_front_sigma", "modal_heave_amp_rear_sigma"),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Amplitudes (Pitch)",
        metrics=("modal_pitch_amp_front", "modal_pitch_amp_rear"),
        error_metrics=("modal_pitch_amp_front_sigma", "modal_pitch_amp_rear_sigma"),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Amplitudes (Roll)",
        metrics=("modal_roll_amp_front", "modal_roll_amp_rear"),
        error_metrics=("modal_roll_amp_front_sigma", "modal_roll_amp_rear_sigma"),
        default_aggregation="first",
    ),
    BarPlot(
        name="Modal Amplitudes (Warp)",
        metrics=("modal_warp_amp_front", "modal_warp_amp_rear"),
        error_metrics=("modal_warp_amp_front_sigma", "modal_warp_amp_rear_sigma"),
        default_aggregation="first",
    ),
]

# ─── POWERPOINT EXPORT MAPS ───────────────────────────────────────────────────
# Reference format: "type/Plot Name" — auto-converts to filename.

POWERPOINT_EXPORT_MAP = [
    Slide("double_plot", "psd/Heave Mode PSD",                  "psd/Pitch Mode PSD"),
    Slide("double_plot", "psd/Roll Mode PSD",                   "psd/Warp Mode PSD"),
    Slide("double_plot", "psd/Front Vertical Acceleration PSD", "psd/Rear Vertical Acceleration PSD"),
]

MODAL_POWERPOINT_EXPORT_MAP = [
    # Per-mode frequency bars (RED vs BLUE per session, modal fit value + sigma).
    Slide("double_plot", "bar/Modal Frequency (Heave)",
                         "bar/Modal Frequency (Pitch)"),
    Slide("double_plot", "bar/Modal Frequency (Roll)",
                         "bar/Modal Frequency (Warp)"),
    Slide("double_plot", "bar/Modal Damping Ratio (Heave)",
                         "bar/Modal Damping Ratio (Pitch)"),
    Slide("double_plot", "bar/Modal Damping Ratio (Roll)",
                         "bar/Modal Damping Ratio (Warp)"),
    Slide("double_plot", "bar/Modal Amplitudes (Heave)",
                         "bar/Modal Amplitudes (Pitch)"),
    Slide("double_plot", "bar/Modal Amplitudes (Roll)",
                         "bar/Modal Amplitudes (Warp)"),
    # plot_modal_evolution emits per-mode comparison figures into plots/modal/.
    Slide("double_plot", "modal/modal_evolution_line_f0_heave",
                         "modal/modal_evolution_line_f0_pitch"),
    Slide("double_plot", "modal/modal_evolution_line_f0_roll",
                         "modal/modal_evolution_line_f0_warp"),
    Slide("double_plot", "modal/modal_evolution_line_zeta_heave",
                         "modal/modal_evolution_line_zeta_pitch"),
    Slide("double_plot", "modal/modal_evolution_line_zeta_roll",
                         "modal/modal_evolution_line_zeta_warp"),
    Slide("double_plot", "modal/modal_evolution_line_amp_heave",
                         "modal/modal_evolution_line_amp_pitch"),
    Slide("double_plot", "modal/modal_evolution_line_amp_roll",
                         "modal/modal_evolution_line_amp_warp"),
]

# Build the list of secondary exports. Legacy single-template fields are still
# supported (see Run_*.py history); below we drive everything through the new
# `powerpoint_exports` list.
POWERPOINT_EXPORTS = [
    (POWERPOINT_TEMPLATE,       POWERPOINT_OUTPUT,       POWERPOINT_EXPORT_MAP,       POWERPOINT_START_SLIDE),
    (MODAL_POWERPOINT_TEMPLATE, MODAL_POWERPOINT_OUTPUT, MODAL_POWERPOINT_EXPORT_MAP, 2),
] if EXPORT_TO_POWERPOINT else None

# ─── COMPARE GROUPING ─────────────────────────────────────────────────────────
# Pass GROUP_BY="group" to `plot_modal_evolution` to split the line plot into
# multiple linestyles per group (e.g. RED vs BLUE car, or 26R07 vs 26R08).
# When all runs share one group (or none have a `group` key), the plot
# collapses to a single series per mode.
GROUP_BY = "group"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plotter = run_workflow(
        WORKFLOW_NAME,
        title=f"{WORKFLOW_NAME.upper()} PLOT GENERATION",
        runs=RUNS,
        root_folder=_INPUT_DIR,
        output_dir=_OUTPUT_DIR,
        waveforms=WAVEFORM_PLOT_DEFINITIONS,
        scatters=SCATTER_PLOT_DEFINITIONS,
        psds=PSD_PLOT_DEFINITIONS,
        bars=BAR_PLOT_DEFINITIONS,
        powerpoint_exports=POWERPOINT_EXPORTS,
        vibrations_fit=VIBRATIONS_FIT,
        fig_size={"waveform": (20, 10), "default": (10, 8)},
    )
    if plotter is not None and getattr(plotter, "modal_results", None):
        # One figure per mode. `compare_by="session"` aligns runs from each
        # group (RED, BLUE) on a shared x-axis keyed by session token (P1,
        # P2, P3, Q, GP) so the two cars overlay for direct comparison.
        for mode in ("Heave", "Pitch", "Roll", "Warp"):
            plot_modal_evolution(
                plotter,
                modes=(mode,),
                name_suffix=mode.lower(),
                group_by=GROUP_BY,
                compare_by="session",
                include_consolidated=True,
                line_ci=True,
                bars=True,
            )

