# Improvement backlog

A catalog of cleanups, bug fixes and quality-of-life improvements, split into
steps that can be done **one at a time**. Nothing here is implemented yet.

Scanned 2026-10-08. Line numbers are approximate and will drift, so search by
symbol name or quoted text instead.

---

## How to use this file

1. Work through the phases in order. Phase 0 is the safety net that later
   steps rely on.
2. Do **one step per session/PR**. Each step lists where to look, what to do
   and how to check it.
3. Steps tagged **[verified]** were confirmed by reading the code. Steps
   tagged **[reported]** came from an automated scan. Reproduce those before
   changing anything.
4. Steps tagged **[decision]** need the user's input first, either because the
   output will look different or because a user workflow changes.
5. When a step is done, change its `[ ]` to `[x]` and add a one-line note
   (date, commit, anything surprising).

### Checks after every step

```powershell
pytest                     # 67+ tests; also writes coverage.xml
ruff check .
python Run_Correlation.py --dry-run
```

- **Refactors** (behaviour shouldn't change): run
  `python tools/golden_regression.py verify Run_Correlation` and expect 0 PNG
  diffs. Take the snapshot first (step 0.2).
- **Visual changes:** take a new snapshot after the user approves the new look.
- Coverage guard: if a file's coverage drops by more than 2 pp, rebaseline as
  described in `tests/golden/README.md`.

### Codebase conventions to keep

- Log with `from engine.logger import log`. `logging.getLogger(__name__)` is
  silently dropped.
- The tool logs INFO to stderr, so measure exit codes with
  `cmd 1>$null 2>$null; $LASTEXITCODE`, never through a `2>&1 |` pipe.
- `Run_*.py` are user-edited config and are excluded from `ruff format`. Keep
  their aligned layout.
- Calculated-channel lambdas must fit on one physical line or be wrapped in
  `calc_channel("Self", "dep1", ...)`, with the channel's own name listed first
  when it can also be native.
- Every symbol in `engine.__all__` needs a docstring
  (`tests/test_public_api_docstrings.py`).

---

## Phase 0: Safety net

### [ ] 0.1 Fix the CI test job [verified]
- **Problem:** `.github/workflows/ci.yml` job `test` installs
  `requirements.txt` + `pytest` only. `pyproject.toml` `addopts` passes
  `--cov=engine`, which needs `pytest-cov`, so the job should fail with
  "unrecognized arguments".
- **Do:**
  - Install `".[dev]"` (or add `pytest-cov`) in the `test` job.
  - Pin ruff to match `.pre-commit-config.yaml` (`v0.6.9`).
  - Add `ruff format --check .` to the lint job.
  - Remove the stale "auto-bootstrap venv re-exec" comment.
- **Check:** CI is green on all four OS × Python legs.

### [ ] 0.2 Take golden snapshots before any refactor
- **Do:**
  - Run `python tools/golden_regression.py snapshot Run_Correlation`, plus one
    box-plot runner and one PSD-heavy runner.
  - Keep the JSONs local or commit them. They depend on local `Data/`.
- Also clean up `tools/golden_regression.py`:
  - `_resolve_expected_pptx` reads the deleted `EXPORT_TO_POWERPOINT` flag.
  - Its docstring still mentions the win32com fallback.
  - `_hash_file_bytes` is unused.
  - The module docstring says PPTX is "raw bytes", but it's actually a
    ZIP-entry hash.

### [ ] 0.3 Unit tests for the config layer
These are pure functions and need no telemetry files. Add
`tests/test_plot_runtime.py`:
- `build_plot_groups`: 8-tuple, box-grid expansion
- `workflow_config`
- `validate_config`: duplicates, missing file, consolidated entries
- `validate_export_map`
- `Slide` / `_plot_ref_to_filename` / `_resolve_export_map`
- `_session_sort_key`, `_session_token_from_stem`
- `_resolve_consolidate_by`: a regex with no capture group gives `ValueError`
- `_make_consolidated_entry` naming
- `_expand_folder_runs`, using `tmp_path`

Then `tests/test_dataplotter_pure.py`:
- `make_unique`, `_find_split_column`, `_format_split_key`,
  `_split_by_required_columns`
- `_extract_calculated_dependencies`, `collect_referenced_channels`
- `compute_slap_alignment`, `_scan_for_track_code`

And for `datafunctions`:
- `apply_calculated_channels`: duplicate-column regression, required gating
- `apply_channel_mappings`, `resample_to_uniform_rate`, `detect_sample_rate`
- `add_units_to_label`, `format_gate_text`

### [ ] 0.4 Invariant test for `channel_config`
- For every `CALCULATED_CHANNELS` entry, check one of:
  - its source spans one physical line, or
  - its `__dls_deps__` contains every `df["X"]` token it reads.
- This would have caught 1.11.

---

## Phase 1: Confirmed bugs (small, high value)

### [ ] 1.1 `BoxPlotGrid(render_mode="grid")` crashes [verified]
- **Where:** `engine/plotting/generate_bar_box.py` `_generate_boxplot_grid`,
  which has `from .plot_definitions import _normalise_gate_list`.
- **Problem:** the relative import resolves to the non-existent
  `engine.plotting.plot_definitions`.
- **Do:** change it to `from ..plot_definitions import ...` and move it to
  the top of the module.
- **Check:** `Run_BoxPlots.py` grid plots render.

### [ ] 1.2 Garbled text in PSD legends [verified]
- **Where:** `generate_psd_hist.py`, `lbl = f"{group_name.upper()} â€” {ch}"`.
  A nearby comment also has `Î”f`.
- **Do:** replace with `—` (or ASCII `-`) and fix the comment.
- Also fix `Run_RideReport.py`, which has `?f` instead of `Δf`
  in the `NPERSEG` comment.
- **Note:** this is a visual change, but an unambiguous fix.

### [ ] 1.3 `lorentz_fit` documented with the wrong meaning [verified]
- **Problem:**
  - `_coerce_lorentz_fit` in `plot_definitions.py` takes **`(f_lo, f_hi)`
    windows**, and every runner uses it that way (`lorentz_fit=(4, 7)`).
  - The `PsdPlot` docstring and `docs/plot-reference.md` describe
    `(f0, half_width_hz)` tuples, plus a bare-float list (`[5.0, 15.0]`).
  - That docs example raises `TypeError`, and `(5.0, 1.0)` raises
    `ValueError`.
- **Do:** fix the docstring and the docs example. Optionally accept a bare
  float `f0` and expand it to `(0.75·f0, 1.25·f0)`.

### [ ] 1.4 Export maps reference plots that don't exist [verified]
- **Problem:** these map entries match no plot:
  - `Run_Bumpstop.py`, `Run_RideDIL.py` and `Run_RideReport.py` reference
    `psd/Heave Mode PSD`. Those runners only define `... - abs`,
    `- gated` and `- ungated` variants.
  - `Run_Correlation.py` references `scatter/Steering Moment`, which is
    commented out.
- **Do:**
  1. Fix the runner maps.
  2. Make `validate_export_map` also report **map entries with no matching
     plot**. Today it only reports the reverse.
  3. Run the check for every `powerpoint_exports` entry, not just the
     single map.
- **Check:** `--dry-run` warns before any data loads.

### [ ] 1.5 Runners delete each other's plots [verified] [decision]
- **Problem:**
  - `Run_DLSCorrelation`, `Run_DILCorrelation` and `Run_OC_Checks` all use
    `WORKFLOW_NAME="correlation"` and `EVENT="26R16SEP"`.
  - `run_plot_job` deletes every `*.png` in each plot-type folder before
    plotting, so each runner wipes the others' plots.
  - `Run_Correlation` and `Run_DLSCorrelation` both write
    `Correlation_Report.pptx`, so with the same event they overwrite each
    other.
  - `Run_Vibrations` shares the `ride_dil` folder with `Run_RideDIL`.
- **Options:**
  - (a) A per-runner subfolder: `outputs/<wf>/<event>/<runner_stem>/`.
  - (b) Only delete PNGs whose names this job would *not* regenerate.
- **Recommended:** (b).

### [ ] 1.6 `annotate_fit_at` reads the wrong fit segment [reported]
- **Where:** `generate_scatter.py`, the
  `for s, ic in zip(slopes_p, intercepts_p): ... break` loop inside
  `annotate_fit_at` handling.
- **Problem:** with multi-segment fits it takes the first segment with a
  non-None slope, ignoring which segment `x_at` falls in.
- **Do:** pick the segment whose x-range contains `x_at`. If none does,
  skip it with a debug log.

### [ ] 1.7 Waveform `show_delta` flags shift after a dropped row [reported]
- **Problem:** `show_delta` is validated per *original* channel, but the
  generator indexes it by *prepared-row* index. When
  `_prepare_waveform_channels` drops a missing channel, every later flag
  shifts by one.
- **Do:** carry the original index through preparation.

### [ ] 1.8 Box-grid jitter points land on the wrong box [reported]
- **Where:** `_render_grid_cell_per_run`.
- **Problem:** point positions come from `enumerate(run_names)`, which
  still counts runs that were skipped for having no data.
- **Do:** enumerate only the runs that were kept.

### [ ] 1.9 Code uses APIs newer than the declared minimum versions [verified]
- **Problems:**
  - `legend.legend_handles` (`dataplotter.py`, `generate_scatter.py`) needs
    matplotlib ≥ 3.7. The pin allows 3.5.
  - `ax.boxplot(labels=...)` (5 sites) is deprecated in matplotlib ≥ 3.9 in
    favour of `tick_labels`. The pytest filter hides the warning.
  - `engine/workflow/color_utils.py` uses `matplotlib.cm.get_cmap`, which was
    removed in 3.11. Use `matplotlib.colormaps[name]`.
  - `vibrations_io.py` calls `np.trapezoid` unconditionally, which breaks on
    numpy < 2. Use the existing `datafunctions._np_trapezoid` shim.
  - The local venv has pandas 3.0.2, but the pin is `<3.0`.
- **[decision]:** raise the minimums (`matplotlib>=3.7`, test and bump the
  pandas ceiling) or add shims.

### [ ] 1.10 Brake-energy integrals can flatline on NaN gaps [verified]
- **Where:** `channel_config._cum_energy_kj`, which feeds `EBrakeFL..RR`.
- **Do:** wrap the integrand in `np.nan_to_num(..., nan=0.0)`, as
  `EPlank_F/R` already do.

### [ ] 1.11 `tLap_Report` declares the wrong dependencies [reported]
- **Problem:** `calc_channel(...)` lists `tLapSolver`, `TimeIntoExport` and
  `vCar`, but the lambda reads `df["tLap_Calc"]`. It only works because
  `vCar` happens to pull in `tLap_Calc`.
- **Do:** add `tLap_Calc` to the declared dependencies.

### [ ] 1.12 Calc channels and filters receive the run name instead of its type [verified]
- **Where:** `DataPlotter.preprocess_data` passes `name` (the run name) as
  the `source_type` argument of `datafunctions.apply_calculated_channels`
  and `apply_filters`.
- **Problems:**
  - The per-source-type branch can never work.
  - A run named like a calc channel (e.g. `"gLat"`) would make `calc_set`
    a lambda instead of a dict, so all calc channels are silently skipped.
- **Also:** `run.get("type", run_name)` in `load_data` falls back to the run
  name, so a run with no `type` silently gets no mappings (see 3.5).
- **Do:** pass `run["type"]`, or remove the per-type keying if nobody
  uses it.

### [ ] 1.13 Logging handlers leak [reported]
- **Problems:**
  - `DataPlotter.plot_data` adds `_AggregatingHandler`, then returns early on
    "No plot types matched" without removing it.
  - `run_plot_job` adds `_RunAggregator` without `try/finally`.
- **Do:** wrap both in `try/finally`.

### [ ] 1.14 Unsafe plot filenames and an incomplete stale-file cleanup [reported]
- **Problems:**
  - `helpers._sanitize_plot_filename` only strips space, `()`, `/` and `\`.
    Names containing `: * ? " < > |` fail on Windows.
  - The `box_grid/` folder is missing from the per-kind stale-PNG cleanup
    list in `run_plot_job`.
- **Do:** strip all Windows-forbidden characters and add `box_grid` to the
  cleanup list.

### [ ] 1.15 Bar error bars disappear without a warning [reported]
- **Problem:** `generate_bar_plots` skips error bars when the secondary axis
  is active (`if errs is not None and ax2 is None`).
- **Do:** draw them on `ax2`, or at least log a warning.

### [ ] 1.16 PowerPoint aspect-ratio check never matches [reported]
- **Where:** `plot_runtime.get_template_plot_aspect_ratios` (or nearby).
- **Problem:** `name.startswith(("scatter_", "psd_", "bar_"))` is tested
  against `"scatter/scatter_x.png"`, so it's always False. `histogram_` is
  also missing from the tuple, which differs from `_is_misc_plot`.
- **Do:** reuse `_is_misc_plot`.

### [ ] 1.17 Edge cases in modal compare and heatmap [reported]
- **Problems:**
  - `modal_plots._build_compare_layout` with any string other than
    `"session"` returns that literal for every run, so all runs collapse
    onto one x position.
  - `generate_heatmap_plots`: `nanmin`/`nanmax` on an all-NaN array after
    gating raises an error.
- **Do:** reject unknown `compare_by` strings with an error, and skip
  empty heatmaps with a warning.

### [ ] 1.18 `tools/validate_config.py` never runs the config [verified]
- **Problem:** it executes the runner with `__name__="_run_config"`, so
  `main()` never runs and the injected `--dry-run` is ignored.
- **Do:** call `mod.main()` after `exec_module`, or delete the tool and
  point users at `--dry-run`.
- **After:** update `tools/README.md`.

### [ ] 1.19 `Run_BoxPlots` box-plot settings replace the defaults [reported]
- **Problem:** the runner's `BOX_PLOT_SETTINGS` replaces the global dict
  instead of merging into it, so unspecified keys fall back to the
  generator's hard-coded defaults.
- **Do:** merge it in `workflow_config` / `run_plot_job`:
  `{**channel_config.BOX_PLOT_SETTINGS, **override}`.

### [ ] 1.20 PSD `show_envelope` can never take effect [reported]
- **Problem:** in `generate_psd_plots`, `run_groups[run["name"]] = [run]`
  puts exactly one run in every group, so the weighted-mean/envelope branch
  never runs.
- **[decision]:** implement grouping (by `group` key or `consolidate`
  source), or remove the field and the dead branch.

---

## Phase 2: Clearer, neater plots

Most of these change how plots look. Do 2.1a and 2.2 first, as
refactors that keep output identical, then the visual steps with user
sign-off.

### [ ] 2.1a Centralise styling constants without changing values
- **Do:** create `engine/plotting/style.py` holding:
  - colours: `INK "#1A1A1A"`, `FRAME_EDGE "#3C3C3C"`,
    `ZERO_LINE "#5E5E5E"`, `REF_LINE "#4A4A4A"`, and the fallback bar
    colour (code `"#3498DB"` vs config `"#2E7D99"`)
  - the info-box style dict:
    `facecolor="white", alpha=0.92, edgecolor=FRAME_EDGE, linewidth=1.4`
  - a font-size ladder (title, axis label, tick, legend, annotation,
    info box)
  - layout constants: callout `min_sep`, offsets, headroom fractions
  - `SAVE_KW`
- Replace the ~70 literal sites in `engine/plotting/*`, `modal_plots.py` and
  `vibrations_io.py`. Remove the duplicate `_INK` definitions.
- **Check:** 0 golden diffs.

### [ ] 2.1b Unify the values [decision]
- One font-size ladder across all plot types. Today axis labels are 14 on
  scatter, 13 on histogram, 12 on bar/box and 9.5 on waveform.
- One zero-line, reference-line and grid style. Modal, PSD minor grid, bar
  `ax2` and scatter3d differ today.
- One units notation: `(unit)` everywhere. Modal and vibrations use
  `[unit]`.
- Show the bar aggregation without brackets (`"metric · max"`) so it isn't
  confused with units.

### [ ] 2.2 One `save_figure` helper
- **Problem:** the `fig.savefig(..., dpi=output_dpi, pad_inches=0.15,
  facecolor="white", bbox_inches="tight"); plt.close(fig)` block is copied
  about 12 times.
- **Do:** add `helpers.save_figure(plotter, fig, subdir, filename)` and use
  it everywhere, including modal and vibrations.
- **Note:** histogram uses `pad_inches=0.05`, and scatter3d/modal have no
  padding. Unifying those is a visual change for 2.1b.

### [ ] 2.3 One matplotlib style setup and the resolved font everywhere
- **Problems:**
  - `vibrations_io._configure_style` copies
    `DataPlotter._configure_plot_style`.
  - `family="Montserrat"` is hard-coded in scatter/PSD text boxes, and
    `PLOT_FONT["family"]` is used in the legend/gate helpers. Both bypass the
    fallback font that `_configure_plot_style` resolves, so machines without
    Montserrat get findfont warnings and mixed fonts.
- **Do:** one `style.apply_rcparams()` that returns the resolved family,
  and use that family everywhere.

### [ ] 2.4 One legend helper and smarter legend placement
- **Problems:**
  - Legend styling is implemented three times: `_add_standard_legend`,
    `_add_waveform_figure_legend` (whose two branches repeat 8 kwargs) and
    `vibrations_io._add_legend`.
  - Bar and box use a fixed `loc="upper right"`, which can cover the
    tallest bar's value label.
  - Scatter3D and modal use unstyled `ax.legend("best")`.
- **Do:**
  - Merge into one helper.
  - Use corner ranking (`_rank_info_corners`) for bar and box.
  - Use the standard style for scatter3D and modal.

### [ ] 2.5 One info-box builder
- **Problem:** the scatter `_display_segment_boxes`, PSD
  `_draw_lorentz_compact_box` and `vibrations_io._add_lorentz_info_box`
  each build `AnnotationBbox`/`VPacker` boxes. The last two render the same
  "Lorentz fit [lo–hi Hz]" format.
- **Also merge:**
  - the duplicated corner ranking (PSD re-implements it, and scatter has
    `_select_trendline_anchor`)
  - `DataPlotter._legend_corner`'s own `loc_map`, which duplicates
    `corner_layout._LOC_TO_CORNER`

### [ ] 2.6 One callout-placement helper, also fixing a units bug
- **Problem:** "sort, convert to display Y, enforce `min_sep`, annotate"
  is copied in scatter `annotate_fit_at`, PSD `annotate_at`, the PSD
  remaining-Lorentz labels and waveform `_render_annotations`.
- **Bug:** nudges are computed in display **pixels** but applied as
  `offset points`, and before `tight_layout` resizes the axes. Spacing
  therefore scales with DPI.
- **Do:** one helper that works in points after layout.

### [ ] 2.7 Plot titles and run context [decision]
- **Problems:**
  - Every dataclass docstring says `name` is the "Plot title and filename
    stem", yet waveform, scatter, PSD, histogram and bar render **no
    title**. Box only shows one through `options["title"]`.
  - No PNG shows the event or runs. Only vibrations has an
    event/run/method subtitle.
- **Options:**
  - a `PLOT_TITLES` setting (`"none" | "name" | "name+event"`)
  - an optional small footer (event · runs · date)
- PPTX decks may not want titles baked in, so the default may differ for
  exported plots.

### [ ] 2.8 Units on every axis
- **Problems:**
  - Scatter3D axes and its Plotly HTML have no units.
  - Waveform delta axes have none, because `"Δ hRideF"` isn't found by the
    exact-match `add_units_to_label`. Strip the `"Δ "` prefix before the
    lookup.
  - Multi-metric bars and bar `ax2` have no y label.
  - The PSD y label uses only the first channel's units. Warn or combine
    when channels differ.
  - `datafunctions.format_psd_ylabel` renders `deg/s²/Hz`. It should be
    `(deg/s)²/Hz`, without italic units.
  - In `channel_config.UNITS_MAP`, `"PMGUK_Deploy (MJ)": "kW"` gives the label
    "PMGUK_Deploy (MJ) (kW)". Fix the name or the unit.
  - `"nwheelr_avg"` doesn't match the channel `nWheelAvg_R`.

### [ ] 2.9 Colour-blind-friendly default colours [decision]
- **Problems:**
  - Auto run colours are tab10 (`_AUTO_COLORS`), so runs 3 and 4 are
    green and red.
  - Modal colours Roll green vs Warp red.
  - The vibrations diagnosis plot uses red `#D70000` against green
    `#00AA55`.
  - The vibrations palette includes `#FFD700`, yellow on white.
- **Do:** one palette source in `engine/workflow/color_utils.py` (e.g.
  Okabe-Ito or a tuned tab10), used by dataplotter, modal and vibrations.
  Remove the three duplicate 8-colour palettes.

### [ ] 2.10 One baseline rule for all deltas
- **Problems:**
  - Waveform and bar deltas use `reference_run_name()`, which only honours
    `"reference"`.
  - The scatter fit box uses `plotter.runs[0]`.
  - `compute_slap_alignment` / `compute_slap_offsets` honour `"baseline"`
    as well.
- **Do:** one `reference_run_name()` honouring both flags, used everywhere.
  Warn when more than one run is flagged.

### [ ] 2.11 Modal uncertainty drawn consistently
- **Problem:** `_line_ci_figure` draws asymmetric bootstrap CIs, but
  `_bar_figure`, `_line_ci_amp_figure` and `_bar_amp_figure` use the
  symmetric sigma even when `*_lo/_hi` exist.
- **Do:** use asymmetric CIs everywhere they exist.

### [ ] 2.12 Heatmap layout
- **Problems:**
  - It doesn't use `_resolve_plot_figsize`.
  - Width grows by 9 inches per run, so 4 runs give about 10,800 px at
    300 dpi.
  - There's no layout call, and the suptitle collides with the per-axis
    titles.
  - In count mode, empty bins show the lowest colormap colour. They should
    be NaN, i.e. blank.
- **Do:**
  - Wrap runs into a grid with at most 2–3 columns.
  - Use `constrained_layout`.
  - Mask empty bins.

### [ ] 2.13 Make the three box-plot modes consistent
- **Problems:**
  - `_generate_boxplot_per_run`, `_aggregated` and `_per_run_aggregated`
    each repeat about 40 lines (settings, axes styling, jitter, suptitle,
    save).
  - `_per_run_aggregated` hard-codes `"#2C3E50"` and bypasses
    `_apply_boxplot_artist_style`.
  - `suptitle(y=1.02)` is placed outside the figure.
  - `_MAX_FLIERS = 200` is defined inside a loop.
- **Do:** share the setup, styling and saving code. Each mode then only
  draws its boxes.

### [ ] 2.14 Shorter legends
- **Problems:**
  - Multi-channel PSD has one legend entry per (run, channel). Line style
    already encodes the channel, so use two compact legends: run colours,
    and channel line styles.
  - The modal line legend has `mode × group` entries. Use the same split.

### [ ] 2.15 Waveform reference lines and markers use the shared helpers
- **Problem:** waveform draws its own reference lines (alpha 0.65 left vs
  0.55 right, no value labels), and `_draw_line` /
  `_draw_static_label` / `_draw_inline_label` re-implement
  `helpers._draw_static_markers`.
- **Do:** use the shared helpers so waveforms match the other plot types.

### [ ] 2.16 Scatter3D cleanup [decision]
- **Problems:**
  - Remove the TkAgg tail in `Run_Correlation.main()` and
    `Run_DLSCorrelation.main()`. It renders twice, ignores `--only` and
    `--no-open`, and blocks on `plt.show(block=True)` in
    `generate_scatter3d_plots`.
  - It has no grid style and no units.
  - `include_plotlyjs="cdn"` makes the HTML need internet access, yet the
    docstring says "self-contained".
- **Do:**
  - Replace the interactive window with a `--interactive-3d` CLI flag or
    a `Scatter3DPlot(show=True)` field, handled once in the engine.
  - Add grid style and units.
  - Fix the Plotly bundling or its docstring.
- After this, remove the related troubleshooting row from README.md.

### [ ] 2.17 Fewer global pyplot side effects
- **Problems:**
  - Generators call `plt.tight_layout(...)`; use `fig.tight_layout` or
    `constrained_layout` instead.
  - `matplotlib.use("Agg")` runs at import time in `dataplotter.py`,
    `modal_plots.py` and `vibrations_io.py`.
- **Do:** switch to the `fig.` calls, and move the backend choice to the
  entry point (`run_from_config`, CLI). Do this together with 2.16.

### [ ] 2.18 Small clarity fixes
- When `x_channel` is missing, the waveform x label silently becomes
  `"Sample"`. Log a warning.
- Render the Nyquist label as mathtext `$f_{nyq}$`.
- Unify the value-label rotation rule: bar rotates at ≥ 5 runs, modal at
  `n_x·n_series ≥ 8`.
- Fix the default `fig_size` mismatch: the `DataPlotter.__init__` default
  dict (waveform 15.5) vs `DEFAULT_FIG_SIZE` (17.28). The latter has no
  `"box"` key, and the `__init__` default is unreachable from runners.

---

## Phase 3: User quality of life

### [ ] 3.1 Consistent runner headers and comments
- **Wrong docstrings:**
  - `Run_Bumpstop.py` and `Run_RideReport.py` say "Ride/DIL workflow".
  - `Run_OC_Checks.py` and `Run_DLSCorrelation.py` say "Correlation
    workflow".
- **Copy-paste comments:**
  - Bumpstop's PPTX example points at `DIL_Ride_Report.pptx`.
  - Bumpstop has a waveform named `"DIL TELEM"`.
  - RideReport says "512 samples", but `NPERSEG = 400`. `PSD_MIN_AVERAGES`
    does nothing unless NPERSEG is `"auto"`.
  - Bumpstop and RideDIL comments say `nperseg=256`.
- **Typos:** `"COG Sensitivies"`, `"Aero Sensitivites"` (`Run_OC_Checks.py`).
- **Header width:** the `POWERPOINT EXPORT` rulers are about 95 characters;
  the others are 80. OC_Checks calls the section `POWERPOINT`, and RideDIL
  puts `NPERSEG` under the PowerPoint header.
- **[decision]:** trim old commented-out `RUNS` and plot blocks? Users may
  rely on them as quick toggles, so ask first.

### [ ] 3.2 Plot lists that are declared but never passed
- **Problem:** `BOX_PLOT_DEFINITIONS = []` and `HEATMAP_PLOT_DEFINITIONS = []`
  exist in `Run_Correlation`, `Run_DLSCorrelation` and `Run_OC_Checks`, but
  aren't passed to `run_workflow`. Filling them in silently does nothing.
- **Do:** pass them or delete them. Step 3.4 removes this whole class of
  mistake.

### [ ] 3.3 The engine resolves folders; fewer arguments per runner
- **Do:**
  - `run_workflow(workflow, event=EVENT, ...)` resolves the folders itself.
    Runners stop calling `get_workflow_dirs` and passing
    `root_folder`/`output_dir`.
  - Default the title to the runner and event, not
    `"CORRELATION PLOT GENERATION"` for four different runners.
  - Optional: pick up `RUNS`, `*_PLOT_DEFINITIONS` and `POWERPOINT_*`
    by convention via `run_workflow_from_module(globals())`.
  - Delete the then-dead `_WORKFLOW_MAP` in `workflow_config`. It also
    still maps `dampers`.
- **Enables:** `--event` (3.8) and per-runner folders (1.5).

### [ ] 3.4 Validate run dictionaries
- **Problems:**
  - Unknown keys are ignored: `"nlpa"` loads the whole file, `"colour"`
    gets an auto colour, `"refrence"` is ignored.
  - A missing `type` is silently accepted, and the mappings are then
    skipped (see 1.12).
  - `nlap` + `nrun` together: `nlap` is silently ignored.
  - Multiple `reference: True` runs: the first one silently wins.
  - Dead code in `validate_config`: `if not run.get("color"): pass`.
- **Do:**
  - Add an allowlist of about 20 known keys with `difflib` "did you mean"
    hints. Known keys include `name`, `file`, `type`, `color`, `colors`,
    `color_range`, `nrun`, `nlap`, `best_n`, `reference`, `baseline`,
    `group`, `split_by`, `consolidate`, `consolidate_by`,
    `consolidated_name`, `folder`, `filetype`, `contains`, `name_prefix`
    and `use_python_engine`.
  - Make a missing `type` an error.
  - Warn on `nlap` + `nrun`, and on more than one reference run.

### [ ] 3.5 Friendlier plot-definition errors
- **Problems:**
  - An unknown field raises the raw `TypeError: __init__() got an unexpected
    keyword argument`, with no plot name.
  - `PsdPlot(nperseg="atuo")` raises `invalid literal for int()` with no plot
    context.
  - Nothing detects duplicate plot names, or names that turn into the same
    filename.
- **Do:** wrap or catch these to report the plot name plus the closest
  valid field.

### [ ] 3.6 One channel-reference walker
- **Problem:** three implementations cover different things:
  - `dataplotter.collect_referenced_channels` has no `box_grid`,
    histogram/bar gates or `highlight_zones`. Every Run_BoxPlots grid
    therefore escapes the typo check.
  - `DataPlotter._get_required_source_columns` has no bar
    `error_metrics`.
  - `plot_runtime._plot_referenced_channels` has no gates, markers or
    `color_gate`, and hand-parses bar metrics.
- **Do:** one walker covering every kind and field, used by all three.

### [ ] 3.7 Command-line additions
- **Problems with existing flags:**
  - `--only` is an exact match, and gives no warning when nothing matches.
    Add substring/glob matching plus a "no plots matched — did you mean"
    warning.
  - A partial run (`--only`/`--runs`) currently re-exports and overwrites
    the full PPTX. Skip the export, or write `<name>_partial.pptx`.
- **New flags:**
  - `--type psd scatter` (`plot_data(plot_types=)` already supports this)
  - `--list-runs` (expanded names and files, without loading data)
  - `--event CODE` (needs 3.4)
  - `--verbose` / `--quiet`
  - `--log FILE` (`logger.configure(log_file=)` exists but isn't wired)
  - `--no-pptx`

### [ ] 3.8 Clearer console output
- **Problems:**
  - The output path prints three times, while PPTX paths show only the
    filename and the quality-report path is missing from the summary.
  - There are two warning summaries (`plot_data`, then the RUN SUMMARY).
  - The data-quality summary always prints, because `has_issues` is always
    true (the Summary section is never empty), and it scrolls away before
    plotting.
  - Missing channels are reported three times.
  - Logging breaks the tqdm bars.
  - `print` (stdout) and `log` (stderr) interleave out of order when
    redirected.
  - Non-ASCII text (`•`, `—`, `…`, `²`) still reaches cp1252 consoles.
  - Export failures use `traceback.print_exc()`.
  - `--dry-run` "missing in" ignores `CHANNEL_MAPPINGS` and
    `CALCULATED_CHANNELS`, so it gives false positives for `SM`,
    `FPRodAvgF` and similar. It also assumes the header is on line 2 and
    uses a fake size estimate.
- **Do:**
  - Print one final summary block listing every artefact path.
  - Show the quality summary only when there are issues, and move it to
    the end.
  - Use `logging_redirect_tqdm`, ASCII symbols and `log.exception`.
  - Fix the dry-run channel check.

### [ ] 3.9 PowerPoint export
- **Problems and fixes:**
  - `Slide()` doesn't check its layout name or image count. Validate both
    against the layouts it knows. Its docstring is also wrong: the layout
    is not a template layout name.
  - The filename slug is implemented three times (`_plot_ref_to_filename`,
    inside `validate_export_map`, `helpers._sanitize_plot_filename`). Keep
    one.
  - There's no way to opt a plot out of the deck. Add `export=False`, which
    also quiets the orphan warnings (e.g. `DLS Diagnostics`).
  - When a mapped plot is missing, `_delete_pictures` has already emptied
    the slide. Keep the template picture, or add a "missing: <name>" text
    box.
  - If the template has too few slides, the rest are skipped. Append them
    instead.
  - Auto-fill slide titles from the plot names (blank decks use layout 6
    with no title).
- The picture-box parsing is duplicated between
  `get_template_plot_aspect_ratios` and `_export_via_pptx._picture_boxes`.

### [ ] 3.10 `Run_Vibrations` on the shared command line
- **Problems:**
  - No `--runs` or `--no-open`, no validation and no run summary.
  - `FS = 100` duplicates `RESAMPLE_RATE`.
  - It has its own `OPEN_OUTPUT_FOLDER` and uses `print("#"*60)`.
  - It shares `WORKFLOW_NAME="ride_dil"` with RideDIL.
- **Do:** use `parse_plot_cli`, the logger, and its own `vibrations`
  workflow folder.

### [ ] 3.11 `Run_RideReport` overwrites every event [decision]
- **Problem:** it has no `EVENT`; `get_workflow_dirs(WORKFLOW_NAME)` plus the
  hard-coded `"GP_Comparison_*"` output names and `"event": "GP_COMPARISON"`
  mean every event overwrites the same outputs.
- **Do:** add `EVENT` (or a comparison label) to the output folder and file
  names.

### [ ] 3.12 Optional extras
- `plots/index.html`: thumbnails by type, plus links to the quality report
  and decks. All the paths are already known in `run_plot_job`.
- Stable run colours: when `color` is missing, hash the run name or
  `group`, so a run keeps its colour across runners.
- Detect `EVENT` from filenames (`26R16SEP_...`) or the newest input folder.
- `--last`: re-run with the previous arguments, stored in
  `outputs/.last_run.json`.
- `Run_BoxPlots`: replace the 10 hand-listed files with folder shorthand.
  User decision.
- Console scripts for `Run_DILCorrelation` / `Run_DLSCorrelation`, or
  document why they don't have one.

---

## Phase 4: Engine deep-clean (no behaviour change)

Check every step with 0 golden diffs.

### [ ] 4.1 Delete dead code
- **DataPlotter:** the 8 `generate_*` class-attribute bindings. `plot_data`
  calls the module functions directly.
- **Re-export shims:** the `# noqa: F401` "back-compat" imports in
  `dataplotter.py` and `plot_runtime.py` have no external users. Keep the
  names that are used locally.
  - Repoint the lazy `from .plot_runtime import _interpolate_two_colors`
    in `_expand_split_runs` to `engine.workflow.color_utils`, imported at
    the top. That also removes its duplicate fallback palette.
- **channel_config:**
  - the `DEFAULT_FILTERS = CORRELATION_FILTERS = BOXPLOT_FILTERS = ...`
    aliases. Update `docs/architecture.md` at the same time.
  - `gLatAbs`, a duplicate of `gLat_Abs`
  - `FILTERS` keys `FPRodHeave_Shape`, `FPRodPitch_Shape` and `CosPhi`
    (the calc channel is `CosPhi_Calc`)
  - identity transforms like `"FPushrodRL": lambda x: x`
- **plot_runtime:**
  - the unused `_resolve_export_map(plot_definitions)` parameter
  - the dead `if not run.get("color"): pass` and redundant `elif` in
    `validate_config`
- **dataplotter:** `keep_runs = []` in `_build_consolidated_runs` is
  overwritten before use.
- **plotting:**
  - unused `plotter` parameters on `_prepare_scatter_xy`,
    `_build_gradient_segment_labels`, `_parse_eq_list_to_segments`,
    `_compute_segment_*`, `_normalize_waveform_row_spec` and
    `_normalize_waveform_reference_lines`
  - the unused `runs` parameter of `modal_plots._resolve_group_colors`
  - the dead `x_channel_available = True`
  - imports inside loops (`mpatches`, `defaultdict`)
- **Mid-edge corners:** `_INFO_CORNER_XY` keeps 4 mid-edge entries that are
  never used for placement. Remove them once `_legend_corner_from_bbox` no
  longer needs them.
- **tools:** `validate_config.load_module_from_path`;
  `argparse(description=__doc__)` with no module docstring in
  `validate_config.py` and `organise_data.py`.
- **Stale docstrings:**
  - `modal_plots` mentions `get_plotter` and "does not return the plotter".
  - The `helpers.py` module docstring mentions "Prompt 10" and mixins.
  - The `label_records` type hint says 5-tuple; it's actually 6.

### [ ] 4.2 The "dampers" leftovers [decision]
- **Where:**
  - `_WORKFLOW_MAP["dampers"]`
  - `DAMPER_INPUT_DIR` / `DAMPER_OUTPUT_DIR`, recreated by `mkdir` on every
    import
  - `DAMPER_FILTERS`
  - the `run_workflow` docstring
  - the `pyproject` description ("…damper plotting toolkit")
  - `Data/{inputs,outputs}/dampers/`
- **Options:** delete them all, or restore a runner with
  `git show b6d08d9^:Run_Dampers.py`.

### [ ] 4.3 Merge duplicated code
- **Highest risk first:** PSD smoothing in
  `vibrations_io.run_fit_from_arrays` (`_ma3` + `medfilt(5)` + peak
  normalisation) vs `vibrations_bootstrap._smooth_and_normalise`. If one
  changes and the other doesn't, the bootstrap CIs and the point fit drift
  apart.
- **Modal result parsing:** `_as_list` / `_ci_pair` ladders in
  `DataPlotter._run_modal_fits` and `modal_plots._modal_records`.
- **vibrations:**
  - `_peak_normalise` (io) and `_normalise` (body4dof)
  - `MODE_ORDER` ×3
  - `_INK` ×2
- **Bar aggregations:** `_VALID_BAR_AGGS` has a hard-coded fallback copy
  in `normalize_bar_metric_specs`, behind an unnecessary
  `except Exception` guard.
- **Warning aggregators:** `_RunAggregator` (run_plot_job) and
  `_AggregatingHandler` (plot_data). Keep one.
- **Parquet column/alias resolution, 4+ copies:**
  - `_find_parquet_column`, `_find_split_column`,
    `loaders._normalize_parquet_column_aliases`,
    `vibrations_io._resolve_parquet_column`
  - the schema readers: `_get_parquet_schema_columns`,
    `vibrations_io._parquet_schema_columns`, and inline in `_print_dry_run`
- **Rank/value filter:** `_apply_parquet_rank_value_filter` (141 lines)
  implements the multi- and single-value cases separately. Treat a single
  value as a one-element list.
- **Track codes:** `_extract_track_code_from_filename` overlaps
  `_scan_for_track_code(mode="tail")`.
- **Filename sanitising:** `vibrations_io._safe_name` vs
  `helpers._sanitize_plot_filename`.
- **Plotting:**
  - "finite mask, argsort, `np.interp`" appears about 8 times in the
    waveform delta/annotation code. Make it one `_interp_at`.
  - Robust-outlier logging is pasted twice in `generate_scatter_plots`.
  - Histogram and bar call `apply_gate_to_dataframe` directly (histogram
    twice) instead of the cached `_get_filtered_run_dataframe`.

### [ ] 4.4 Import structure
- **vibrations_io:**
  - `sys.path.insert(0, PROJECT_ROOT)` at import is the cause of all 6
    remaining E402s.
  - `from engine.dataplotter import DataPlotter` is only used to read
    `PLOT_FONT` / `GRID_STYLE`. After 2.1a, import `engine.plotting.style`
    instead. That may also remove the circular-import note in
    `engine/__init__.py`.
  - `RESAMPLE_RATE` is used as a default argument value.
- **Unnecessary lazy imports:**
  - `plot_runtime._enforce_channel_typo_check` (dataplotter is already
    imported)
  - `data_quality_report.build_quality_sections`
  - `plot_runtime` importing `plot_definitions` mid-file with
    `# noqa: E402, F401`
- **Double imports:** `workflow_config` imports `channel_config` twice and
  uses `getattr(_cc, X, default)` on attributes that always exist.
- **Import-time side effects:**
  - `channel_config` runs `mkdir` on 9 folders.
  - stdout/stderr are reconfigured twice (`logger.py` and `plot_runtime.py`).
- **Library code calls `SystemExit`:** `run_from_config` and
  `engine/__init__.py` should raise a `ConfigError` and let the runner turn
  it into exit code 1. That keeps the exit-0-on-success guarantee.

### [ ] 4.5 Slim down the `DataPlotter` façade
- **Current size:** the file is 2,306 lines; the class is about 1,850.
- **Layers to remove:**
  - 9 `@property`/setter pairs forwarding to `self.ctx` (about 72 lines).
  - About 32 class-attribute bindings, each used once from one generator.
    Generators can import the helper directly.
  - 19 one-line delegators.
- **Do it in sub-steps:**
  1. Bindings.
  2. Delegators.
  3. Properties: rewrite `plotter.run_data[...]` to `plotter.ctx.run_data`
     (mechanical but wide).
- **Target:** under 1,000 lines.

### [ ] 4.6 Stop swallowing errors
- **Silent `except Exception: pass`:**
  - `apply_calculated_channels` dependency expansion: on failure, required
    channels quietly become optional.
  - `estimate_slap_alignment` / `compute_slap_offsets` scipy refinement.
  - `_get_parquet_schema_columns`: on failure, column projection is
    silently disabled.
  - The legend-bbox helpers and `os.startfile`.
- **Do:** at least `log.debug(..., exc_info=True)`.
- **`load_data`:** one `try` wraps loading *and* nlap/best_n logic, so a
  programming error is reported as "Failed to load run". Narrow it.
- **`_run_modal_fits`:** warnings lose the traceback.
- **Performance:** `_legend_axes_bbox` calls `fig.canvas.draw()` (a full
  render) on every gated plot. Use `fig.canvas.get_renderer()`.

### [ ] 4.7 Sample-rate handling
- **Problems:**
  - Calc lambdas integrate with the global `channel_config.RESAMPLE_RATE`,
    not the run's actual rate.
  - With resampling off (`RESAMPLE_RATE=0`), you get a `ZeroDivisionError`
    that is caught, so the channel silently disappears.
  - Native 1 kHz data integrates with a 100 Hz `dx`, a 10× error.
  - 9 inline `1.0 / RESAMPLE_RATE` calls exist alongside the `_DT()` helper.
  - Hard-coded 100 Hz defaults: `DataPlotter(sample_rate=100)`,
    `detect_sample_rate(default=100.0)`, `calculate_cplv`, and the
    `vibrations_io` CLI `--fs`.
- **Do:** expose the per-run rate to calc lambdas (e.g. a `df.attrs["fs"]`
  convention read by `_DT(df)`), and use `_DT` everywhere.

### [ ] 4.8 One dependency extractor for calc channels
- **Problem:** `apply_calculated_channels` expands dependencies with an
  `inspect.getsource` regex and ignores `__dls_deps__`, while
  `_extract_calculated_dependencies` honours `__dls_deps__` first.
- **Do:** use one function for both.

### [ ] 4.9 Preprocessing robustness
- **Problems:**
  - `preprocess_data` isn't idempotent: sign-flip transforms run again and
    modal columns are re-concatenated. Add a guard.
  - `_load_run_data` reads the whole CSV with `readlines()` and no
    `encoding=`, just to get the header. Read only the first lines,
    explicitly.
  - In `loaders`, `dict(zip(header, units_row))` silently truncates a
    malformed units row. Warn when the lengths differ.

---

## Phase 5: Split up long functions

Do one function per step. Pure code motion with 0 golden diffs. Do 2.4–2.6 first,
because they remove much of the bulk.

| Step | Function | Lines | Split into |
|---|---|---|---|
| [ ] 5.1 | `generate_waveform_plots` (generate_waveform.py) | 671 | x-channel resolution, normalisation ranges, per-row trace, secondary axis, highlight zones, delta panel (primary and secondary are copy-paste), annotations, markers, legend/save |
| [ ] 5.2 | `generate_psd_plots` (generate_psd_hist.py) | 415 | nperseg, per-run curve and Lorentz fit, `annotate_at`, leftover Lorentz labels, Nyquist, save |
| [ ] 5.3 | `generate_scatter_plots` (generate_scatter.py) | 384 | per-run draw (colour-gate / multi / single fit), info boxes, colour-gate legend, `annotate_fit_at`, save |
| [ ] 5.4 | `generate_bar_plots` (generate_bar_box.py) | 313 | aggregation, secondary-axis decision, bar draw, value labels and deltas, legend/save |
| [ ] 5.5 | `modal_plots` `_line_ci_figure` / `_line_ci_amp_figure` (169 / 128) and `_bar_figure` / `_bar_amp_figure` (175 / 166) | ~640 | Each pair is near-identical: make one function parameterised by the series spec |
| [ ] 5.6 | `vibrations_lorentz._fit_subsystem` | 342 | bounds/seed, flat L-BFGS/DE stage, sloped LS polish, covariance/R² |
| [ ] 5.7 | `vibrations_io.run_fit_from_arrays` (239), `bootstrap_modal_fit` (197) | — | PSD prep (shared with 4.3), fit, bootstrap, result assembly |
| [ ] 5.8 | `DataPlotter._run_modal_fits` | 203 | per-run extraction, fit call, channel injection |
| [ ] 5.9 | `datafunctions.plot_scatter_with_multi_fit` (174), `build_quality_sections` (165), `_expand_split_runs` (153), `_expand_folder_runs` (152), `run_from_config` (146), `generate_scatter3d_plots` (142), `_export_via_pptx` (134), `DataPlotter.__init__` (133) | — | as appropriate |

---

## Phase 6: Config, docs, repo hygiene, lint

### [ ] 6.1 Reorganise `channel_config.py` [decision]
- **Problem:** about 550 lines mixing paths (with `mkdir` at import),
  workflow-dir helpers, mappings, units, transforms, plot style
  (`SCATTER_MAX_POINTS`, `BAR_SECONDARY_AXIS_RATIO`, `BOX_PLOT_SETTINGS`),
  calc channels, filters and `TRACK_LENGTHS`.
- **Do:**
  - Move paths and style into the engine, or a `settings.py`.
  - Add a `_COMMON` mapping merged into each source. `PMGUKActual`,
    `rThrottlePedal`, `PPlankWearF`, `EPlankWearLapF` and
    `aUndersteer_aSlip` are repeated across sources.
  - Split `CALCULATED_CHANNELS` into section dicts merged at the end.
  - Use the `_passthrough` helper for the `(raw)`/`(high)`/`(smooth)`
    copies.
  - Flag the "temporary" `-1.16` calibration and the hard-coded 780 kg × 0.45
    in `CLF_Proxy` for review.

### [ ] 6.2 Docs out of date
- **`docs/architecture.md`:**
  - `configure(verbose=)` should be `configure_logging`.
  - PNGs go to `<output_dir>/plots/<kind>/`.
  - The report is `data_quality_report.md`, not `.txt`.
  - `EXPORT_TO_POWERPOINT` was removed.
  - Filenames are `type/type_plot_name.png`.
  - `*_CALCULATED` dicts don't exist; there is only `CALCULATED_CHANNELS`.
  - Run types are missing `FMIOpt`.
  - Missing run keys: `best_n`, `group`, `split_by`, `color_range`,
    `consolidate*`.
  - It still shows the `dampers/` layout.
- **`docs/plot-reference.md`:**
  - Missing fields:
    - Waveform: `annotate_at`
    - Scatter: `dil_friendly`, `robust_threshold`
    - Bar: `error_metrics`, `secondary_axis`, `show_delta`
    - Box / BoxPlotGrid: `options`
  - Wrong statements:
    - `nperseg=None` means auto, not 512.
    - "`annotate_fit_at` only works with single fits" is outdated.
    - Waveform `normalise` is described differently in the doc, the
      dataclass and the Template.
    - The gate operator lists are missing `robust`, `!=` and `outside`.
  - The `lorentz_fit` example is wrong (see 1.3).

### [ ] 6.3 Repo hygiene
- `ARBS with no TB.csv` at the root is byte-identical to the copy in
  `tools/`. Delete the root copy.
- `tools/farb_deadband.py`, `tools/FARB_8000.json` and the CSV are
  **untracked**. Commit them.
- Ignored local leftovers to delete:
  - `_capture/`
  - `dls_correlation_tool.egg-info/`, which is stale: it still lists
    `dls-dampers`, `bootstrap` and `Run_Dampers`
  - the root `plots/` folder. `vibrations_io._plots_dir(None)` writes
    `./plots/vibrations`; fix it to use a real output folder.
  - the root `__pycache__/`
- `Data/outputs` has both `test/` and `tests/`.
- `pyproject` `packages.find include = ["tools*"]`, but `tools/` has no
  `__init__.py`.
- `requirements.txt` calls python-pptx "required", while `pyproject` makes
  it an optional extra. Make the wording match.

### [ ] 6.4 Lint debt in `pyproject.toml`
The ignore-list comment counts are stale.

| Rule | Comment says | Actual count | Action |
|---|---|---|---|
| B904 | 1 | 0 | Remove from the ignore list now |
| E402 | 25 | 6 | All come from the vibrations_io `sys.path` hack; fixed by 4.4 |
| B905 | 29 | 38 | Add `strict=True`, or `strict=False` with a reason. Check `loaders` `dict(zip(header, units_row))` first |
| B023 | 8 | 10 | No live bug (closures are called within the same iteration), but bind loop variables through default arguments |
| B007 | 10 | 7 | Rename unused loop variables to `_` |
| SIM105 | 8 | 9 | Use `contextlib.suppress` (3 sites are in `Run_*`) |
| SIM102, UP031 | 1 each | 1 each | Fix |

**B023 sites:**
- `dataplotter` `_obj` closures and `_ci_pair`
- `modal_plots` `res`
- `generate_psd_hist` `lorentz_by_color`
- `generate_waveform` `fig`
- `vibrations_io` `modes` / `fn`

---

## Suggested order (shortest path to value)

1. 0.1 → 0.2 → 1.1 → 1.2 → 1.3 → 1.4 → 1.10 → 1.18: quick bug fixes.
2. 0.3 → 0.4: tests for the config layer.
3. 2.1a → 2.2 → 2.3: plotting foundations, with identical output.
4. 1.5 + 3.4 (shared output folders, engine-resolved folders), then 3.5 →
   3.7 → 3.8 → 3.9.
5. 2.1b → 2.4 → 2.5 → 2.6 → 2.7 → 2.8 → 2.9: visual consistency, after
   user sign-off.
6. Phase 4, then Phase 5, then Phase 6. Fit in the remaining Phase 1 items
   as convenient.
