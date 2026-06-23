# Tools

Utility scripts for the DLS Correlation Tool.

## `organise_data.py`

Moves raw data files into per-event subfolders based on the event code in
filenames (e.g. `26R03SUZ`, `26T01BCN`).

```bash
python tools/organise_data.py --workflow correlation --dry-run
python tools/organise_data.py --workflow correlation
```

## `validate_config.py`

Validates a `Run_*.py` configuration file without generating plots.

```bash
python tools/validate_config.py Run_Correlation.py
```

Runs the configuration in `--dry-run` mode, which also exercises folder-based
run expansion (`folder` + `filetype`, including the optional `contains`
substring filter). Empty folders, unknown filetypes, and `contains` filters
that match no files are reported as errors before any plotting starts.

## `vibrations.py`

Fits a 4-DOF body dynamics transfer function (Heave, Pitch, Roll, Warp)
to measured FPushrod Power Spectral Densities.

**Preferred usage** — edit settings in `Run_Vibrations.py` and run directly:

```bash
python Run_Vibrations.py
```

Standalone CLI is also available:

```bash
python tools/vibrations.py Data/inputs/ride_dil/26R07BCN/26R07BCN_260612_MAC26-03_BOT_P2_R01.txt
python tools/vibrations.py path/to/file.txt --fmin 1.5 --fmax 15 --no-plots
```

### Fit methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `lorentzian_combined` | 2 × (f₀, ζ, A_F, A_R) per subsystem | Shared-pole Lorentzians: each mode shares (f₀, ζ) across front/rear DOFs with independent amplitudes. Fast (~2–4 s per subsystem). Default. |
| `body4dof` | 13 physical params (M, C, K) | Full mass-stiffness-damping model. Produces self-consistent MCK matrices and mode shapes. Slower (~30–60 s). |

### Pipeline

1. **Load** — FPushrod corner forces, high-pass filtered at 2 Hz
2. **Transform** — Corner → body coordinates via `T`: [FL, FR, RL, RR] → [z_F, θ_F, z_R, θ_R]
3. **PSD** — Welch method (configurable `NPERSEG`, 50% overlap, Hanning window)
4. **Normalise** — Each DOF PSD scaled to [0, 1] for shape matching
5. **Fit** — Differential evolution (Sobol init, peak-picked x0, L-BFGS-B polish)
6. **Plot** — Diagnosis, measured vs fitted, full-band PSDs, transfer functions, mode shapes (body + corner coords), multi-run comparison

### Key settings in `Run_Vibrations.py`

| Setting | Default | Description |
|---------|---------|-------------|
| `F_MIN` / `F_MAX` | 2.0 / 15.0 Hz | Fit frequency window. Must exclude wheel-hop (~15+ Hz). |
| `EXPECTED_FREQS` | per-mode (lo, hi) | Soft band preference + DE seed. Optimiser can escape the band when data supports it. `None` to disable. |
| `NPERSEG` | 1024 | Welch segment length. Higher = finer Δf but fewer averages. |
| `DISPLACEMENT_MODE` | `False` | Fit to xDamperPot displacements instead of FPushrod forces. |

### Spectral averaging

Number of Welch segments (50% overlap) ≈ `2 × N_samples / NPERSEG - 1`.
Aim for ≥ 30 segments for a smooth PSD estimate.

| NPERSEG | Δf (Hz) @ 100 Hz | Min signal for 30 segments |
|---------|-------------------|---------------------------|
| 256 | 0.39 | ~39 s |
| 512 | 0.20 | ~77 s |
| 1024 | 0.10 | ~154 s |
| 1500 | 0.07 | ~225 s |
| 2048 | 0.05 | ~307 s |

### Output plots (per run)

| Plot | Filename | Content |
|------|----------|---------|
| Diagnosis | `vibrations_diag_*.png` | Measured vs fitted (amplitude-scaled) + residual SSE |
| Measured vs Fitted | `vibrations_fit_*.png` | 4-DOF overlay with mode frequency markers |
| Full-band PSDs | `vibrations_psd_modes_*.png` | Log-scale PSDs with mode frequency markers |
| Transfer function | `vibrations_transfer_function_*.png` | Normalised fitted |H(jω)|² |
| Mode shapes (body) | `vibrations_mode_shapes_body_*.png` | Bar chart of [z_F, θ_F, z_R, θ_R] participation |
| Mode shapes (corner) | `vibrations_mode_shapes_corner_*.png` | Bar chart of [FL, FR, RL, RR] participation |

### Output plots (comparison)

| Plot | Filename | Content |
|------|----------|---------|
| Fitted comparison | `vibrations_comparison_fit.png` | Normalised fitted PSDs overlaid across runs |
| Measured comparison | `vibrations_comparison_psd.png` | Normalised measured PSDs overlaid across runs |

### Potential future improvements

- **Half-power bandwidth extraction** — Direct ζ estimation from -3 dB points of measured peaks, as a cross-check or replacement for curve-fitting on clean data.
- **Confidence intervals** — Bootstrap or jackknife resampling of Welch segments to quantify uncertainty on (f₀, ζ) estimates.
- **Multi-band Lorentzian count** — Automatic detection of the number of peaks per subsystem (currently fixed at 2 per heave/pitch and roll/warp). Would handle cases where additional suspension modes appear in the fit window.
- **Wheel-hop mode fitting** — Extend the model to include unsprung mass DOFs, enabling analysis above 15 Hz.
- **Coherence-weighted cost** — Use coherence between corner channels to down-weight frequency bins dominated by noise or non-linear excitation.
- **Cross-spectral phase** — Use phase between front/rear or left/right channels to resolve mode shapes directly from measured data, rather than inferring from amplitude ratios.
- **Automatic NPERSEG selection** — Choose segment length based on signal duration to target a specific number of averages (e.g., 50) while maximising frequency resolution.
- **Batch processing** — Process all events in a directory and produce a summary table of modal parameters across the season for trend analysis.
