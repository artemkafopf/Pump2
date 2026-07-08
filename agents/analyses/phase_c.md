# Phase C — Operational (Block 2) + Completion (Block 3) Cox, Merged θ, First Holdout

Annotation for the `phase_c_*` analysis family. Task brief:
[phase_c_operational_completion.md](phase_c_operational_completion.md).

## Task / Research question

Fit Blocks 2 (operational) and 3 (completion/design) of the Cox program on the
Phase A/B foundation, merge the surviving covariates of all three blocks into one
θ(t) VBA-handoff candidate, and — for the first time in the stack — validate out of
sample with a temporal holdout. Decision served: which covariates (if any) earn a
place in the forecasting layer, and whether the operational mode can be changed to
extend pump life (answered only as *association under adjustment*, never causation).

## Status

**complete** · last run 2026-07-06 · entry points: `scripts/run/phase_c_*.py`
(order: features → block3 → block2 → mode_forests → joint_theta → holdout).

## Data

| Source | Access | Key columns / rows |
|---|---|---|
| `mart__weibull_input` + `raw__v03_runs` | `analysis.data.run_covariates.build_run_covariates` | 2,634 runs, 1,527 events, ttf_mix clock |
| `proc__daily_merged` (1.25M rows) | early-window SQL in `run_covariates` | freq/load/qliq/rzab/rpump → Block-2 early features |
| `proc__daily_lab`, `lab.sqlite` | `run_covariates` | ion chem, КВЧ (14 % measured) |
| `failed_node` → mode groups | `analysis.data.competing_risks_loader` | hydraulic/electro-thermal/protector event indicators |

## Methods & models

- **C0** — extended `run_covariates.py` with `freq_std_early`, `load_std_early`,
  `freq_above_55hz_pct_early` and the four mandatory adjusters (`install_period`,
  `run_seq`, `days_since_prev_failure`, `has_telemetry`); pure derivations factored to
  `analysis.features.operational` (kpod, restarts) with unit tests.
- **Reusable cascade** — `analysis.models.survival.screening_cox` (univariate →
  Spearman prune → joint stratified Cox + VIF refit → Schoenfeld PH). Ridge fallback
  and zero-variance guards added to `screening_cox` and `cause_specific_cox`.
- **C1/C2** — the cascade on completion / operational candidates; cause-specific
  passes + per-field heterogeneity via `analysis.common.cox_reporting`.
- **EXTENDED γ** — only via `time_interaction_cox` (event-time episode split);
  curve-regression banned.
- **C4** — merged θ; `theta_final_coeffs.csv` (supersedes `chem_final_coeffs.csv`).
- **C5** — `analysis.models.survival.temporal_holdout`: stratum-KM baseline vs θ Cox,
  C-index + IPCW Brier + calibration on the post-cutoff test set.

## Key findings

- **H-GLF closes the last suspect finding:** protective GLF→hydraulic HR 0.85 **survives**
  operating-point adjustment (0.854→0.847, p=1e-4) and is mode-corroborated
  (hydraulic-significant, electro-thermal-null) → a real gas effect, not a proxy.
- **H-FREQ:** frequency *instability* (`n_freq_steps`, ET HR 1.017 p<1e-4) and >55 Hz
  exposure land on electro-thermal — corroborates the Vt 60 Hz prior fleet-wide.
- **H-KPOD:** no hydraulic U-shape; chronic underload (kpod<0.7) hits **electro-thermal**
  (HR 1.53, p<1e-4) — motor overheating, not hydraulic wear.
- **Block 3:** motor power ↑, depth (`vg_m`) ↑, nominal_freq ↓, pbubble ↓;
  `curvature_missing` informative; gabarit ΔAIC=55. `vg_m` PH flag is a scale artifact
  (episode-split γ≈0).
- **Merged θ is all-STANDARD** (γ negligible) → current VBA can consume it unchanged.
- **C5 (headline):** θ adds **~nothing out of sample** (test C-index ≈0.47–0.53, ≈ baseline;
  Brier slightly worse), both cutoffs agree — the covariate layer does not yet transport
  to unseen 2024+ installs.

## Known limitations

- Associations under adjustment only (§0.2); no within-well design.
- Block 2 / merged θ run on the telemetry subpopulation; `has_telemetry` degenerates
  to an imputation flag there (dropped from those models).
- C5 test is young/censored/complete-case (681→213 runs) — the null delta is robust in
  direction but underpowered as an effect estimate.
- §0.1 proxy semantics (`vg_m`, `curvature`, gabarit→OD, tubing) remain open.

## Outputs

```
results/phase_c_features/2026-07-06/           registry.csv, coverage heatmap, spearman
results/phase_c_block3_completion/2026-07-06/  c1_*.csv, forests, C1 notes
results/phase_c_block2_operational/2026-07-06/ c2_h_*.csv, verdicts, forests
results/phase_c_mode_forests/2026-07-06/       c3 consolidated grid + master verdicts
results/phase_c_joint_theta/2026-07-06/        theta_final_coeffs.csv (+ EXTENDED γ)
results/phase_c_holdout/2026-07-06/            c5_holdout_metrics.csv, calibration, bars
results/phase_c_report/2026-07-06/reports/     phase_c_report.md
```

## How to re-run

```bash
python scripts/run/phase_c_features.py && \
python scripts/run/phase_c_block3_completion.py && \
python scripts/run/phase_c_block2_operational.py && \
python scripts/run/phase_c_mode_forests.py && \
python scripts/run/phase_c_joint_theta.py && \
python scripts/run/phase_c_holdout.py
cd backend && python -m pytest tests/ -q
```

## Related analyses

[[phase_a_cox_foundation]] · [[phase_b_competing_risks]] · [[chemistry_cox]] ·
[[vt_60hz]] · [[cox_hr_vba_integration]]
