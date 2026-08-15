# Prompt: Add a tunable U-shape Kpod (load-mismatch / IOR) hazard dial

You are working in `D:\GitHub\Pump2` on the production-risk workflow. Read this whole file
first, then implement. The design decisions below are already made — do not re-litigate them.

## Goal

Add a **stress-scenario-only**, externally-tunable **U-shape Kpod hazard dial** that
penalizes load mismatch — primarily **high Kpod (IOR / overload)**, with an optional
underload side — mirroring the existing Ql dial. Kpod = Ql / Qnominal. It must be tunable by
editing a bundle CSV **without rebuilding the EXE**, exactly like the Ql dial already is.

Expected honest outcome: this is a physical **sensitivity dial**, not an OOS-validated
forecaster. Do **not** promote it into `base_p50`. Ship it in `stress_p75` only.

## Essential background (already true in the repo)

### The Ql dial is the template — copy its pattern exactly

- `backend/analysis/workflows/production_risk/config.py`
  - Constants `QL_HAZARD_ENABLED/BETA/GAMMA/CAP_LOG_RATIO/GLOBAL_REF_LOG/FIELD_REF_LOG/SOURCE`.
  - `ql_hazard_path()`, `_load_ql_hazard_overrides()` (reads `esp_ql_hazard.csv` from the
    bundle at import and overrides the in-code constants; falls back to defaults if the file
    is missing/invalid). Called once at module import. **Replicate this for Kpod.**
- `backend/analysis/workflows/production_risk/survival.py`
  - `ql_hazard_theta(log_ql, model_field, ages)` returns a per-day hazard multiplier
    `theta = exp(z * (beta + gamma*log(max(age,1))))`, applied daily as
    `q_eff = 1 - (1 - q)^theta`. Threaded through:
    - `project_well(..., log_ql_monthly, model_field)` — forecast projection.
    - `current_pump_p_fail_monthly(...)` — 90d/monthly path.
- `backend/analysis/workflows/production_risk/failure_rate.py`
  - `_append_interval_predictions(...)` history replay: `_ql_inputs(month)` reads
    `plan.liquid_volume.at[code, month] / plan.op_days.at[code, month]` to get monthly Ql,
    `_apply_ql(...)` applies the daily multiplier. Ql is only applied when
    `hazard_mode == "stress"` (`use_ql_hazard`). The stress scenario id is
    `C.STRESS_SCENARIO_ID == "stress_p75"`.
- Externalized bundle file (already shipped):
  `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_ql_hazard.csv`
  (param,value rows). The whole `esp_survival_vba_models` dir is packaged into the EXE via
  `--add-data` in `scripts/build_production_risk_exe.ps1`, so editing the CSV in
  `dist\Pump2ProductionRisk\_internal\results\esp_survival_vba_models\2026-07-15-mc2023plus\`
  changes the frozen EXE with **no rebuild** (verified).

### Kpod == Ql / Qnominal (definitionally)

`backend/analysis/data/run_covariates.py` SQL:
```
kpod_mean            = AVG(qliq / nominal_flow_m3d)                 -- raw Ql/Qnominal
kpod_freq_mean       = AVG(qliq / (nominal_flow_m3d * freq/50))     -- freq-normalized
frac_kpod_below_0p7  = AVG(qliq/nominal_flow_m3d < 0.7)             -- fraction underloaded
```
Fleet distribution of `kpod_freq_mean` in the shipped bundle: mean 0.786, p10 0.385, p90
1.135 (so a `k_hi` of 1.2 rarely triggers on history — see cut-off tuning note).

### Qnominal sources (for computing Kpod)

- Historical/existing wells: `backend/analysis/data/equipment_big.py` maps
  `q_nom_m3d <- ("Насос", "Ном. Произв")` (and `freq_nom_hz`). Also
  `backend/analysis/data/pump_type_parser.py` parses `q_design` from the ЭЦН type string
  (e.g. `5а-800-2000` -> 800).
- **Planned/new wells (DECISION ALREADY MADE): field-typical nominal.** Set
  `Qnom_new = median(historical q_nom_m3d over runs whose model_field == this field)`, then
  `Kpod_month = plan_monthly_Ql / Qnom_new`. Rationale: sizing the pump to the well's own
  rate self-neutralizes the IOR/overload side; a field-typical nominal lets a well the plan
  pushes above typical capacity show Kpod > k_hi and get penalized. Do **not** back-calc
  Qnom from the planned rate.

### Existing reference implementations to mirror (do not wire these in; they are for a
different modeling app, but the transforms are correct)

`backend/analysis/features/stress_term_presets.py`:
- `stress_low_Kpod`: transform `negative_excess`, reference 0.7, `coefficient_non_negative=True`.
- `stress_qliq_vs_nominal`: `relative_negative_excess` vs the nominal column, only used
  `if qliq and nominal and not kpod` (the codebase already treats qliq/nominal as a Kpod
  duplicate).

### Validation infra to reuse (Workstream C)

- `backend/analysis/workflows/production_risk/hazard_refit_c.py` and
  `scripts/run/production_risk_hazard_refit_c.py` build a per-run-month training table
  **from the production replay itself** via the additive `emit_rows` hook in
  `failure_rate._hist_predicted_failures_by_field` (baseline offset `mu_baseline` reconciles
  to the D/E predicted total 707; A-population events ~680). Reuse this to fit/OOS-validate
  the Kpod dial and to run the lagged/lead causal check.

## The U-shape design (implement exactly this)

```
k        = Kpod (Ql / Qnominal) for the month
s_under  = clip(k_lo - k, 0, cap_under)         # below lower cut-off (underload)
s_over   = clip(k - k_hi, 0, cap_over)          # above upper cut-off (overload / IOR)
coef_u   = beta_under + gamma_under*log(max(age,1))   # extended-Cox age interaction
coef_o   = beta_over  + gamma_over *log(max(age,1))
theta    = exp( clip( s_under*coef_u + s_over*coef_o , -8, 8 ) )
q_eff    = 1 - (1 - q_base)^theta               # applied per operating day, stress only
```
- Inside `[k_lo, k_hi]` -> theta = 1 (neutral band). `beta_* >= 0` (hazard-increasing).
- `beta_under = 0` disables the underload side (the primary target is the overload/IOR side).
- Use **raw** Kpod = Ql/Qnom (NOT freq-normalized) for the dial, because planned wells have
  no frequency and IOR-via-overspeed shows up as high raw Kpod. (Document this; the static
  `kpod_freq_mean` covariate is freq-normalized — different quantity.)

## Required work

### K0. Config constants + externalization loader (`config.py`)
Add, mirroring the Ql block/loader:
```
KPOD_HAZARD_ENABLED = True
KPOD_HAZARD_K_LO = 0.7
KPOD_HAZARD_K_HI = 1.2
KPOD_HAZARD_BETA_UNDER = 0.0
KPOD_HAZARD_BETA_OVER  = 0.8
KPOD_HAZARD_GAMMA_UNDER = 0.0
KPOD_HAZARD_GAMMA_OVER  = 0.0
KPOD_HAZARD_CAP_UNDER = 0.7
KPOD_HAZARD_CAP_OVER  = 0.8
KPOD_HAZARD_SOURCE = "kpod_ushape_v1 (stress sensitivity, not OOS-validated)"
```
Add `kpod_hazard_path()` and `_load_kpod_hazard_overrides()` (same shape as the Ql loader:
read `param,value` rows, override the module globals via `global`, warn-and-fallback on
error), and call it at import.

### K1. Externalized bundle file
Create `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_kpod_hazard.csv`:
```
param,value
enabled,TRUE
k_lo,0.7
k_hi,1.2
beta_under,0.0
beta_over,0.8
gamma_under,0.0
gamma_over,0.0
cap_under,0.7
cap_over,0.8
```
Add a one-line note to the bundle `manifest.json`.

### K2. `kpod_hazard_theta(kpod, ages)` in `survival.py`
Implement the U-shape math above; handle scalar and np.ndarray `ages` like `ql_hazard_theta`;
return 1.0 / ones when disabled or `kpod` is non-finite.

### K3. Per-well Qnominal resolution
Add a helper (in `crosswalk.py` or a small new module) `resolve_qnominal(code, model_field)`:
1. existing well -> its installed `q_nom_m3d` from `equipment_big` (fallback: parse from pump
   type via `pump_type_parser`);
2. else (planned/new) -> field-typical median `q_nom_m3d` for that `model_field`
   (precompute a `{model_field: median_qnom}` map from `equipment_big` once).
Expose the field-typical map so the projection can use it. Guard divide-by-zero / missing.

### K4. Wire monthly Kpod into the stress paths
- History replay (`_append_interval_predictions` / a `_apply_kpod` mirroring `_apply_ql`):
  `Kpod_month = monthly_Ql / Qnom(code)`; multiply the daily multiplier by
  `kpod_hazard_theta`. Only when `hazard_mode == "stress"`.
- Forecast projection (`project_well`): pass a monthly Kpod array (from `plan` Ql and the
  resolved Qnom) alongside the existing `log_ql_monthly`; apply `kpod_hazard_theta`
  compounding with the Ql theta. Only in stress.
- Keep base_p50 byte-identical (the term must never touch the baseline scenario).

### K5. Remove the double-count
`kpod_freq_mean` and `frac_kpod_below_0p7` are already in
`results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_cox_coeffs.csv` (applied in stress
via `HazardLayer.apply_theta`). Disable those two rows (set `enabled=FALSE`) so load mismatch
is not counted twice. They are OOS-null (Workstream C: refit p=0.52 / 0.65), so nothing is
lost. This is a bundle-CSV edit (no code change).

### K6. Tests (`backend/tests/`)
- `kpod_hazard_theta`: neutral band -> 1.0; overload (k>k_hi) -> >1; underload with
  beta_under>0 -> >1; caps saturate; age interaction behaves; disabled/NaN -> 1.0.
- Loader: `esp_kpod_hazard.csv` is read and applied to config (like
  `test_ql_hazard_bundle_override_applied`).
- Regression: base_p50 focus fact/model unchanged (Mc 0.994 / Ya 0.948 / Vt 1.020 /
  Global 0.962); stress moves.

### K7. Validate honestly before trusting (reuse Workstream C)
- Refit the U-shape on the accepted baseline offset (emit training table, add a monthly Kpod
  column), OOS split fit `<=2024-12` / score `2025-01..2026-06`: per-field & global
  fact/model, monthly MAE, Poisson deviance, calibration.
- **Lagged/lead causal check** (critical — low Kpod near failure is likely an EFFECT of a
  dying well): compare `beta` at lag t-3/t-6 vs lead t+3. For Ql the lead t+3 beta (+0.84)
  was ~10x the lag (+0.09) = reverse causation; expect Kpod to be at least as contaminated.
  Report it plainly. If reverse causation dominates, keep the dial stress-only and say so.

### K8. Regenerate + rebuild
- CLI: `python scripts/run/production_risk.py --full-tables` (regenerates the 3 workbooks;
  base unchanged, stress reflects Kpod).
- EXE: `& "D:\GitHub\Pump2\scripts\build_production_risk_exe.ps1"` (invoke by absolute path;
  the preflight checks `BUNDLE_DATE`, retired calibration, `QL_HAZARD_ENABLED`).
  **Gotcha:** close any Excel workbook open from `dist\Pump2ProductionRisk\results\` first —
  an open `~$*.xlsx` lock makes PyInstaller's clean step fail with WinError 32.
- Prove tunability without rebuild: edit `beta_over` in the packaged
  `_internal\...\esp_kpod_hazard.csv`, re-run the frozen EXE, show stress changes, restore.

## Guardrails / constraints

1. Stress-only. `base_p50` must stay byte-identical (verify focus fact/model unchanged).
2. Tunable without rebuild: all Kpod knobs live in `esp_kpod_hazard.csv`, code falls back to
   defaults if absent.
3. Do not overwrite the accepted survival baseline or re-enable manual calibration.
4. Additive only: default the dial so behavior is explainable; document that it is a
   sensitivity what-if, not OOS-validated.
5. Raw Ql/Qnom (not freq-normalized) for forecastability; field-typical Qnom for planned wells.
6. Cut-off tuning note: with k_hi=1.2 few historical wells exceed it (p90 Kpod ~1.13); the
   user may want a lower k_hi to catch more IOR. Leave it tunable and document the
   distribution so cut-offs can be chosen informedly.
7. Add focused tests for every reusable function; `python -m py_compile` + `pytest` the
   touched suites (`tests/test_production_risk.py`, `tests/test_hazard_refit_c.py`, plus new).

## Final response expected

1. where the code + externalized file live;
2. base_p50 unchanged proof + stress delta;
3. causal-check verdict (is high-Kpod predictive or reverse causation?) and OOS numbers;
4. confirmation the frozen EXE reads `esp_kpod_hazard.csv` with no rebuild;
5. tests/commands run.
