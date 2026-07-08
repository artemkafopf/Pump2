# Prompt — Enable the Cox hazard overlay as a physical regime-sensitivity layer

## Framing (read once, then build exactly what is asked)

The user has made an **informed decision** to override the out-of-sample null. The
operational+completion θ subset failed the OOS gate (it forecasts *worse* than the
stratum baseline OOS — `results/vba_model_v2_hazards/2026-07-08/README-data.md`,
model report §3.8/§8.5). The user nonetheless wants the hazards **applied to RUL/TTF**
so the numbers respond to operating-regime, well-history and vintage change in the
physically expected direction.

**This is legitimate as a clearly-labeled engineering what-if / sensitivity tool — it
is NOT a validated risk-ranking forecast.** Non-negotiable guardrails:

- The **baseline** columns (`ESP_B50_op_d`, `ESP_RUL_op_d`, …) stay exactly as they
  are — the stratum + K=2 baseline remains the reference forecast.
- Every hazard-adjusted output is a **separate, suffixed column/UDF** (`_sens`) and
  carries a visible "sensitivity — not OOS-validated" provenance (header note, the
  `ESP_Version`/`ESP_CoxCoeffs` sheet, and the card).
- The OOS holdout table stays in the manifest and card unchanged.
- **Loudest caveat — the cohort terms:** `log_run_seq` (+0.495, the single largest
  coefficient) and vintage were the *primary* reasons the merged θ failed to transport
  (§2.7/§3.8). Enabling them makes `ESP_RUL_sens` move the most **and** generalize the
  least (they are cohort-bound: a 2019 install can't become a 2023 one). Ship them
  because the user asked; label them the most emphatically.

Do not relitigate the decision; do not silently overwrite the baseline; do build it.

## Hazard groups to enable ("best parameters we have so far" = in-sample stratified Cox)

Use the in-sample stratified global-β Cox coefficients fit by
`backend/analysis/workflows/esp_survival/hazard_layer.py`. Ship these seven groups:

| hazard_group | covariates (mart columns) | window | expected RUL direction (β sign) |
|---|---|---|---|
| GLF | `log_glf_mean_opdays` | early | higher gas → longer RUL (β<0) |
| load | `load_std_early`, `load_mean` | early | more load swing → shorter RUL (β>0) |
| kpod | `frac_kpod_below_0p7`, `kpod_freq_mean` | early | chronic underload → shorter RUL (frac β>0; note `kpod_freq_mean` has a confounding-by-indication sign — surface it) |
| frequency | `freq_above_55hz_pct_early`, `n_freq_steps_per_100d` | early | more instability / >55Hz → shorter RUL (β>0) |
| curvature | `curvature_deg10m` (see rule below) | t0 | steeper dogleg → shorter RUL (fit the sign; report it) |
| well_history | `log_run_seq`, `log_days_since_prev_failure` | t0/cohort | more prior runs on the well → shorter RUL (log_run_seq β>0, large) |
| vintage | `install_pre2020`, `install_2023plus` (dummies; ref = 2020–2022) | t0/cohort | older installs (≤2019) → shorter RUL (β>0) |

(Motor power / pbubble / nominal_freq remain available via the same mechanism but are
not in this set — keep the group list one constant so they can be added later.)

### Curvature rule (user-specified — replaces `curvature_missing`)

Use the curvature **value**, not the missingness indicator. Units are **degrees per
10 m** (dogleg severity). **When `curvature` is missing, impute 0.0** (a straight /
low-curvature well, assumed < 0.3 °/10m — the benign case):
`curvature_deg10m = pd.to_numeric(df["curvature"], errors="coerce").fillna(0.0)`.
Do **not** use `curvature_filled` (that is the well→pad→stratum cascade, a different
assumption) and do not use `curvature_missing`. Fit the Cox coefficient on
`curvature_deg10m`; report its sign and p (the report found curvature *value*
non-significant, §3.5 — keep that honest in the card, but ship it per the user).

---

## Part 1 — Python

Files: `backend/analysis/workflows/esp_survival/hazard_layer.py`,
`backend/analysis/workflows/esp_survival/vba_bundle.py`,
`scripts/run/hazard_layer.py`, `scripts/run/export_model_csv_for_vba.py`, tests.

1. **Build the covariates.**
   - `curvature_deg10m = curvature.fillna(0.0)` (as above).
   - well_history: `log_run_seq`, `log_days_since_prev_failure` (both already in the
     frame; `log_days_since_prev_failure` is NaN on first runs → treat as reference /
     θ-neutral, never impute a fake pause).
   - vintage: `install_pre2020`, `install_2023plus` (already 0/1 int; ref = 2020–2022).
2. **Restrict + group.** Set `SUBSET`/its group tags to the seven groups above; add a
   `hazard_group` column to `esp_cox_coeffs.csv` plus `ship_reason =
   "physical_sensitivity_not_oos_validated"`.
3. **Enable the multiplier.** `enabled = TRUE`. Keep `hazard_manifest.txt` + the card's
   OOS holdout table intact and truthful; add `ship_mode : multiplier (USER-OVERRIDE,
   not OOS-validated)`.
4. **Direction-sanity table** in the card: each group's coefficient sign vs the
   expected RUL direction; flag mismatches (`kpod_freq_mean`, and whatever curvature
   comes out as) rather than hiding them.
5. **B20/B80 instead of B10/B90.** In `vba_bundle.py` change exported quantile columns
   `b10`/`b90` → `b20`/`b80` (0.20 / 0.80); keep `b50` + `b50_lo/hi`. Regenerate
   `esp_models.csv` and `mdlModelSeed.bas`; update `MODEL_COLUMNS` and
   `test_vba_bundle.py` (schema + bisection now covers b20/b50/b80).
6. **Per-run export** (`esp_run_covariates.csv`): include every shipped covariate as a
   raw column — `curvature_deg10m`, `log_run_seq`, `log_days_since_prev_failure`,
   `install_pre2020`, `install_2023plus`, plus the operational ones — keyed by
   `well_key_norm` + `run_seq`. Vintage dummies are exported directly (do not make VBA
   parse dates). `covariate_available` continues to reflect early-window telemetry;
   t0/cohort covariates (curvature, history, vintage) are available for essentially
   every run, so those groups' deltas apply even to pumps with no telemetry.
7. **Tests:** extend `test_hazard_layer.py` — `enabled==TRUE`; `hazard_group` present
   for all rows and covers the seven groups; θ at the reference profile == 1; a
   synthetic run perturbing one group moves θ_group in the expected direction; missing
   curvature → 0.0 contributes θ_curvature toward the ref (not NaN). Keep green.

Re-run both exports so `esp_models.csv` (B20/B80) and `esp_cox_coeffs.csv`
(enabled=TRUE, grouped, 7 groups) land in the same dated bundle folder, and copy them
beside the current model bundle if the dates differ.

---

## Part 2 — VBA

Files: `vba/mdlHazardLayer.bas`, `vba/mdlPublicFunctions.bas`,
`vba/mdlBatchProcess.bas`, `vba/mdlValidation.bas`, `vba/README.md`. Honor every VBA
rule in `agents/analyses/vba_model_v2_handoff.md` (ChrW; no MsgBox in UDF paths; no
reserved-word variable names like `cStr`; delete-before-reimport; `Option Explicit`;
UTF-8/BOM reads; `PrepWorkbook`/`FreshSheet` guards).

1. **B20/B80 swap.** `ESP_B10`→`ESP_B20` (quantile 0.20), `ESP_B90`→`ESP_B80` (0.80) in
   `mdlPublicFunctions`; registry columns `b20`/`b80`; `RunPredictions` headers
   `ESP_B20_op_d` / `ESP_B80_op_d`; `mdlModelRegistry` field names; `ValidateRegistry`.
   Keep `ESP_B50` central.
2. **Group θ + adjusted survival** in `mdlHazardLayer`:
   - `HazardThetaGroupForKey(wellNorm, runSeq, group)` → θ over ONLY that group's
     covariates (exp(Σ βᵢ(xᵢ−refᵢ)), missing→ref), gated by `enabled`.
   - `HazardAdjustedRUL(field,h2s,ctr,age,wellNorm,runSeq,group)` and
     `HazardAdjustedB50(...)`: registry-resolve the stratum params, rescale η via
     `mdlCoxHR.ApplyCoxEta(eta, beta_shape, theta_group)`, return `LatentRUL` /
     `LatentQuantile(0.5,…)` on the rescaled model. `group=""`/`"all"` = full θ.
3. **Per-hazard delta.** For each group g:
   `ESP_dRUL(...,g) = HazardAdjustedRUL(...,g) − ESP_RUL(baseline)` and
   `ESP_dB50(...,g) = HazardAdjustedB50(...,g) − ESP_B50`. Zero at the reference
   profile. Add matching UDFs.
4. **RunPredictions columns (suffixed; baseline untouched).** After the existing block:
   `ESP_RUL_sens_op_d`, `ESP_B50_sens_op_d` (full-θ), and one Δ column per group —
   `ESP_dRUL_GLF`, `ESP_dRUL_load`, `ESP_dRUL_kpod`, `ESP_dRUL_freq`,
   `ESP_dRUL_curv`, `ESP_dRUL_history`, `ESP_dRUL_vintage` (operating days; + = longer
   than baseline, − = shorter). Keep `ESP_HazardFlags`.
5. **Provenance.** Header note / `ESP_Status` line: the `_sens` and `dRUL_*` columns are
   USER-OVERRIDE, not OOS-validated (and specifically that `history`/`vintage` are
   cohort-bound). `ESP_HazardTheta` now varies.
6. **ValidateRegistry:** θ(reference)==1; `ESP_B50_sens==ESP_B50` at the reference; per
   group a direction-sanity check on a perturbed synthetic run; B20 < B50 < B80
   ordering per stratum; curvature θ==1 when curvature=0 (the imputed-missing case
   equals the reference only if ref≈0 — assert the code path, i.e. missing → the
   group's own reference).
7. **Docs:** `vba/README.md` (new UDFs/columns, B20/B80, the sensitivity + cohort
   caveats) and append to `results/vba_model_v2/2026-07-07/README.md`.

## Delta semantics (be precise)

`delta_RUL_g` is the **one-group-at-a-time marginal**: RUL with only group g's θ
applied minus the baseline RUL. Groups are not additive (θ is multiplicative on the
hazard, nonlinear in RUL) — the per-group deltas are interpretable single-factor
shifts, not a partition. The full-θ `ESP_RUL_sens` is the combined effect. Document
this; never present Σ(group deltas) as the total.

## Done when

Bundle re-exported (B20/B80, enabled=TRUE, 7 grouped groups incl. curvature-value +
well_history + vintage), Python tests green; VBA compiles and `ValidateRegistry`
all-green including θ/ordering/direction checks; `RunPredictions` shows a moving
`ESP_RUL_sens_op_d` and non-zero per-group Δ columns (operational groups only for
telemetry runs; curvature/history/vintage for essentially all runs), baseline columns
unchanged; the "not OOS-validated" + cohort caveats are visible in the workbook and
the card retains the failing holdout table. Update `[[project_vba_model_v2]]`.
