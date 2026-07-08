# Hazard-layer extension — two agent prompts

Wire the §2 operational + completion hazards (GLF, frequency, chronic underload,
load, curvature-missingness, pump size) into the VBA prediction workbook, on top of
the v2 stratum + K=2 baseline.

**Read the discipline first — this is not optional.**
`results/model_report/2026-07-07/README.md` §2 (accepted hazards, with the mode each
lands on), §3.5 (curvature value rejected; *missingness* borderline informative),
§3.8 (**the merged θ failed the out-of-sample gate — no covariate θ ships as a
forecasting multiplier**), §5.6, §6. The v2 build shelved θ deliberately
(`agents/analyses/vba_model_v2.md` §0.2 deprecate-keep) and left a re-enable switch
(`ESP_CoxCoeffs` + `enabled=TRUE`; η-rescaling in `mdlCoxHR.ApplyCoxEta`). These two
tasks either (i) re-fit the *operational+completion subset* and get it through the
OOS gate so it can ship as a real multiplier, or (ii) ship it as display-only
advisory flags. **The OOS gate decides which — not preference.**

Two hard feasibility facts to design around:
- **Operational covariates (GLF, frequency, underload, load) are EARLY-WINDOW** (first
  30 *operating* days of telemetry). They are unknowable for a brand-new pump and are
  NOT derivable from the `Свод` snapshot columns — they must be exported per run and
  joined into the workbook.
- **Curvature-missingness and motor-power (pump size) are t0 passport** features —
  install-knowable, applicable to every row, and the most useful for new-pump B50.
- Every hazard is **mode-corroborated** (GLF→hydraulic; frequency/underload→electro-
  thermal; load: std→hydraulic, level→ET; size→ET+protector). A principled adjuster is
  mode-aware; a baseline-wide multiplier is the crude version.

---

## PROMPT A — Analysis: curvature & pump-size hazards + a ship-ready operational θ card

### Goal
Produce a **ship-ready hazard coefficient card** and a **per-run covariate export**
for the operational+completion layer, with the **curvature-missingness** and
**pump-size (log motor power)** hazards freshly analyzed and the whole subset put
through the **out-of-sample temporal-holdout gate**. Output a clear verdict:
ship-as-multiplier vs ship-as-advisory-flag.

### Read first
`results/model_report/2026-07-07/README.md` §2.2–2.7, §3.5, §3.8; the covariate
dictionary (§4) with the shelved-θ β values; `agents/analyses/phase_c*.md`. Existing
machinery to reuse (do NOT rebuild): `backend/analysis/data/run_covariates.py`,
`backend/analysis/features/operational.py`, `backend/analysis/models/survival/`
(`screening_cox.py`, `cause_specific_cox.py`, `temporal_holdout.py`), and the
scripts `scripts/run/phase_c_features.py`, `phase_c_block2_operational.py`,
`phase_c_block3_completion.py`, `phase_c_joint_theta.py`, `phase_c_holdout.py`. The
merged fit already exists: `results/phase_c_joint_theta/.../theta_final_coeffs.csv`
(GLF, freq, underload, load, curvature_missing, log_motor_power_kw are all in it).

### Tasks
1. **Curvature hazard (fresh).** Quantify `curvature_missing` ("Работа в кривизне" not
   recorded) as an informative-missingness risk marker: univariate → cascade
   (correlation/VIF/Schoenfeld) → stratified global-β Cox on `ttf_mix` → mode
   attribution → per-field heterogeneity. Confirm curvature *value* stays rejected
   (§3.5, p≈0.26). Report the reference (population) value and answer/flag the open
   question: why is it recorded for only ~56% of wells (MNAR)?
2. **Pump-size hazard (fresh).** `log_motor_power_kw`: same cascade; confirm it lands
   on ET+protector; report the Ic sign-flip heterogeneity (§2.6) and the ρ=0.94
   collinearity with nominal flow (keep one size axis). Reference value.
3. **Operational subset.** Assemble GLF (`log_glf_mean_opdays`), frequency
   (`n_freq_steps_per_100d`, `freq_above_55hz_pct_early`, `freq_std`), chronic
   underload (`frac_kpod_below_0p7`), load (`load_std_early`, load level/quadratic) —
   early window, `ttf_mix`. Reuse the Phase C fits; re-fit only if needed for a clean
   subset joint model.
4. **Joint subset θ.** Fit a stratified global-β Cox with ONLY this operational+
   completion subset (**exclude the cohort-bound terms** — vintage, `log_run_seq`,
   `log_days_since_prev_failure` — those are what broke transport in §3.8). Export per
   term: `beta, hr, ci_lo, ci_hi, p, reference_value, window (t0|early), mode`.
5. **OOS gate (decisive).** Temporal holdout via `temporal_holdout.py`: train on
   installs ≤2023-12-31, score the unseen cohort; repeat at 2022. Compare
   `stratum baseline + subset θ` vs `stratum baseline alone` — C-index and Brier, with
   well-cluster CIs. **Record the verdict:** if Δ C-index > 0 with CI clearing 0 →
   `ship_mode = "multiplier"` and `enabled = TRUE`; else → `ship_mode = "flag"` and
   `enabled = FALSE` (advisory only). Do not fudge this — it is the whole point.
6. **Bundle exports** (extend `vba_bundle.py` or a sibling, under
   `results_dir("esp_survival_vba_models")`):
   - `esp_cox_coeffs.csv` — `covariate, beta, hr, ci_lo, ci_hi, p, reference_value,
     window, mode, enabled, clock`; plus one header row carrying the `ship_mode` and
     the holdout Δ so the workbook can display provenance.
   - `esp_run_covariates.csv` — one row per run with the EARLY-WINDOW + t0 covariates,
     keyed so VBA can join to `Свод`: emit `well_key_norm` (trim+casefold, matching the
     Python `normalize_well_key`) **and** `run_seq` (run number on the well) **and**
     `install_date` — pick the join key VBA can reconstruct from the sheet (well +
     run_seq is the safest; confirm `run_seq` is derivable from the mart run order).
     Include a `covariate_available` flag per row (early-window features are absent for
     runs with <30 telemetry op-days → VBA must fall back to reference → θ neutral).
   - A short card `results/vba_model_v2_hazards/<date>/README.md`: each hazard with β,
     HR, mode, window, heterogeneity caveat, and the OOS verdict in plain language for
     planning staff.
7. **pytest** (`backend/tests/`): schema of both CSVs; `clock=="ttf_mix"`; every
   coefficient has a finite β and reference; `enabled` is consistent with the recorded
   holdout Δ; θ recomputed from (β, ref) at the reference profile == 1 within 1e-9.

### Constraints
- Paths via `analysis.paths`; heavy logic in `backend/analysis/`, thin CLI in
  `scripts/run/`; no hardcoded `D:\Projects\...`.
- One clock (`ttf_mix`), early window = first 30 operating days, t0 = install-knowable;
  stamp `window` on every term.
- Honesty over completeness: if the subset fails the OOS gate, say so and set
  `ship_mode="flag"`. A truthful null is the correct deliverable.

### Done when
Both CSVs + card + pytest exist and are green; the OOS verdict is recorded in
`esp_cox_coeffs.csv` and the card; `[[project_vba_model_v2]]` updated with the
hazard-layer status. Hand the exact `esp_cox_coeffs.csv` / `esp_run_covariates.csv`
schemas to PROMPT B.

---

## PROMPT B — VBA: wire the operational+completion hazard layer into the model

### Goal
Turn the shelved 3-covariate θ stub in `mdlCoxHR` into a **general, sheet-driven
N-covariate hazard layer** fed by PROMPT A's exports, honoring the OOS-gate verdict:
a real η-rescaling multiplier when `enabled=TRUE`, or display-only advisory flags when
not. Do not touch the v2 stratum + K=2 baseline math.

### Read first
`agents/analyses/vba_model_v2.md` (§0.2 deprecate-keep, T3), `vba/mdlCoxHR.bas` (the
re-enable mechanism + `ApplyCoxEta`), `vba/mdlBatchProcess.bas` (`ImportBundle`,
`RunPredictions`, `PrepWorkbook`/`FreshSheet`, `ReadAllLines` UTF-8), `vba/README.md`,
`agents/analyses/vba_model_v2_handoff.md` (the VBA bug/rule list — **honor all of it**:
`ChrW`, no `MsgBox` in UDF paths, no reserved-word variable names like `cStr`,
delete-before-reimport, `Option Explicit`, UTF-8/BOM-safe reads, protected/read-only
guard). The coefficient/covariate schemas come from PROMPT A.

### Tasks
1. **Generalize `mdlCoxHR`.** Replace the 3 hardcoded legacy constants with a
   sheet-driven loader: read `ESP_CoxCoeffs` into a module-private array
   (`covariate, beta, reference, window, mode`) and an `enabled` flag. `ComputeTheta`
   returns `exp(Σ βᵢ·(xᵢ − refᵢ))` over the covariates supplied, clamped as today;
   returns 1 when `enabled=FALSE` or when no covariates resolve (keeps
   deprecate-keep + the θ≡1 validation check intact). Keep `ApplyCoxEta`.
2. **Import the new artifacts** (extend `ImportModelCSV`-style header-driven readers;
   refuse non-`ttf_mix`): `esp_cox_coeffs.csv` → `ESP_CoxCoeffs` (with the `enabled`
   cell); `esp_run_covariates.csv` → a lookup sheet `ESP_RunCov`. Add both to
   `ImportBundle`. Load `ESP_RunCov` into a keyed cache (`well_key_norm` + `run_seq`).
3. **Covariate sourcing per row.** t0 features (motor power, curvature_missing) from
   passport columns if present, else from `ESP_RunCov`; early-window features
   (GLF/frequency/underload/load) from `ESP_RunCov` joined on the same well+run_seq key
   `RunPredictions` already computes for `ESP_RunSeq`. Any covariate missing / run not
   in `ESP_RunCov` / `covariate_available=FALSE` → substitute the reference value so
   that term contributes 1 (θ neutral). Never fabricate.
4. **Outputs, gated by the verdict.**
   - If `enabled=TRUE` (**multiplier**): add back the Cox columns to `RunPredictions`,
     clock-suffixed — `ESP_Theta`, `ESP_B50_Cox_op_d`, `ESP_RUL_Cox_op_d` — computed
     via `ApplyCoxEta`. Keep the plain baseline columns beside them.
   - If `enabled=FALSE` (**advisory flags**): DO NOT alter B50/RUL. Instead emit
     display-only per-mode risk flags (e.g. `ESP_Flag_ET`, `ESP_Flag_Hydraulic`) that
     fire when a run's operational covariates cross documented thresholds
     (underload `frac_kpod_below_0p7` high, frequency instability high, low GLF), each
     labeled as an operating-policy hint, not a survival adjustment.
5. **UDFs.** `ESP_Theta(...)` reads the same sheet/lookup; add `ESP_HazardMode(field,
   h2s, ctr, well, run)` returning the dominant elevated mode for a run (advisory).
   Keep `ESP_B50_Cox` etc. equal to baseline when disabled.
6. **Validation.** Extend `ValidateRegistry`: (a) when disabled, `ESP_B50_Cox ==
   ESP_B50` exactly (existing check); (b) when enabled, θ at the reference profile ==
   1 and `ESP_B50_Cox(reference) == ESP_B50`; (c) `ESP_CoxCoeffs` clock == `ttf_mix`
   and `enabled` matches the manifest verdict; (d) a spot-check run's θ recomputed by
   hand matches the UDF.
7. **Docs.** Update `vba/README.md` and `results/vba_model_v2/2026-07-07/README.md`
   with the hazard-layer section, the enabled/disabled state, and the early-window
   caveat (θ only adjusts pumps with ≥30 telemetry op-days; others stay on the
   baseline). Regenerate `mdlModelSeed`/exports only if their schema changed.

### Constraints & rules
- Everything in `agents/analyses/vba_model_v2_handoff.md` "VBA rules" and "bugs already
  fixed" applies. No `MsgBox` in UDF-reachable code. `Option Explicit`. Delete a module
  before re-importing. No `cStr`-style reserved-word variable names.
- The baseline (stratum + K=2 on `ttf_mix`) is the source of truth; the hazard layer is
  an overlay that must reduce to the baseline exactly at the reference profile and when
  disabled.
- Excel-side compile/validate is iterative with the user (screenshots → fix `.bas` →
  re-import). Close out only when `ValidateRegistry` is all-green with the layer
  imported.

### Done when
The layer imports, compiles, and validates green; θ reduces to baseline at reference /
when disabled; the ship-mode matches PROMPT A's OOS verdict; docs + memory updated.
