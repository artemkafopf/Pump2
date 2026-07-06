# Phase A — Cox Foundation: Consistency Fixes + Covariate Registry

## Purpose

Phase A does two things, in this order:

1. **Fixes internal inconsistencies** in the existing survival stack (wrong clock in chemistry Cox,
   missing uncertainty intervals, invalid Extended Cox inference, mixed time axes, label debris)
   so that later blocks build on a consistent foundation.
2. **Builds and audits the full covariate registry** for the upcoming Cox extensions —
   operational parameters (Block 2) and completion parameters (Block 3) — producing a run-level
   feature table and coverage report, **without fitting Block 2/3 models yet** (that is Phase C).

Context documents (read first):

```
results/analysis_review/2026-07-06/README.md      — critical review that motivates every task here
agents/analyses/chemistry_cox.md                  — Block 1 spec (procedure to be reused)
agents/analyses/cox_hr_vba_integration.md         — stratum table, VBA handoff format
CLAUDE.md                                         — path rules (results_dir, no hardcoded paths)
```

---

## Design decisions (fixed — do not re-litigate)

### Stratification

All Cox fitting uses **one joint stratified model**, not separate per-stratum regressions:

```
stratum_key = {field}_{h2s_class}_{contractor_group}     # e.g. Ya_nonsour_brt
```

- Strata enter as `strata=['stratum_key']` — each stratum keeps its own baseline hazard
  (this is what makes the result compatible with the per-stratum K=2 Weibull baselines in VBA).
- β coefficients are **global across strata**.
- Rationale: 1,527 events total, but only 5 strata exceed 100 failures. A per-stratum
  multivariate Cox with 10+ covariates needs ≥100–200 events per stratum (10–20 events per
  variable) — feasible only for Ya_nonsour_brt/slb. Global β + stratum baselines is the only
  design that uses all the data.
- **Heterogeneity is checked, not assumed away**: for every covariate surviving the joint model,
  report per-field β̂ (refit with `covariate × field` interaction, or field-subset fits for
  Ya/Vt/Az/Ic/Za only) in a standard `*_heterogeneity.csv` output. Flag sign flips
  (the Mc GLF flip is the known example).
- `cluster_col='well_key'`, `robust=True` everywhere (62% of runs are repeat runs on the same well).

### Clock

`ttf_mix` is the single duration column for every model in Phase A and beyond.
`run_days` appears only in explicitly-labeled sensitivity comparisons.

### Covariate exposure windows

Every covariate gets one of three window types, recorded in the registry:

| Window | Definition | Use |
|---|---|---|
| `t0` | Known at install (design, completion, pre-install chemistry) | Baseline Cox — no restriction |
| `early` | First 30 **operating** days of the run (or 180d pre-install fallback for lab data) | Baseline Cox — approximation of t0 for fluid properties |
| `whole_run` | Mean over [install, stop] | **Diagnostics and time-varying models only** — never in a baseline Cox that feeds VBA (reverse-causation bias; see review §Act IV) |

---

## Part 1 — Fixes

### T1. Refit chemistry Cox on `ttf_mix` (+ fix GLF aggregation, КВЧ handling)

**Code:** `backend/analysis/data/chemistry_run_features.py`, `backend/analysis/workflows/chemistry_cox/phase_chem.py`

1. Add a `tte_col` parameter to `build_chemistry_df()` (default `"ttf_mix"`; keep `"run_days"` available).
   Pull `ttf_mix` from `mart__weibull_input`.
2. Re-run the full adaptive cascade (univariate → Schoenfeld → correlation → VIF → joint) on
   **both clocks**. Output a side-by-side table: covariate | β_cal | β_mix | HR_cal | HR_mix | p_cal | p_mix.
3. **GLF fix.** `_load_glf_from_daily` currently averages `gas_factor` over days
   `WHERE gas_factor > 0` — a selection bias (a well gassy 10% of days gets the mean of gassy
   days only). Replace with two covariates:
   - `glf_mean_opdays` — mean over **all** operating days (qliq > 0), zeros included;
   - `glf_frac_days_gas` — fraction of operating days with gas_factor > 0.
   Screen both. Re-examine whether the "protective GLF" (HR=0.85) survives; report per-field β̂
   (Mc flips sign — the known heterogeneity case).
4. **КВЧ (`mechanical_impurities_mg_l`) handling.** The well → pad → stratum mean-imputation
   cascade in `_impute()` is correct and stays. What changes:
   - Include `mechanical_impurities_mg_l_missing` (already computed by the builder, currently
     unused) as a covariate in the joint model — if it is significant, КВЧ missingness is
     informative (wells get sampled *because* solids are suspected → MNAR) and the imputed
     coefficient is biased; report this explicitly.
   - Run a **complete-case sensitivity fit** (runs with real КВЧ measurements only, n≈1,091) and
     report β vs the imputed fit. Expect the complete-case β to be larger in magnitude
     (mean imputation attenuates toward zero).
   - Report the imputation-source share (measured / well / pad / stratum) per stratum in a table.
5. **Early-window sensitivity.** Recompute the chemistry covariates on the `early` window
   (in-run samples from first 30 operating days, else 180d pre-install fallback) and refit the
   final model. Report β_early vs β_whole_run for the surviving covariates. If GLF or watercut
   changes materially, the whole-run version is contaminated by reverse causation — say so.
6. Regenerate `chem_final_coeffs.csv` from the **ttf_mix, early-window** fit, with a `clock` and
   `window` column stamped in the file so the VBA side can verify compatibility.

**Output:** `results_dir("phase_a_chem_ttfmix")` — tables listed above + updated forest plot.

### T2. Bootstrap CIs on every operational B50 table

**Code:** reuse `phase3_diagnostics.py` machinery (raw-data resampling + EM refit).

- For each stratum in the field-level (9) and contractor-level (17) TTF-mix Phase 2 tables:
  cluster bootstrap (resample **wells**, not runs), ≥200 resamples if runtime allows (fall back
  to 100 with a note), refit EM, collect B10/B50/w₁.
- Add `b50_lo`, `b50_hi`, `b10_lo`, `b10_hi`, `w1_lo`, `w1_hi` (percentile 95%) columns to the
  parameter CSVs and regenerate the "Recommended B50" table in the contractor report with
  intervals instead of the Low/Medium/High adjectives.
- Degenerate resamples: count them; if >30% of resamples for a stratum are degenerate, mark the
  stratum interval as unreliable rather than reporting a fake-tight CI.

**Output:** `results_dir("phase_a_b50_ci")`.

### T3. ΔAIC replaces RMSE as the K=1 vs K=2 criterion

The EM already computes the censored-data log-likelihood (`_compute_nll`). For every stratum:

- `aic_k1 = 2·2 + 2·nll_k1`, `aic_k2 = 2·k + 2·nll_k2` (k = 5 independent / 3 two-stage);
- add `nll_k1, nll_k2, aic_k1, aic_k2, delta_aic` columns to `phase2_mixture_params.csv`;
- the "mixture wins" statement in reports must cite ΔAIC; RMSE-to-KM stays as a plot
  annotation only (it is not a proper score under censoring).

### T4. Mixed-clock sensitivity (`ttf_true_source = 'missing'`)

510 runs (19%) have no operating-time measurement; their `ttf_mix` **is** calendar time.

- Re-run field-level Phase 1 + Phase 2 on ttf_mix **excluding** those 510 runs.
- Table: stratum | B50_full | B50_excl | Δ%. If any stratum moves >15%, flag it in the ESP
  survival index README.
- Add a `pct_mixed_clock` column (share of runs with `ttf_true_source='missing'`) to the stratum
  inventory tables so every future report shows it.

**Output:** `results_dir("phase_a_ttf_sensitivity")`.

### T5. Label hygiene (pipeline-level, one-time)

**Code:** the pipeline step that builds `raw__v03_*` / `mart__weibull_input` (`scripts/pipeline.py` chain).

1. `TRIM(failed_node)` on ingest ('НКТ ' with trailing space is currently a separate category).
2. Field-name normalization: the mart contains raw Cyrillic names alongside coded twins
   ('Ичёдинское нефтяное месторождение' n=7 vs 'Ic'; 'ЯНГКМ' n=1; 'АЗЛУ' n=1; 'Гораздинское' n=1).
   Build an explicit mapping table `field_alias → field_code`; anything unmapped goes to a
   `needs_review.csv` for the user — **do not guess silently**.
3. `h2s_class`: `_derive_h2s_class()` hard-codes sour classification to field == 'Vt'. Apply the
   10 mg/l threshold to **all** fields and report how many non-Vt runs would flip to sour.
   Keep Vt-only as the *active* behavior until the user reviews that count (stratum definitions
   feed VBA — don't change them unilaterally).

### T6. Extended Cox inference fix (Vt H₂S γ)

The OLS-on-curve-points p-values and R² in `results/vt_failure/2026-06-29/` are invalid
(autocorrelated points; arbitrary n on the grid variant). Fix:

1. **Primary:** episode-split the 366 Vt runs at, e.g., 30-day intervals and fit the
   time-interaction Cox `β·X + γ·X·log(t)` as a proper partial likelihood
   (lifelines `CoxTimeVaryingFitter`, or `CoxPHFitter` on split data with the interaction term).
   Report γ, SE, p from this model.
2. **Cross-check:** cluster bootstrap (resample wells, 500 draws) → percentile CI for the
   curve-regression γ of both M4 (KM-based) and M5 (mixture-based). The M4 vs M5 gap
   (+0.079 vs +0.343) should fall inside the honest interval — verify.
3. Update `results/vt_failure/2026-06-29/README.md`: keep the γ point estimates and all figures,
   but mark the old p/R² values as descriptive-only, and add the new subject-level γ ± CI as the
   citable number.

**Output:** `results_dir("phase_a_extended_cox_fix")`.

---

## Part 2 — Covariate registry and feature builder (T7)

### Goal

One run-level feature table with **every candidate covariate for the three Cox blocks**,
plus a coverage audit — so Phase C (Blocks 2+3 fitting) starts from data, not data-wrangling.
**No Block 2/3 model fitting in Phase A.**

**Code:** new `backend/analysis/data/run_covariates.py` with
`build_run_covariates(tte_col="ttf_mix") -> pd.DataFrame` (one row per run, 2,634 rows),
following the pattern of `chemistry_run_features.py` (well→pad→stratum imputation with
`*_missing` indicators; log1p transforms for right-skewed columns; `stratum_key` attached).
Materialize to the warehouse as `feat__run_covariates` via the pipeline if convenient, but a
builder function + parquet under `results_dir` is acceptable for Phase A.

### Registry — Block 1: environment / fluid (barely controllable)

Verified coverage from `pump2.db`, 2026-07-06. Window `early` = first 30 operating days with
180d pre-install fallback (see Design decisions).

| # | Covariate | Source | Window | Transform | Coverage | Expected effect | Notes |
|---|---|---|---|---|---|---|---|
| 1.1 | chloride_mg_l | proc__daily_lab | early | log1p | ~100% (imputed) | ↑ corrosion | screened 2026-07-03: null on whole_run |
| 1.2 | sulfate_mg_l | proc__daily_lab | early | log1p | ~100% | ↑ scale | ditto |
| 1.3 | calcium_mg_l | proc__daily_lab | early | log1p | ~100% | ↑ gypsum | ditto |
| 1.4 | bicarbonate_mg_l | proc__daily_lab | early | log1p | ~100% | ↑ carbonate scale | ditto |
| 1.5 | **magnesium_mg_l** | proc__daily_lab | early | log1p | 80% of lab rows | ↑ scale | **never screened — add** |
| 1.6 | **sodium_potassium_mg_l** | proc__daily_lab | early | log1p | 80% of lab rows | salinity proxy | **never screened — add** |
| 1.7 | total_mineralization_g_l | proc__daily_lab | early | log1p | ~100% | ↑ | collinear with ions — VIF decides |
| 1.8 | ph (0–14 filtered) | proc__daily_lab | early | as-is | ~100% | ↓ (acid = corrosion) | keep the BETWEEN 0 AND 14 filter |
| 1.9 | ca_so4_product | derived 1.3×1.2 | early | log1p | ~100% | ↑ gypsum saturation | |
| 1.10 | h2s_proxy_mg_l | mart | t0 | log1p | 73% real (27% missing; 40% of values are pad medians) | ↑↑ | ecological caveat in reports |
| 1.11 | mechanical_impurities_mg_l (КВЧ) | lab.sqlite lab_samples | early | log1p | 41% measured | ↑ abrasion | + missing indicator; see T1.4 |
| 1.12 | watercut_daily | proc__daily_merged.watercut | early | as-is (0–100) | 88% of daily rows | ↑ (corrosion, emulsion) | replaces 83%-coverage lab watercut |
| 1.13 | glf_mean_opdays | proc__daily_merged.gas_factor | early | log1p | ~92% | ? (direction unresolved) | fixed aggregation, see T1.3 |
| 1.14 | glf_frac_days_gas | derived from 1.13 | early | as-is (0–1) | ~92% | ↑ | exposure-frequency companion |

### Registry — Block 2: operational (controllable — "risks we control by changing the operating mode")

| # | Covariate | Source | Window | Transform | Coverage | Expected effect | Notes |
|---|---|---|---|---|---|---|---|
| 2.1 | freq_mean | proc__daily_merged.freq | early | as-is (Hz) | 85% of rows; 1,377 runs ≥30 valid days | ↑ above ~55 Hz | fleet-wide version of the Vt 60 Hz work |
| 2.2 | freq_above_55hz_pct | feat__run_freq_exposure | whole_run→recompute early | as-is | 1,838 runs with ≥1 valid day | ↑ | recompute on early window for baseline Cox |
| 2.3 | freq_below_45hz_pct | same | early | as-is | same | ? | low-freq = low cooling flow |
| 2.4 | freq_std / n_freq_steps | daily freq, day-over-day \|Δf\|>1 Hz per 100d | early | as-is | 85% rows | ↑ (thermal cycling) | new feature |
| 2.5 | load_mean | proc__daily_merged.load | early | as-is (%) | 78% of rows | U-shaped? | test linear + spline |
| 2.6 | load_std | same | early | as-is | 78% | ↑ (instability) | mature-only version exists at 36% |
| 2.7 | **kpod = Ql/Qnominal** (коэффициент подачи) | daily qliq / mart.nominal_flow_m3d | early | as-is | qliq 88% rows × nominal 100% | U-shaped: ↑ off-BEP both sides | THE user-requested Ql/Qnom; also `kpod_freq` variant with Qnom scaled by (f/50) |
| 2.8 | frac_kpod_below_0p7 | derived from 2.7 | early | as-is (0–1) | same | ↑ underload/gas | mature version exists (41%) — recompute early |
| 2.9 | pzab_over_pbubble | daily rzab / mart.pbubble_atm | early | as-is | rzab 80% rows × pbubble 93% | ↓ below 1 (free gas at intake) | plus frac_pzab_below_1 |
| 2.10 | rpump_intake_mean | proc__daily_merged.rpump_intake | early | as-is | 59% of rows | ↓ low intake pressure | coverage-limited; keep with indicator |
| 2.11 | **idle_frac** = 1 − ttf_true/run_days | mart | whole_run (unavoidable) | as-is (0–1) | 81% | ↑ (stop/start cycling) | descriptive first; time-varying in Phase D |
| 2.12 | n_restarts_per_100d | daily qliq==0→>0 transitions | early | log1p | 88% rows | ↑ (restart shock) | new feature |

Block 2 caveat to encode in the builder: operational covariates are *chosen responses* to well
conditions (confounding by indication — bad wells get run harder or softer). Note this in the
registry docstring; causal language is off-limits until Phase C handles it (adjustment for
Block 1 + field interactions at minimum).

### Registry — Block 3: completion / design (fixed at t=0)

| # | Covariate | Source | Window | Transform | Coverage | Expected effect | Notes |
|---|---|---|---|---|---|---|---|
| 3.1 | stages | raw__v03_runs | t0 | as-is | 97% | ? | |
| 3.2 | head_per_stage | nominal_head_50hz_m / stages | t0 | as-is | 97% | ↑ (aggressive stage design) | |
| 3.3 | motor_power_kw | raw__v03_runs | t0 | log1p | 97% | ? (size proxy) | |
| 3.4 | nominal_current_a | raw__v03_runs | t0 | log1p | 97% | collinear w/ 3.3 — VIF | |
| 3.5 | nominal_flow_m3d | raw__v03_runs | t0 | log1p | ~100% | ? (size proxy) | |
| 3.6 | **pump_gabarit / OD group** | parsed from pump_type | t0 | categorical | 100% raw; expect ≥90% parse | ? (user-requested diameter) | see parsing spec below |
| 3.7 | pump_series | parsed from pump_type | t0 | categorical | same | ? | REDA DN/GN/SN vs MT vs Russian ЭЦН/ВНН |
| 3.8 | **curvature** | raw__v03_runs.curvature_flag | t0 | as-is (numeric) | 56% ('-' and NULL = missing) | ↑ dogleg → wear, cable damage | numeric values ~0.32–0.4 observed; **verify units (deg/10 m?) against source workbook before interpreting** |
| 3.9 | vg_m (setting depth?) | raw__v03_runs | t0 | as-is (m) | 97% | ↑ depth → temperature, gas | 0–2,740 m, mean 2,310 — semantics **must be confirmed with the user/source workbook** (submergence_depth_m is 1% and unusable) |
| 3.10 | pbubble_atm | raw__v03_runs | t0 | as-is | 93% | context for 2.9 | reservoir property, but t0-known |
| 3.11 | nominal_freq_hz | raw__v03_runs | t0 | as-is | ~100% | design intent | |
| 3.12 | tubing diameter | — | — | — | **NOT IN WAREHOUSE** | — | user-requested; raise as data request to the data owner |

**pump_type parsing spec** (1,254 distinct strings):
- Russian pattern `^(\d+[аА]?)[- ](\d+)[- ](\d+)$` (e.g. `5а-800-2000`) → gabarit `5а`,
  q_design 800 m³/d, head 2000 m. Gabarit maps to OD group (5 → 92 mm, 5А → 103 mm, 6 → 114 mm —
  confirm mapping table with user).
- Western REDA pattern `^(DN|GN|SN|AN)(\d+)` (e.g. `GN6200`) → series letter → OD group
  (D≈4.00", G≈5.13", S≈5.38" — put the mapping in a reviewable constants dict), number ≈ design
  flow in bbl/d → convert ×0.159 to m³/d for cross-checking against `nominal_flow_m3d`.
- `MT5A-100DP` style → gabarit 5A + q_design 100.
- Anything unparsed → `pump_series='other'` and log the top-30 unparsed strings to
  `tables/pump_type_unparsed.csv` for user review. Target ≥90% parse rate.

### Registry — cohort / history adjustments (not "hazards", but mandatory confounders)

| # | Covariate | Source | Notes |
|---|---|---|---|
| 4.1 | install_year | mart install_date | 2013–2026 span; enter as 3–4 period bins; mandatory in any contractor comparison |
| 4.2 | run_seq | count of prior runs on well_key | 62% of runs are repeats; wells with 5+ runs are a distinct population |
| 4.3 | days_since_prev_failure | previous stop_date → install_date, same well | short gaps = rushed workover? |
| 4.4 | pad | mart | future frailty term; not a fixed covariate in Phase A |

### T7 deliverables

1. `backend/analysis/data/run_covariates.py` — builder producing all rows above with
   `*_missing` indicators and log transforms.
2. `results_dir("phase_a_covariate_registry")`:
   - `tables/registry.csv` — the full table above with **measured** (not estimated) coverage,
     n_measured / n_imputed_well / n_imputed_pad / n_imputed_stratum per covariate;
   - `tables/registry_by_stratum.csv` — coverage per stratum_key (flag covariate×stratum cells
     <50% measured);
   - `tables/pump_type_parse_report.csv` + `pump_type_unparsed.csv`;
   - `tables/spearman_matrix.csv` — correlation across all candidates (pre-screen for the
     Phase C VIF step);
   - `figures/` — coverage heatmap (covariate × stratum), distribution grid (hist per covariate).
3. Two open questions surfaced to the user in the final report, **not silently resolved**:
   - semantics/units of `vg_m` and `curvature_flag`;
   - gabarit → OD mapping table confirmation.

---

## Rules

- All paths via `analysis.paths` (`results_dir`, `WAREHOUSE_DIR`); no hardcoded paths.
- New reusable logic in `backend/analysis/` (models/, data/, workflows/); scripts stay thin.
- `ttf_mix` is the duration column; every output table carries a `clock` column or header note.
- Cluster-robust by `well_key` in every Cox fit.
- Do NOT: modify VBA, start Block 2/3 model fitting, change stratum definitions (T5.3 reports
  only), touch `analysis_outputs/` or other legacy dirs.
- Every table that could reach operations carries intervals, not adjectives.
- Run `cd backend && python -m pytest tests/ -v` before finishing; add tests for the pump_type
  parser (it will be reused).

## Suggested execution order

T5 (hygiene, unblocks clean joins) → T1 (chemistry refit) → T3 (AIC, trivial) →
T4 (sensitivity) → T2 (bootstrap, long-running — start in background early) →
T6 (extended Cox fix) → T7 (registry — the big one, benefits from all fixes).

## Definition of done

- [ ] `chem_final_coeffs.csv` regenerated on ttf_mix + early window, clock/window stamped
- [ ] Calendar-vs-ttf_mix side-by-side β table published
- [ ] GLF re-screened with fixed aggregation; per-field β̂ reported
- [ ] КВЧ: complete-case + missing-indicator sensitivity published
- [ ] B50 tables (field + contractor) carry bootstrap 95% CIs
- [ ] ΔAIC columns in all phase2 parameter CSVs
- [ ] Mixed-clock sensitivity table published; `pct_mixed_clock` in stratum inventories
- [ ] failed_node trimmed; field aliases mapped or queued for review; non-Vt sour count reported
- [ ] Subject-level γ ± CI for Vt H₂S published; old p/R² marked descriptive
- [ ] `run_covariates.py` builds 2,634 × full-registry table; coverage report published
- [ ] Open questions (vg_m, curvature units, gabarit mapping, tubing diameter gap) listed for user
