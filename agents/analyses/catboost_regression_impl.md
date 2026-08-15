# Implementation Prompt — CatBoost Direct Regression (Weibull-free TTF/RUL)

**Type:** executable build spec for an implementing agent. **Design rationale:**
`agents/analyses/catboost_survival_v2.md` (read it first — it explains *why* every choice
below; this file says *what to build*). **Do not** reintroduce any Weibull/survival
machinery; this is a pure regression model that is an independent cross-check on the
survival stack.

## Read first

```
agents/analyses/catboost_survival_v2.md                 — the plan (rationale, §0 answer, honest-null)
results/model_report/2026-07-07/README.md               — verified hazards (§2), rejected covariates (§3), C5 gate
backend/analysis/models/survival/temporal_holdout.py    — REUSE: temporal_split, stratum_baseline_surv,
                                                          ipcw_brier, cindex_at, cindex_bootstrap_ci,
                                                          calibration_table, n_at_risk
backend/analysis/data/competing_risks_loader.py         — build_competing_risks_df(tte_col="ttf_mix")
backend/analysis/data/run_covariates.py                 — the registry the frame wraps: `*_missing` flags,
                                                          `_LOG_COLS` transforms, `_impute_with_source`
backend/analysis/features/restart_events.py             — reconcile_restarts(); needs an early-window
                                                          variant (T2 — the whole-run count leaks)
backend/analysis/data/equipment_big.py                  — load_equipment_big() for the exploratory
                                                          equipment block; its `row_id` is its OWN Excel
                                                          index, NOT the warehouse row_id (T1.3)
CLAUDE.md                                               — path rules (results_dir, no hardcoded paths)
```

Environment: `catboost==1.2.8`, `lifelines`, `scikit-learn` available. Tests run from
`backend/` (`cd backend && python -m pytest`). The 4 pre-existing `analysis.plotting`
collection errors are known — do not touch, do not add to them.

---

## §0 — CONFIRM WITH USER BEFORE IMPLEMENTING

| Gate | Default (proceed unless user changes) |
|---|---|
| **Censoring method** | **IPCW-weighted** primary; pseudo-observation (RMST) as the cross-check (T6). |
| **TTF quantiles** | Pinball at τ = 0.1 / 0.5 / 0.9 → B10 / B50 / B90. |
| **RUL landmarks** | operating-ages L ∈ {0, 30, 90, 180, 365, 730} days; one model with age-as-feature (per-landmark models = sensitivity). |
| **Temporal cutoffs** | primary 2023-12-31, secondary 2022-12-31 (both, per C5). |
| **Scope** | full fleet, all strata (not Vt-only). |
| **Equipment block (new, unverified)** | include as an **exploratory** feature block with a with/without ablation in T5 (these six covariates are not in the model-report §2 verified set and need the T1.3 passport match). |
| **Per-mode (competing) regression** | optional (T7); build only if user wants it. |

If any gate is unconfirmed at implementation time, stop and ask — do not silently default
on the censoring method or scope.

## Design decisions inherited (do NOT re-litigate)

- Clock `ttf_mix`; full population (failures + censored); **no** `kip`/uptime selection.
- Features = **verified hazards** at t0 / first-30-op-day windows (list in T2), plus the
  §0-gated exploratory equipment block (clearly labeled as unverified, ablated in T5).
  **Never** feed: whole-run features of any kind (including whole-run restart counts —
  see T2), `_m_mean` mature features, ion/gypsum/salt chemistry, `idle_frac`, КВЧ,
  raw calendar `run_days` as a feature.
- No Weibull, no η/β, no hazard function, no stratum baseline offset. Strata enter **only**
  as categorical features.
- Every output in operating days, labeled `_op_d`; calendar via uptime factor (T4).
- Judged on the **C5 temporal hold-out vs `stratum_baseline_surv`** — in-sample fit is
  diagnostic only.

---

## Tasks

### T1 — Data + IPCW weights (`backend/analysis/models/ml/regression_ttf.py`, new package `ml/`)

1. `build_regression_frame(tte_col="ttf_mix") -> pd.DataFrame`: `build_competing_risks_df`
   joined with the early-window restart features (T2) on `row_id` (join **only** the columns
   you need — the registry already carries its own `n_restarts_*` columns; avoid suffix
   collisions); keep `tte`, `event`, `well_key`, `install_date`, the T2 features, and
   mode-event columns (for T7). Full population, no selection. `temporal_split` coerces its
   date column itself — pass `date_col="install_date"` directly.
2. **Undo the registry's cross-split imputation** for CatBoost: the registry mean-imputes
   `_IMPUTE_COLS` well→pad→stratum over the *full* dataset — statistics that leak across the
   temporal split. For every imputed T2 numeric, restore `NaN` where `<col>_missing == 1`
   (CatBoost handles NaN natively) and keep the `*_missing` flags as features. Exception:
   `curvature_deg10m` stays `fillna(0.0)` (user rule: missing = straight well).
3. **Equipment block (if §0 confirms it)** `join_equipment_block(frame) -> frame`:
   `load_equipment_big()` — its `row_id` is the workbook's own row index, **not** the
   warehouse `row_id`; never join on it. Match on normalized `well_key` + nearest
   `install_date` within a small tolerance (a few days; follow the
   `production_risk/passport.py` matching pattern), `is_esp` rows only. Unmatched runs →
   NaN + an `equipment_matched` flag. Write the match rate to `tables/` and warn below 80% —
   do not hard-fail on it.
4. `ipcw_weights(train, *, dur="tte", ev="event", strata_col="stratum_key", truncate=0.05)
   -> pd.Series`: per-stratum Kaplan–Meier of the **censoring** distribution
   `Ĝ_s(t)` (fit KM with event = 1−`event`); weight each **observed failure** `w_i =
   1/Ĝ_{s(i)}(T_i)`; censored rows → weight 0. **Small-stratum fallback:** if a stratum has
   < 5 censored runs in train, use the global censoring KM for it (mirror the ≥5-event
   fallback in `stratum_baseline_surv`) — 25 strata, several known-degenerate. Cap weights
   at their `1−truncate` empirical quantile (computed over the nonzero weights) to tame
   long-tail blow-up. (You may lift the private `_censoring_km` pattern from
   `temporal_holdout.py`.) Return weights aligned to `train`.

### T2 — Feature set (module constants `VERIFIED_FEATURES`, `EQUIPMENT_FEATURES`, `VERIFIED_CATS`)

**Verified block** (all present in the T1 registry frame), with `*_missing` flags kept as
features:

```
Environment:   is_sour_flagged, log_glf_mean_opdays
Operational:   freq_above_55hz_pct_early, n_freq_steps_per_100d, freq_std_early,
               frac_kpod_below_0p7, kpod_freq_mean, load_std_early, load_mean,
               n_restarts_early30 (see below)
Completion t0: log_motor_power_kw, vg_m, nominal_freq_hz, pbubble_atm, curvature_deg10m
Cohort:        log_run_seq, log_days_since_prev_failure
Categorical:   field, contractor, h2s_class, pump_gabarit, install_period
               (CatBoost cat_features — install_period is a pd.cut categorical, it MUST
               be declared categorical or CatBoost will choke; cast all cats to str)
```

**Restart leakage fix:** the whole-run `n_restarts_reconciled` from
`build_restart_features()` counts restarts over install→stop — longer-lived runs
mechanically accumulate more, so it encodes the outcome (a whole-run feature, banned
above; the verified §2 covariate was the *early-window* restart rate). Add a window
argument to `build_restart_features` and compute `n_restarts_early30` =
`reconcile_restarts` over the **first 30 operating days** only. Same rule as every other
`_early` feature.

`curvature_deg10m` = `curvature.fillna(0.0)` (verified §2 covariate — it was missing from
earlier drafts of this list; user rule: missing = straight well).

**Equipment block** (exploratory, §0-gated, via T1.3 — NOT in the model-report verified
set; report it as such):

```
Equipment:     pump_od_mm, nkt_diam_mm, any_corr_protection (cast bool→int),
               cable_section_mm2, cable_length_m, ped_max_temp_c
```

Note `pump_od_mm` is a deterministic function of `pump_gabarit` (`norm_gabarit` emits
both) — keep the verified `pump_gabarit` in the main block and `pump_od_mm` only here;
if both survive to the importance audit, say so.

Missing numerics → leave NaN (CatBoost handles; after the T1.2 NaN-restore); categoricals
→ fill `"NA"`. Guard: assert every listed feature exists in the built frame, else fail
loudly (schema drift).

### T3 — TTF quantile regressor

`class TTFRegressor` with:
- `fit(train_df)`: three `CatBoostRegressor(loss_function="Quantile:alpha=τ")` for
  τ∈{0.1,0.5,0.9}, `sample_weight = ipcw_weights(train)`, `cat_features=VERIFIED_CATS`,
  depth 4–6, early-stopping on a temporal validation slice (the latest ~15% of train by
  install date — never test), `allow_writing_files=False`, fixed `random_seed`. Pass the
  IPCW weights on the eval `Pool` too — an unweighted early-stopping metric optimises the
  censoring-biased objective you built the weights to avoid. Zero-weight (censored) rows
  may be dropped from the quantile fits; they carry no gradient.
- `predict_ttf(df) -> DataFrame[b10_op_d, b50_op_d, b90_op_d]`: predict all three, then
  **row-wise sort** the three values to remove quantile crossing (B10≤B50≤B90).
- Also fit one `loss_function="RMSE"` (IPCW-weighted) → `mean_ttf_op_d` (secondary).

### T4 — RUL landmark conditional regressor

`class RULRegressor`:
- `make_landmark_frame(train_df, landmarks) -> DataFrame`: for each `L`, keep runs with
  `tte > L`; row = features + `age_op_d = L`; **target = `tte − L`**; `event` carried
  (residual is censored iff the run is censored). **Leakage guard:** a run contributes a
  landmark row only if it was genuinely alive at `L` (`tte > L`); no future info in features
  (they are t0/early, so this holds by construction — assert no feature depends on `tte`).
- `fit`: pinball τ∈{.1,.5,.9} on the pooled landmark frame, IPCW-weighted with the
  **conditional** censoring weight — do NOT fit a separate KM per stratum×landmark cell
  (25 strata × 6 landmarks = unfittable cells). Reuse the single per-stratum `Ĝ_s` from T1
  and weight a failure at `T_i` in landmark set `L` by `w = Ĝ_s(L) / Ĝ_s(T_i)`
  (probability of remaining uncensored beyond `T_i` *given* uncensored at `L`); censored
  rows → 0; same truncation. `age_op_d` as a feature.
- `predict_rul(df, age_op_d) -> DataFrame[rul_b10_op_d, rul_b50_op_d, rul_b90_op_d]`, sorted.
- `to_calendar(rul_op, stratum) -> rul_cal`: divide by `uptime_factor[stratum]`
  (`= mean(tte/run_days)` per stratum from the mart; compute once, expose as a table). Emit
  both `_op_d` and `_cal_d` columns; never mix.

### T5 — Validation harness (the verdict) `scripts/run/catboost_regression_v2.py`

For each cutoff (2023-12-31, 2022-12-31):
- `temporal_split(frame, cutoff, date_col="install_date")`.
- Fit T3/T4 on train; predict on test.
- **Baseline to beat — give it the same outputs as the model**, so every metric is
  symmetric: per-stratum train-KM quantiles `B10/B50/B90` (KM curve crossing 0.9/0.5/0.1;
  global-KM fallback for small strata, same ≥5-event rule) for TTF, and the conditional
  train-KM residual-life quantiles given survival to `L` for RUL. Keep
  `stratum_baseline_surv(train, test, horizon)` only for the per-horizon C-index
  comparison.
- **TTF metrics** on test:
  - IPCW-weighted **MAE / pinball** on observed failures (censoring-aware), model vs
    baseline quantiles;
  - **concordance**: `lifelines concordance_index(T, B50, E)` — pass **B50 itself** as the
    score (it expects higher = later event; negating it inverts the C-index). Note the
    model's B50 ranking is horizon-free; the baseline's `S(h)` score varies by horizon —
    report the model row once per cutoff and the baseline per horizon, don't fabricate
    per-horizon model scores;
  - **quantile calibration** — **IPCW-weighted** coverage: weighted fraction of failures
    with `T ∈ [B10, B90]` (target 0.8) and with `T ≤ B50` (target 0.5).
    `calibration_table` is survival-probability-binned and does not apply to quantiles
    directly — write a small quantile-coverage helper instead;
  - well-cluster bootstrap CI on the C-index delta vs baseline (`cindex_bootstrap_ci`
    pattern, clustering by `well_key`).
- **RUL metrics** (don't skip — T4 needs its own verdict): per landmark `L`, over test runs
  alive at `L`, the same IPCW-MAE / pinball / concordance / coverage of the predicted
  residual quantiles vs the conditional-KM baseline; well-clustered CIs (a run contributes
  up to 6 correlated landmark rows — cluster by well, never by row). Report per-`L` and
  pooled.
- **Equipment ablation** (if the §0 equipment gate is on): rerun the TTF holdout with and
  without `EQUIPMENT_FEATURES`; one Δ row per cutoff. This is the entire evidence for
  whether the new equipment covariates carry transportable signal.
- Write `results/catboost_regression_v2/<date>/tables/`: `holdout_metrics.csv`
  (per cutoff × horizon, model vs baseline + Δ + CI), `rul_holdout_metrics.csv` (per
  cutoff × landmark), `calibration.csv`, `equipment_ablation.csv`, `equipment_match_rate.csv`,
  `feature_importance.csv` (CatBoost `PredictionValuesChange` + a few SHAP interaction pairs
  for the audit), `uptime_factors.csv`; and `figures/` (predicted-vs-observed, calibration,
  Δ-C-index forest across cutoffs).

### T6 — Pseudo-observation cross-check

RMST pseudo-values at a horizon τ (e.g. 365 op-d) via jackknife on the KM estimator —
computed **per stratum** (pseudo-values assume covariate-independent censoring; censoring
plainly differs by stratum here, same reason the T1 IPCW is per-stratum; small strata →
global fallback). Fit one RMSE CatBoost on the pseudo-values; confirm its test ranking
agrees with the IPCW quantile model (report the agreement, not a second deploy candidate).

### T7 — (optional, if §0 confirmed) per-mode regression

Repeat T3 with the competing event treated as censoring (cause-specific residual life) for
{hydraulic, electro-thermal, protector}; report per-mode B50 and whether any mode’s ordering
transports where all-cause did not.

### T8 — Report `results/catboost_regression_v2/<date>/reports/README.md`

The story: what it is (independent regression, §0 answer that the old model had no Weibull
params), how TTF/RUL are computed here (quantiles + landmark residual life, no distribution),
the C5 verdict vs baseline with CIs, the honest-null framing (state up front; both outcomes
reportable), feature-importance/interaction audit, and the triangulation conclusion
(does an orthogonal method agree the covariates don't transport?). Every table `_op_d`
stamped; calendar conversions labeled.

---

## Rules

- Paths via `analysis.paths`; reusable logic in `backend/analysis/models/ml/`; scripts thin.
- No VBA changes, no stratum-definition changes, no survival/Weibull machinery, no rejected
  covariates, no whole-run features, no calendar clock as a target.
- Determinism: fixed seeds; `allow_writing_files=False` on every CatBoost.
- `cd backend && python -m pytest tests/ -q` green including new tests.

## Tests (`backend/tests/test_regression_ttf.py`)

- IPCW weights: on a synthetic set with known censoring KM, `w=1/Ĝ(T)` matches by hand;
  censored rows weight 0; a stratum with < 5 censored runs falls back to the global KM.
- Conditional landmark weight: on the same synthetic set, `w = Ĝ(L)/Ĝ(T)` matches by hand
  and equals 1 when no censoring occurs in `(L, T]`.
- Early-window restarts: a synthetic run with restarts on op-days 10 and 200 gets
  `n_restarts_early30 == 1` (whole-run reconciled count would say 2).
- Quantile monotonicity: `predict_ttf` rows satisfy B10≤B50≤B90 after the sort.
- Landmark target: a run failing at `tte=200` contributes residual `200−L` at each `L<200`
  and none at `L≥200`; censored run flagged censored in its residual.
- Leakage guard: assert no feature column correlates perfectly with `tte` (catch accidental
  outcome leakage).
- `to_calendar`: RUL_cal = RUL_op / uptime_factor, both columns present, no mixing.

## Definition of done

- [ ] §0 gates confirmed and recorded in the report header.
- [ ] `regression_ttf.py` builds the full-population `ttf_mix` frame with verified features
      only (schema-asserted) + IPCW weights.
- [ ] TTF B10/B50/B90 (quantile, monotone) and RUL landmark quantiles, both `_op_d` +
      calendar via uptime factor.
- [ ] C5 temporal hold-out (both cutoffs) vs the stratum-KM quantile baseline, with
      well-cluster CIs; IPCW-MAE, concordance, quantile coverage reported — for **both**
      TTF and the per-landmark RUL.
- [ ] Equipment block (if gated on): passport match rate reported; with/without ablation
      row per cutoff.
- [ ] Pseudo-observation cross-check agrees (or the disagreement is explained).
- [ ] Report with the honest-null verdict + triangulation conclusion; no Weibull anywhere.
- [ ] Tests green; suite unbroken (only the 4 known plotting errors remain).
