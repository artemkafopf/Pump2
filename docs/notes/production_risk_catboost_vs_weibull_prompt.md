# CatBoost vs Weibull Failure-Rate Comparison — Handoff Prompt (v2)

You are working in `D:\GitHub\Pump2` (branch `data_storage_and_processing`) on the
production-risk failure-rate workflow.

Read `docs/notes/production_risk_failure_rate_handoff_prompt.md` and
`docs/notes/production_risk_failure_rate_refit_prompt.md` first for how the failure-rate deliverable
is built today. Do **not** change the Weibull path, the denominators, the display window, or the
chart/materiality logic — you are **adding a parallel model line**, not replacing anything.

## 1. Primary goal (this is the whole point)

**Decide whether the CatBoost ML model matches the actual monthly ESP failure-rate data better than
the Weibull survival model.** The deliverable is the **failure-rate comparison graph** — the factual
line vs Weibull vs CatBoost, per УН and fleet-global — plus the quantitative verdict (which model
tracks the fact line more closely, where, by how much). Everything else serves that comparison.

The CatBoost line **must be hazard-inclusive**, on equal footing with the Weibull *survival* model:
it already uses the full verified hazard-covariate feature set (freq / load / GLF / restarts /
completion / cohort), so it is the covariate/hazard-driven ML counterpart to the Weibull stress
model. Overlay it against **both** Weibull lines that ship today — the base stratum line and the
hazard/stress (θ) line — so the covariate-aware models are judged head-to-head against fact.

**Scope: the fact window only (2024-01..`fact_through`).** The verdict lives entirely where fact
exists. There is **no CatBoost forecast line**: the Weibull forward line is a *renewal simulation*
(`project_well`: failed mass re-enters at age 0 after downtime), a non-renewal CatBoost overlay
there would be apples-to-oranges, and building a proper renewal simulation on the ML survival curve
is real work that serves nothing without a fact line to judge it against. CatBoost columns are NaN
from `forecast_start` on; extend forward only if the user later asks.

## 2. Priorities (in order)

0. **Prerequisite — the OOS holdout verdict must exist before the comparison is built** (§4 step 0).
1. **The comparison + verdict.** Build the CatBoost failure-rate line and overlay Fact /
   Weibull-base / Weibull-hazard / CatBoost, then report which matches the fact data best. A
   lightweight rendering is fine: one comparison worksheet (native openpyxl chart like today's)
   and/or standalone figures under `results/`. Get this right and complete — it is the goal.
2. **Diagnostics.** Per-УН fact-vs-model ratios and monthly-rate MAE for all model lines; CatBoost
   covariate-coverage share; past-b90 tail-dependence share; the honest-null framing.
3. **Excel / VBA / EXE productionisation — LOWEST priority, deferred.** Do **not** wire CatBoost into
   the launcher, VBA, or the frozen EXE unless the comparison shows CatBoost is clearly worth shipping
   *and* the user then asks. CatBoost is a heavy PyInstaller dependency — keep it out of the deployed
   tool; the comparison is produced from the Python/CLI path alone.

## 3. What already exists (reuse, do not rebuild)

**Weibull side (the thing being paralleled):**

- `backend/analysis/workflows/production_risk/failure_rate.py`
  - `_hist_predicted_failures_by_field(...)` replays each observed pump interval (Big/Свод) and, per
    month, calls `_append_interval_predictions`, which calls
    `survival.current_pump_p_fail(age_pmf, params, op_days_month, model)`.
  - `current_pump_p_fail` returns the conditional monthly failure probability
    `clip(1 − S(age+Δ)/S(age), 0, 1)` where `S` is the Weibull stratum survival and `Δ = op_days` that
    month. This is the single hook you parallel.
- Stratum resolution: `crosswalk.map_model_field_from_well` + `StrataModel.resolve(field, sour, ctr)`.
- **Two Weibull failure-rate lines already ship**, both are comparison targets: the **base** line
  (`Прогноз_ремонтов.xlsx`, base stratum scenario) and the **hazard/stress** line
  (`Прогноз_ремонтов_hazard.xlsx`, the θ Cox-stress overlay via `survival.HazardLayer` /
  `scenario_params`, driven by the same run covariates). `failure_rate.compute(..., scenario_id=...)`
  selects which. The hazard line is the covariate-aware Weibull; that is the one most directly
  comparable to CatBoost, but overlay both. (Note the project's own finding: θ adds ~nothing
  out-of-sample, so base ≈ hazard — part of why an ML cross-check is interesting.)

**CatBoost side (the model to plug in), already implemented:**

- `backend/analysis/models/ml/regression_ttf.py`
  - `build_regression_frame(tte_col="ttf_mix", include_equipment=...)` → full-population frame keyed by
    `row_id`, `well_key`, `install_date`, `stratum_key`, with the verified feature set + `*_missing`
    flags (NaN-restored for CatBoost), `run_days`, `tte`, `event`.
  - `TTFRegressor.fit(train_df)` / `.predict_ttf(df) → [b10_op_d, b50_op_d, b90_op_d, mean_ttf_op_d]`
    (IPCW-weighted quantile CatBoosts; quantiles row-sorted to be monotone). **All in operating days
    (`ttf_mix` clock)** — the same clock the failure-rate replay ages pumps in.
  - `RULRegressor` (conditional residual life given age; landmark model) — available as a cross-check.
- `scripts/run/catboost_regression_v2.py` — the temporal hold-out harness (C5 cutoffs, stratum-KM
  quantile baseline, well-clustered CIs). See §4 step 0: its outputs do not currently exist in the
  repo.

## 4. Step 0 — establish the OOS verdict (prerequisite)

**Check `results/catboost_regression_v2/` for the temporal hold-out outputs. As of writing the repo
has none** — the harness script exists but has not left results. If absent, run it once and record
the OOS verdict (hold-out metrics vs the stratum-KM baseline, both cutoffs) before building the
comparison. That verdict is the citation anchor for everything below: **cite it, do not restage it
on the failure-rate line.**

**Honest-null context you must respect** (from `agents/analyses/catboost_survival_v2.md` and the
model report): every out-of-sample gate so far (linear θ in C5, Phase-D landmark dynamics) found the
covariates **do not transport**; the design expectation is that CatBoost's C5 hold-out is essentially
tied with the stratum-KM baseline — confirm the actual numbers with the step-0 run. So the expected
outcome of this comparison is that CatBoost tracks the failure rate **about as well as, or worse
than, Weibull, not dramatically better.** Do not tune, feature-hack, or leak to make CatBoost "win."
A null/triangulation result is the expected, publishable outcome.

## 5. The bridge — CatBoost TTF → monthly p_fail (parallel to `current_pump_p_fail`)

Build one new function that mirrors `current_pump_p_fail` exactly, differing **only** in where the
survival curve comes from:

1. **One inference per run, not per month.** For a run's feature row, call `predict_ttf` once to get
   `(b10, b50, b90)` op-day TTF quantiles. These anchor a per-run CDF: `F(b10)=0.10`, `F(b50)=0.50`,
   `F(b90)=0.90`. Add `F(0)=0`. Build a **monotone** survival `S_cb(t) = 1 − F(t)` by interpolating
   the quantile points (monotone/PCHIP or linear on the sorted anchors), with explicit tails:
   - below `b10`: linear from `(0, 0)` to `(b10, 0.10)`;
   - above `b90`: an exponential tail whose rate matches the implied hazard of the `b50→b90` segment
     (hazard continuity at the joint); document it.
2. **Anchor guards.** `predict_ttf` returns raw regression outputs that are only row-sorted — they
   can be **≤ 0 or tied**. Clip anchors to a small positive floor and enforce strictly increasing
   values with an epsilon separation before building `F`; a degenerate row (b10 == b50 == b90) must
   yield a valid steep-but-finite CDF, not a division by zero or a step function. Guard monotonicity
   and clip `S_cb ∈ (ε, 1]`.
3. **Tail-dependence diagnostic.** Do **not** dismiss the >b90 tail as a corner case: long-lived
   survivors accumulate exactly at ages past their predicted b90, and the historical replay evaluates
   them every month. Report the **share of pump-month evaluations at ages > b90** (per УН and
   global) — if it is material, the CatBoost line is partly tail-choice-driven and the verdict must
   say so.
4. `catboost_p_fail(surv_curve, age_op_d, op_days) -> float` = `clip(1 − S_cb(age+Δ)/S_cb(age), 0, 1)`
   — **byte-for-byte the same conditional formula** as `current_pump_p_fail`, so the only difference
   between the two model lines is the survival curve, not the downstream math. This is what makes it
   a fair apples-to-apples comparison. It is also cheap: analytic evaluation at each month's age off
   a single cached per-run curve (do **not** call CatBoost once per pump-month).

(Secondary cross-check, optional: the `RULRegressor` gives residual-life quantiles conditional on age
`L`; `P(fail this month) = F_res(Δ)` at `age_op_d=L`. Report whether it agrees with the TTF-survival
bridge; keep the TTF-survival bridge as primary because it reuses the identical conditional math and
needs one inference per run.)

## 6. Covariate join (the hard part)

The failure-rate replay iterates Big/Свод intervals keyed by well + install date. CatBoost needs each
run's feature vector. Join fleet intervals to `build_regression_frame` rows by `well_key` + nearest
`install_date` (±7 days, reuse the `production_risk/passport.py` / `join_equipment_block` matching
pattern; nearest-wins, deterministic tie-break). For each interval:

- **matched** → use its full covariate row → `predict_ttf` → per-run `S_cb`.
- **unmatched** (no covariate row: some history gaps) → **still predict with CatBoost.** CatBoost
  handles NaN natively, so score the pump from its known categoricals (`field`, `contractor`,
  `h2s_class`, `pump_gabarit`, `install_period`) with the operational numerics left NaN (plus their
  `*_missing` flags). This keeps the comparison line **100% ML — no Weibull anywhere in it.** Flag
  these as `covariate_partial` and report the share of predicted failures scored on full vs
  categorical-only features, per УН and global. A CatBoost line built mostly on categorical-only
  inputs is weakly informed — surface that, but it is still a pure-ML prediction, not a Weibull
  fallback.

Do **not** fall back to Weibull for any pump — that would contaminate the ML line and defeat the
comparison. The two lines must be independent in their *survival curves*: Weibull uses stratum
survival; CatBoost uses its gradient-boosted TTF quantiles, and nothing else. (What the lines
**share** is exposure plumbing — see §7.)

Note the operational features (freq/load early-window stats) only exist for runs that actually
operated, so CatBoost is best-informed on runs with telemetry; on gap runs it leans on
categorical-only inputs. State this as a known limitation, don't paper over it.

## 7. Fairness / parity rules (get these right or the comparison is meaningless)

- **Capacity asymmetry — the headline line must be cross-fitted.** "In-sample for both" is **not**
  parity: a depth-5 boosted ensemble over 2,634 runs can partially memorize per-run failure times
  (its `S_cb` then drops right at each observed failure, spiking the predicted rate in exactly the
  right month), while a ~7-parameter stratum Weibull cannot. An all-data CatBoost line that "wins"
  the history MAE is therefore uninformative — the exact misleading result the honest-null section
  warns about. Produce **two** CatBoost lines:
  - (a) the **all-data in-sample** line — parity with how the shipped Weibull bundle is fit;
  - (b) a **cross-fitted** line — K-fold (K≈5), **well-clustered** (group folds by `well_key`), each
    history run predicted by the model that never saw it (nor any run of its well).
  **Lead the verdict with (b)**; report (a) alongside it. This is cheap (K refits) and is what makes
  the headline MAE meaningful. The step-0 temporal hold-out remains the OOS *transport* verdict;
  cite it, don't restage it here.
- Same clock (`ttf_mix` op-days), same fleet denominator, same months, same УН grouping, same
  fact-blanking, same materiality — reuse the existing `failure_rate.compute` plumbing; only the
  prediction source changes.
- **Exposure/aging plumbing is shared data plumbing, not model contamination.** The replay's month
  slicing, plan op-days scaling, and the registry `uptime_factor` used to age runs with no observed
  total op-days (see `_append_interval_predictions`) must stay **byte-identical** for both lines —
  that is what "same clock, same denominators" means. Yes, that `uptime_factor` ships in the Weibull
  registry CSV; it parameterises *exposure*, not the survival curve, so using it does not violate the
  "no Weibull in the ML line" rule. Say so explicitly in the write-up; do not rebuild exposure
  differently for CatBoost, or the lines stop being comparable.
- Do not change the Weibull numbers. After your change, `predicted_rate` (Weibull) must be identical
  to today; you are adding `predicted_rate_catboost` (+ `_xfit`) next to it.

## 8. Integration (Python/CLI path only — no VBA/EXE)

1. New module `backend/analysis/workflows/production_risk/failure_rate_catboost.py`: a
   `CatBoostFailureModel` that (a) trains/loads the TTF model(s) once (lazy `import catboost`; the
   all-data fit plus the K cross-fit folds), (b) joins covariates, (c) exposes a per-interval
   `p_fail_fn(age_pmf, op_days) -> float` backed by the cached per-run `S_cb` (pure ML, no Weibull
   fallback). Heavy logic in `backend/analysis/`, per repo rules.
2. **Parity by construction, not by parallel code.** Refactor `_append_interval_predictions` (and
   its two call sites in `_hist_predicted_failures_by_field`) to accept an injected
   `p_fail_fn(age_pmf, op_days) -> float` whose default is the current `current_pump_p_fail`
   closure — the Weibull path's behaviour and output stay byte-identical. The CatBoost line then
   reuses the **identical** month-slicing/aging/exposure code, so "only the survival curve differs"
   holds by construction instead of by promise, and the two implementations cannot drift.
3. In `failure_rate.compute`, carry the CatBoost history predictions through
   `FailureRateResult.monthly` as extra columns (`predicted_failures_catboost` /
   `predicted_rate_catboost`, plus `_xfit` variants for the cross-fitted line; NaN from
   `forecast_start` on) without disturbing the existing Weibull columns. Add
   `catboost_covariate_share` and the past-b90 tail share to `coverage`. Gate behind
   `RunConfig.enable_catboost_compare` / CLI `--catboost-compare` (default off), so the shipped
   Weibull deliverable is byte-identical unless the flag is set.
4. **Rendering (lightweight, in the comparison run only).** Overlay on the failure-rate charts:
   `Факт` (blue solid), `Прогноз (Weibull, база)` (red dashed), `Прогноз (Weibull, hazard)`
   (orange dotted), `Прогноз (CatBoost, cross-fit)` (green dash-dot; the in-sample CatBoost line
   goes in the tables, not the headline chart — five lines per panel is unreadable). Reuse the
   existing date-axis / 2024+ clip / materiality machinery unchanged. A standalone matplotlib figure
   set under `results/catboost_vs_weibull_failure_rate/<date>/figures/` is an acceptable primary
   rendering too — the point is the graph, not the workbook.
5. Full-tables export `F_failure_rate_compare_monthly.csv`: field, month, fleet_size,
   observed_failures, weibull_base_pred, weibull_hazard_pred, catboost_pred, catboost_xfit_pred, the
   matching `_rate` columns, and `catboost_covariate_share`. Run both Weibull scenarios (base and
   hazard) so all model lines sit on one table.
6. **VBA / launcher / EXE: not now.** The comparison never needs the frozen tool; do not touch
   `mdlProductionRiskLauncher.bas`, the launcher xlsm, or the PyInstaller spec. `catboost` must never
   enter the frozen production import chain.

## 9. Comparison outputs — THE deliverable (primary priority)

The verdict, over the fact window (2024-01..`fact_through`), per УН and fleet-global:

- **the graph**: Fact vs Weibull-base vs Weibull-hazard vs CatBoost (cross-fit) monthly failure
  rate, per major УН and global (same materiality as today);
- fact-vs-model ratio for each model line (reuse the existing ratio-table code path);
- **which matches fact best**: monthly-rate MAE (and/or RMSE) of each model line vs fact per УН and
  pooled — this is the headline number that answers the primary question. **The headline row is the
  cross-fitted CatBoost line vs the Weibull lines**; the all-data in-sample CatBoost MAE is reported
  as a secondary row (it is upper-bounded by memorization and says little on its own). The
  out-of-sample transport verdict is the step-0 hold-out (cite it, don't restage);
- CatBoost covariate-coverage share per УН (so a "good" CatBoost fit built mostly on
  categorical-only inputs is visible as such), and the past-b90 tail-dependence share per УН;
- a short `docs/notes/` write-up with the verdict: does the ML model track the actual failure rates
  better, worse, or the same as Weibull, and does it change the **Мирнинский** story specifically.
  **Pre-register the structural expectation here:** Weibull under-counts Мирнинский
  (fact/pred ≈ 1.29) because ~34% of its failures fall in plan-non-producing months, and the
  exposure plumbing — which skips those months — is *shared* by both lines, so **CatBoost cannot
  place those failures either**; no covariate can fix a month the replay assigns zero exposure. The
  answerable Мирнинский question is narrower: does CatBoost redistribute predicted failures across
  the *producing* months closer to fact? State this up front so a null Мирнинский result is read as
  structural, not as a CatBoost failure. Frame the whole write-up against the honest-null
  expectation (covariates didn't transport OOS, so a null result is fully expected and reportable).

## 10. Files to touch

- New: `backend/analysis/workflows/production_risk/failure_rate_catboost.py`
- `backend/analysis/workflows/production_risk/failure_rate.py` (`_append_interval_predictions` gains
  an injected `p_fail_fn` defaulting to today's Weibull closure; parallel columns + extra chart
  series behind the flag; Weibull outputs byte-identical)
- `backend/analysis/workflows/production_risk/config.py` (`enable_catboost_compare`)
- `scripts/run/production_risk.py` (`--catboost-compare`) — and/or a dedicated
  `scripts/run/catboost_vs_weibull_failure_rate.py` if that keeps the comparison self-contained
- `backend/tests/test_production_risk.py` (+ maybe `test_failure_rate_catboost.py`)
- New comparison write-up in `docs/notes/`
- **Not touched:** `mdlProductionRiskLauncher.bas`, launcher xlsm, `scripts/build_production_risk_exe.ps1`
  / `.spec` — VBA/EXE productionisation is out of scope for this task.

## 11. Acceptance / verification

1. **Step 0 done**: `results/catboost_regression_v2/` outputs exist (run the harness if not) and the
   OOS verdict is recorded in the write-up before the comparison is built.
2. `cd backend && python -m pytest tests/ -v` green (currently ~302). Add tests: the bridge reproduces
   `current_pump_p_fail` behaviour when fed a Weibull-shaped S (monotone, conditional formula); the
   injected-callable refactor with the default callable leaves Weibull outputs byte-identical;
   quantile monotonicity → monotone `S_cb`; degenerate anchors (b10 ≤ 0, b10 == b50 == b90) yield a
   valid CDF, no division by zero; covariate-join match rate computed; categorical-only (unmatched)
   path still yields a CatBoost prediction (no Weibull anywhere); cross-fit fold assignment is
   well-clustered (no `well_key` spans folds); Weibull `predicted_rate` unchanged when the flag is
   off.
3. Run with `--catboost-compare`; confirm the overlay renders Fact / Weibull-base / Weibull-hazard /
   CatBoost-cross-fit (date axis, 2024+ clip, materiality unchanged), CatBoost columns are NaN from
   `forecast_start` on, the Weibull columns are byte-identical to the no-flag run, and the
   covariate-coverage and past-b90 tail shares are reported.
4. **The verdict**: report per-УН fact-vs-model ratios and monthly-rate MAE for all model lines
   (cross-fitted CatBoost as the headline, in-sample as secondary), state which matches fact best
   (esp. Мирнинский, against the pre-registered structural expectation), and answer the primary
   question — does CatBoost match the data better than Weibull? — with the honest-null caveat.
5. Do **not** rebuild or touch the EXE/VBA (deferred; comparison runs from Python/CLI).

## 12. Repo rules

- Paths via `analysis.paths`; results via `results_dir("...")`; heavy logic in `backend/analysis/`,
  scripts thin; `sys.path.insert(0, str(REPO_ROOT / "backend"))` bootstrap.
- No VBA / stratum / Weibull-machinery changes; the Weibull lines must stay byte-identical with the
  flag off. Determinism: fixed CatBoost seeds, `allow_writing_files=False`. Do not add to the known
  pre-existing `analysis.plotting` test-collection errors.
- `import catboost` must be lazy and confined to the comparison path so the frozen production EXE
  never imports it. Productionising CatBoost into Excel/VBA/EXE is explicitly deferred to a later,
  separate task, only if the comparison shows CatBoost is worth shipping.
