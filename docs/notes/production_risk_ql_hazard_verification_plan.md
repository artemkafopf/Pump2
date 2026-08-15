# Production Risk — Ql Hazard Update: Verification Procedure

Date: 2026-07-14
Scope: working-tree changes described in
`docs/notes/production_risk_ql_field_ref_review_handoff.md` (Ql field-ref clip5
hazard layer, validation-chart semantics change, packaging changes).

The update touches three independent contracts, and the procedure is organized
around them:

- **A. Base deliverable invariance** — the primary (validated) scenario must be
  unchanged by the hazard layer.
- **B. Hazard mechanism correctness** — the Ql multiplier must be wired the same
  way everywhere it is applied (forward projection, 90-day horizon, historical
  replay) and match the fitted definition.
- **C. Calibration and chart semantics** — the retired Mc survival-weight and the
  fixed calibration factors must still produce a defensible fact-vs-model chart.

Each tier below lists concrete checks with pass criteria. Tiers 0–2 are cheap and
should gate any commit; Tiers 3–5 gate the shipped package.

---

## Tier 0 — regression gate (existing tests + scenario invariance)

**0.1 Unit suite** (already green on 2026-07-14, 20 passed):

```powershell
cd backend && python -m pytest tests/test_production_risk.py -v
```

**0.2 Base-scenario invariance under the Ql flag.**
Run the workflow twice — once as-is, once with `C.QL_HAZARD_ENABLED = False`
(monkeypatch or temporary edit) — and diff every `base`-scenario table under
`results/production_risk_forecast/<date>/tables/`.

*Pass:* base tables byte-identical; only `stress`/hazard tables differ.
This proves the layer is stress-only, including the historical replay path
(`hazard_mode="baseline"` no-op in `_hist_predicted_failures_by_field`).

**0.3 CatBoost flag off is a no-op.** Run with and without
`--catboost-compare` absent (default) and confirm the three deliverable
workbooks are identical to a run built before the flag existed (or diff
against the flag-on run: only `*_catboost*` columns/diagnostics may appear).

---

## Tier 1 — hazard mechanism invariants (add as unit tests)

**1.1 Theta identities** for `ql_hazard_theta(log_ql, field, age)`:

- `log_ql == field_ref` → θ == 1.0 exactly, all ages.
- Clipping: θ(log_ql = ref + 10) == θ(log_ql = ref + log 5).
- Non-finite `log_ql` → θ == 1.0 (scalar and array paths).
- Age crossover: with β=−0.587058, γ=+0.093473 the coefficient flips sign at
  age = exp(β/−γ) ≈ **533 op-days**. Assert θ < 1 for (high Ql, age 100) and
  θ > 1 for (high Ql, age 1000). This pins the intended sign/age interaction
  so a future refit that changes it breaks a test, not silently the workbook.

**1.2 θ=1 wiring equivalence.** `project_well(..., log_ql_monthly=ref_vector,
model_field=f)` must equal `project_well(...)` without Ql args to ~1e-12.
Catches any wiring bug in the `q_eff = 1−(1−q)^θ` branch independent of the
coefficient values. Same check for `current_pump_p_fail_monthly` vs
`current_pump_p_fail` (tolerance ~1e-6; ages are rounded to ints in both).

**1.3 Mass conservation in `current_pump_p_fail_monthly`.**
At end of horizon: `failed + up.sum() ≈ 1.0` (single-age pmf) within 1e-9.

**1.4 Daily-vs-monthly application drift (bounded, not exact).**
The historical replay applies θ at the monthly-probability level with
month-start age (`_apply_ql`: `1−(1−p_month)^θ(age₀)`); the projection applies
it daily with age-varying θ. Under proportional hazards these agree exactly
only for age-constant θ. Measure the relative difference over a grid
(age 30–1500, Ql ratio 0.2–5, p_month up to 0.3).
*Pass:* max relative difference in monthly failure probability < ~2%.
If larger, the history line and forecast line are not the "same model" and
one side must switch convention.

**1.5 Train/serve covariate equality (highest-value mechanism check).**
The coefficients come from `results/chem_ql_transform_screen/2026-07-14`
fitting `log_mean_qliq_m3d_field_ref_clip5` on **telemetry-derived** rates.
Deployment computes `log1p(liquid_volume_m3_month / op_days_month)` from the
**plan**. Verify:

- fit-time covariate is also `log1p` of an op-day-normalized (not
  calendar-day) m3/d rate, same field refs;
- distribution match: compare quantiles of deployed z (from
  `by_well.planned_ql_m3d`, per field) against fit-time z. *Pass:* median |Δ|
  of the deciles < ~0.15 in log units; larger drift means the β applies to a
  different scale than it was fitted on and the layer needs re-centering.

---

## Tier 2 — historical replay audit

**2.1 Ql-hazard firing coverage (silent-failure exposure).**
`_apply_ql` swallows every lookup failure (`except Exception: return p_base`)
and the plan only has months from the ПП workbook (2026+), while the replay
starts at 2018-01. Instrument the replay (temporary counter or audit rows)
and report, per field and per month: slices where θ was applied vs. slices
where the lookup failed / Ql ≤ 0.

*Expected:* ~0% applied before the first plan month, high % after.
*Decide explicitly:* either (a) accept and label the stress history line
"Ql layer active from <plan-start> only" in the workbook, or (b) feed
historical actual Ql (telemetry/warehouse) into the replay. A mid-line
regime change that is not labeled will read as a model artifact.
*Also pass:* 0 lookup failures **within** plan-covered months for wells in the
plan (a failure there means a `code`↔plan-index join bug that the bare
`except` is hiding).

**2.2 Planned-vs-actual Ql in the fact window.**
For a sample of ~30 wells with 2026-01..06 fact, compare `planned_ql_m3d`
against actual rates (registry/telemetry). *Pass:* no systematic bias > ~20%;
otherwise the stress fact-window line reflects plan optimism, not operations.

**2.3 Fact/model ratio audit after retiring the Mc survival-weight.**
The reporting-field factors were derived as fact/model over 2024-01..2026-04,
but «Мирнинский УН» 0.929 = 46/49.508 was computed **with Mc
survival-weighting active** (commit 3d05324). The weight is now off, which
raises the Mc model line back up (the documented unweighted ratio was ~0.60),
while the 0.929 factor is retained verbatim. Recompute per reporting field on
the new line:

```
ratio_f = fact_f(2024-01..2026-04) / calibrated_model_f(2024-01..2026-04)
```

*Pass:* |ratio − 1| < 0.10 for every field with ≥ 20 fact failures.
*Expected failure:* Мирнинский near ~0.6/0.929 ≈ 0.65 (model ≈ 55% hot).
Resolution is a decision, not a code fix: re-derive the Мирнинский factor on
the unweighted line (and update the "determined ONCE" note with the new
derivation date), or restore the Mc weight. Doing neither ships a chart whose
stated calibration derivation is false.

---

## Tier 3 — full-run output validation

Run: `python scripts/run/production_risk.py --full-tables`

**3.1 Stress/base envelope.** Per well-month,
`stress_expected_failures / base_expected_failures` must stay within the
theoretical θ envelope: static Cox θ range × exp(±cap·|β+γ·ln age|); with
cap = ln 5 and ages ≤ ~2000d that Ql component is ≈ [0.39, 2.6]. Flag anything
outside; it indicates double application or an age-vector bug.

**3.2 θ audit columns.** In `by_well`: distribution of `theta_ql`
(quantiles, by field, by age band); `theta_qw` ≡ 1.0; `theta ==
theta_static × theta_ql` row-wise. Note: `theta_ql` is currently an
**unweighted mean over age-pmf support points** (`np.mean([...for age in
well.age_pmf])` ignores pmf weights) — for imputed multi-point pmfs the
reported θ misstates the effective multiplier even though the projection math
is correct. Verify against a pmf-weighted recomputation; fix or annotate.

**3.3 Line continuity.** On the stress failure-rate line, jump diagnostics at
(a) the first plan month (where the Ql layer switches on — ties to 2.1) and
(b) `forecast_start` (history replay → renewal projection handoff). Compare
the jump size to typical month-to-month noise per field.

**3.4 Fact-window fit does not degrade.** Per-field monthly MAE and mean bias
of model vs fact over 2024-01..2026-06, base line vs stress line. The Ql layer
is shipped as sensitivity, so it may differ — but a stress line that fits the
fact window *worse* than base should be stated as such in the workbook.

**3.5 May/June 2026 fact completeness.** Fact display now extends through
2026-06. Verify these months are genuinely complete in the new frozen source
workbook (`data/inputs/Отказы свод с анализом.xlsx`); the June Верхнетирский
denominator-only wells (14 active in plan, no Big/Свод interval) still dilute
the June rate — run the June rate with and without them as a sensitivity and
record which convention ships.

---

## Tier 4 — statistical honesty of the Ql term

**4.1 OOS support.** Confirm whether `chem_ql_transform_screen/2026-07-14`
evaluated the field-ref clip5 term out-of-sample (time split). If not, the
existing `HAZARD_SHIP_REASON = "physical_sensitivity_not_oos_validated"`
label must demonstrably cover the Ql layer too (workbook text says
"only as sensitivity" — keep it that way).

**4.2 Reverse-causation probe.** β < 0 at young ages means above-reference
liquid rate *reduces* hazard until ~533 op-days — consistent with selection
(healthy wells produce more), not physics. Check empirical hazard by Ql
tercile within age bands ({<180, 180–533, >533}) against the model's partial
effect. If the empirical pattern disagrees in the old-age band (where the
model says harm), the γ interaction is extrapolation and the cap/γ should be
revisited before anyone reads causal meaning into the stress deltas.

---

## Tier 5 — packaging / end-to-end

**5.1 Source precedence.** `resolve_prediction_workbook_path()` now prefers
repo-local `data/inputs/Отказы свод с анализом.xlsx` over the newest
simple-prediction mirror — for **every** caller, not just production-risk.
Enumerate other consumers (grep `resolve_prediction_workbook_path`) and
confirm none silently switched source. Record a staleness policy for the
frozen copy (who refreshes it, when).

**5.2 Build + frozen-run parity.**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_production_risk_exe.ps1 -Name Pump2ProductionRisk_QlFieldRef
```

Preflight must pass (calibration hooks present, dicts non-empty, Ql enabled).
Run the frozen EXE in a clean directory; diff its three workbooks against the
repo-run outputs from the same input workbook. *Pass:* identical values
(timestamps aside).

**5.3 Launcher note consistency.** `vba/mdlProductionRiskLauncher.bas` dropped
the "Mc survival-weight" phrase; `scripts/deploy/build_production_risk_launcher.py`
`MODEL_CALIBRATION_NOTE` **still contains it** while the weight is retired.
Align both with whatever Tier 2.3 decides, and rebuild the launcher.

**5.4 Coverage metadata truthfulness.** `historical_hazard_layer` reports
`"static+Ql(field-ref clip5)"` for the stress scenario unconditionally; make
it reflect `C.QL_HAZARD_ENABLED` (and ideally the Tier 2.1 firing window).

---

## Execution order and gating

| Gate | Tiers | When |
|---|---|---|
| Commit gate | 0, 1 | every change to survival/failure_rate/layers |
| Model-review gate | 2, 3, 4 | before the workbook goes to reviewers |
| Ship gate | 5 | before replacing `dist/Pump2ProductionRisk_QlFieldRef` |

Known open decisions the procedure forces (do not ship without answering):

1. Мирнинский calibration factor stale after Mc survival-weight retirement (2.3).
2. Ql layer's silent partial coverage on the history line (2.1).
3. June denominator-only wells convention (3.5).
4. Launcher note inconsistency (5.3).
