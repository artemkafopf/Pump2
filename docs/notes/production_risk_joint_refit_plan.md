# Joint Refit Plan — Hazard-Aware Model That Predicts Survival Curves AND Historical Failure Rates

Date: 2026-07-15
Goal owner: presentation of results — Мирнинский (Mc), Ярактинский (Ya), Верхнетирский (Vt) first.

## Objective

One model bundle such that, for Mc/Ya/Vt, the SAME parameters reproduce:

1. stratum survival curves (censoring-aware KM/IPCW on the full ESP population), and
2. monthly historical failure rates 2024-01..2026-06 (fact/model per reporting field),

**without the manual calibration factors** (`_CALIBRATION_FACTORS` Ya=0.756/Vt=0.837
and the reporting-field factors, incl. stale Мирнинский 0.929). Acceptance targets in §D.

Current state (why they disagree): global fact/model ≈ 0.75 on genuine failures
(Мирнинский 0.60, Ya 0.76, Vt 0.84) — the model over-predicts. Three candidate causes,
each a workstream:

| # | Cause | Evidence | Workstream |
| --- | --- | --- | --- |
| 1 | Weibull fit misses censored mass — curves too short | `weibull_big_censoring_refit_findings.md`: +810 Big-only censored rows → B50 ×1.5–3.1 (Vt_nonsour 150→410). **Corrected 2026-07-15 (see A0):** only ~500 of the 810 survive the proper ESP filter, so shifts are overstated and must be re-measured — direction stands, magnitude TBD | A |
| 2 | op-day↔calendar time-scale transform is a constant | single per-stratum `uptime_factor` scalar / per-run linear interpolation; 26% of failures in plan-non-producing months; op-day↔ttf_true >10% off on ⅓ of runs (Phase D) | B |
| 3 | hazard layer fit on the old baseline | Ql β=−0.587 from `chem_ql_transform_screen` vs old curves; Kpod/other Cox coeffs from bundle 2026-07-08 baseline | C |

Direction check: causes 1 and 2 both push predictions DOWN when fixed (longer curves,
less phantom exposure), i.e. toward fact — consistent with the observed 0.75.

---

## Workstream A — Weibull refit with Big censoring (biggest lever)

Owner artifacts: extend `scripts/run/refit_weibull_big_censoring.py` audit into the
production bundle builder (`analysis/workflows/esp_survival/vba_bundle.py` input side).

**A0. Corrected ESP row definition (user correction 2026-07-15, RECHECKED).**
The audit's `is_esp` was only a negative filter on `Тип ГНО` (not ВОРОНКА/УГРП/ПАКЕР/
ОТСУТСТВ) and ignored «Цель спуска» entirely. Corrected definition, all three gates:

1. **«Цель спуска» = «Мех. добыча»** (col E; add `purpose` to `equipment_big._SELECT`
   and expose `is_esp_strict`) — drops водозаборная ВД/НД, нагнетательная, пьезометр,
   фонтанная, ГРП, техн. операции rows;
2. `Насос (50Гц)/Тип ГНО` confirms an ESP via a **positive vocabulary**: `ЭЦН*`
   families PLUS REDA/SLB model designations (`D####N`, `G####N`, `S####N`, `GN####`,
   `ESP 5xx-…`, `MT5A-…DP`) — the literal-«ЭЦН» test alone loses 167 genuine SLB units;
   **exclude `ВНН`** (screw pumps, 242 мех-добыча rows) and ШГН-type entries. The
   ~25 distinct non-ЭЦН model families go to the user for a one-time vocabulary review;
3. `Насос (50Гц)/Собственник оборудования` = contractor (`Борец`→brt, `Шлюмберже`→slb,
   else oth) — loader support already in `equipment_big.py`.

**Recheck result (2026-07-15):** of the audit's 810 Big-only censored rows: 567 are
Мех. добыча (243 were non-producer wells wrongly included); only 66/567 exist in the
fresh Свод (`data/inputs/Отказы свод с анализом.xlsx`, sheet «Свод», 2891 rows) by
well+mount±7d; only 10/810 are in V03 at ±7d (the exact-date dedup was sound). So
**501 genuinely missing censored ESP runs remain** (Ya-stratum 165, Vt 131, Mc 56;
median ННО 333 op-days; installs mostly 2024–2026). Verified these are **one current
pump per well across 501 distinct wells** — every row is the max-install-date run on
its well, no fail date, no demount, ZERO cases of a later run superseding them. This
matches the physical reality the user flagged: Big is a workover/failure ledger with no
"running-well" category, but the last run of each well IS the currently-alive pump →
legitimate right-censoring, exactly the tail the fit is missing. 501 ≈ the ~500-well
fleet. List for manual check: `results/big_open_runs_501.csv`. Residual risk: wells
stopped/converted without a recorded demount — spot-check against the techregime
current-status feed in A1. The audit's B50 shifts (×1.5–3.1) were computed on the
polluted 810 and are overstated; **re-run the refit on the corrected subset FIRST**.
Recheck scripts: scratchpad `recheck_big_only{,2}.py`, `open_runs`/`wo3` (this session).

**A1. Survival input = fresh Свод events + Big-only censored ESP rows.**
Event side rebases on the fresh Свод (`data/inputs/Отказы свод с анализом.xlsx`,
factual through 2026-06 — same source the packaged run now pins) rather than the stale
V03 mart alone; reconcile counts mart-vs-Свод per stratum as an audit table. Big-only
selection: corrected A0 filter, absent from Свод AND V03 by `(well_key,
install_date±7d)`, ННО > 0, prefix-mappable to a model field.

**A2. Event vs censoring by PULL REASON, not by presence of a fail_date.**
This is the second over-prediction lever and applies to the WHOLE fit population
(mart/Свод AND Big-only), not just the added rows. Big stamps «Дата отказа» on
workover runs too (57b commit), so any fit that treats fail_date as the event marker
over-counts failures → curves too short. Serial evidence (2026-07-15, pump stage-1
serial col 31) shows a ГТМ pull is a genuine run termination — the same physical pump
returns only **4.4%** of the time (5.6% after a real failure, 12.5% after ППР); the
`нов/рем` flag is unusable (98.5% empty). So classify by pull reason:

- pull reason ∈ {ГТМ, ППР} → **censored at ННО** — pump was healthy, pulled for well
  intervention, ~95% replaced by a different pump. Applies EVEN when fail_date is
  filled. (мех.добыча ГТМ count ≈ 1,469; ППР ≈ 21 — a large censored mass currently
  likely mis-booked as failures.)
- pull reason a genuine failure (Снижение изоляции/R-0, Отсутствие подачи, Нет звезды,
  Клин ЭЦН, Нет подачи-токи х.х., Снижение подачи, …) → **event at ННО**.
- last run per well, no pull → **censored** at current ННО (the 501).
- Same-pump-back after workover (~4–5%) is negligible — do NOT stitch consecutive runs
  into one pump-life; keep the run (install→pull) as the survival unit.

Report event/censor counts per stratum vs the old fit, AND the count of runs that flip
event→censored under this rule (the spurious-failure removal) — that delta is the
expected downward pressure on predicted rates.

**A3. Sour/contractor strata inheritance** for Big-only rows: `h2s_proxy_mg_l` from
mart by `well_key`, then by `pad`; unknown → nonsour + audit count (per the findings
note recommendation).

**A4. Fit.** Existing K=2 EM machinery, same constraints (β₂>max(1,β₁), η₂/η₁≥2),
same cascade `field_sour_ctr → field_sour_Pooled → Global_Pooled`. Clock stays
**op-days (ttf_mix)** — the calendar mapping is Workstream B's job, not a clock change.

**A5. Mc special path.** Integrate `production_risk.mc_refit` (Свод+Big, MR→Mc,
install-window) into the same builder rather than a post-hoc row swap. Re-test the
vintage window on the censoring-corrected population: the 2024+ window (b50=224) was
chosen when censored mass was missing — with censoring included the vintage shift may
shrink or move. Fit 2023+, 2024+, and all-vintage variants; pick by A6 + D1 criteria,
document the choice.

**A6. Per-stratum acceptance (survival side).**
For Mc/Ya/Vt strata: model S(t) overlaid on IPCW-KM of the extended population;
max |ΔS| < 0.05 on t ∈ [0, b80]; b50 inside the KM CI; no degenerate strata regression
(k1_degenerate list from project memory). Charts to `results/esp_survival_big_censored_refit/<date>/figures/`.

---

## Workstream B — data-based ttf_true → calendar mapping

Today's transform (in `failure_rate._append_interval_predictions` and the forecast):
per-run linear interpolation when ННО is known, else `calendar_overlap × uptime_factor`
(one scalar per stratum, from `vba_bundle._uptime_factor`). Replace with a fitted
mapping; "data-based, not simple fix".

**B1. Build the mapping dataset.**
Sources: warehouse daily telemetry (op flags/qliq>0, 1.25M rows), techregime
`treg_in_operation`, Big ННО vs calendar span (mount→fail/demount). One row per
run-month: calendar days, operating days, field, stratum, calendar month (season),
op-age at month start, run ordinal. Reconcile the known op-day↔ttf_true >10%
discrepancy (Phase D) on this dataset first — decide which op-day definition is the
model clock's ground truth and use it consistently (fit AND replay).

**B2. Candidate mapping models** (fit per field, pooled fallback; select on held-out
runs by two errors: (i) reconstructed total ННО vs actual, (ii) monthly op-day
placement MAE):

1. age-dependent uptime curve u(a_op) — monotone spline / piecewise constant on age
   bands (early flush, mature, late);
2. \+ calendar-month seasonal component (winter shut-ins) — additive logit;
3. two-regime with change point (cheap fallback if 1–2 overfit).

Deliverable: per-stratum mapping table/coefficients in the bundle
(`esp_time_map.csv`), loader in `survival.StrataModel`, used by BOTH the historical
replay (replacing the flat `uptime_factor` branch AND the within-run age placement of
the linear-interpolation branch where telemetry gives a better profile) and the
forward projection (converting plan op-days to hazard exposure consistently).

**B3. Idle-month failures (the 26% problem).**
26% of genuine failures (34% Мирнинский) fall in plan-non-producing months where the
exposure-masked replay places zero hazard. Measure, per field, the empirical hazard in
idle vs producing months (failures / month-at-age, split by op-days>0). Then decide
from the data, not by fiat:

- if idle hazard ≈ 0: failures are mis-dated (fail recorded at pull, not stop) →
  attribute to last producing month (numerator shift);
- if idle hazard > 0: add a fitted idle-hazard fraction λ_idle/λ_op to the model
  (hazard accrues on calendar clock at reduced rate during idle) — this changes both
  fit (interval durations) and replay.

This single decision is likely the main Мирнинский fix beyond A (memory: on a
consistent producing basis the model already over-predicts Mc; the 1.29→0.60 story
flipped with the workover fix — re-measure after A lands).

**B4. Acceptance.** Held-out run reconstruction: median |ННО_pred − ННО| / ННО < 10%
(vs the current scalar's error, must strictly improve); monthly op-day placement MAE
reported per field.

---

## Workstream C — hazard analysis redone on the new baseline (Ql added, Kpod kept)

The Cox layer (`esp_survival/hazard_layer.py` coefficient card → `esp_cox_coeffs.csv`)
was fit against the old-baseline strata on ttf_mix. After A+B change the baseline and
exposure, every β is stale by construction. Redo, don't patch:

**C1. Covariate set.** Keep the existing card groups — incl. **Kpod**
(`frac_kpod_below_0p7` chronic_underload, `kpod_freq_mean` delivery_coef), GLF, load,
frequency, curvature, well_history, vintage. **Add Ql** as a time-varying monthly
covariate: field-ref-centered capped log rate (re-screen the transform on the new
baseline — do NOT carry β=−0.587/γ=+0.093 forward; they were fit on the short curves).
Qw stays audit-only.

**C2. Fit design.** Stratified Cox (strata = new A-baseline strata) on the extended
population; static covariates as now; Ql via monthly time-varying intervals
(lifelines TV-Cox — mind the Phase-D gotchas in memory). Age×Ql interaction refit;
verify the sign-crossover against empirical hazard by Ql tercile within age bands
{<180, 180–533, >533} before shipping any interaction.

**C3. Covariate availability audit.** Big-only rows have no telemetry early-window
features → Kpod/GLF missing there; mean-imputation (reference-neutral) as now, but
report the share of runs with real vs imputed covariates per stratum (the CatBoost
work's `covariate_share` machinery is reusable).

**C4. OOS gate (project standard — honest null).** Time split: fit ≤2024-12, score
2025+. Metrics: ΔC-index and Δmonthly-rate-MAE (hazard line vs base line) per УН.
Ship the layer in the stress scenario regardless (sensitivity), but promote to the
PRIMARY scenario **only** if it wins OOS — previous θ overlay added nothing
(ΔAUC ≤ 0.004), so expect null and don't tune to win.

**C5. Deployment consistency.** Whatever transform C1 selects must be computed
identically at fit and serve (train/serve skew check from
`production_risk_ql_hazard_verification_plan.md` Tier 1.5): same op-day-normalized
rate, same refs, same clip. Historical replay must then use **actual** historical Ql
(from B1's run-month dataset), not plan Ql — this also closes the silent
partial-coverage hole in `_apply_ql`.

---

## Workstream D — joint validation and integration (presentation gate, revised after B review)

Workstream B is **not** a confirmed level-fix. The age/season time map is useful mainly
as a within-run monthly placement profile: unscaled it improves monthly op-day MAE only
modestly and reconstructs total `ННО` worse than the scalar; the "scaled_to_run_nno" mode
passes total-life reconstruction only by construction. D must therefore treat B as
placement-only unless it wins a true pre-fit holdout.

The idle-month result is also provisional. Idle failures are non-zero, but the current
measurement is confounded by pull/stop-date misdating: a failure in a month may make that
month non-producing in the plan. This is especially dangerous for Мирнинский, where
idle/producing hazard was >1 before clipping. Do **not** enable the Mc idle fraction by
default until D0 below disambiguates this.

**D0. Pre-gates before any presentation acceptance.**

1. **True time-map holdout.** Refit `esp_time_map.csv` with the well/run split applied
   *before* fitting. Compare against a scalar fitted on the same train split:
   - monthly op-day placement MAE must improve on holdout by a material margin (target
     ≥10%, but report if smaller);
   - unscaled total-`ННО` reconstruction must be reported honestly; the deployed
     known-`ННО` historical mode remains scaled-to-run-total.
   If the edge disappears, D uses the time map only as an audit/sensitivity line, not
   as a claimed cause of level correction.
2. **Idle misdating disambiguation.** For every genuine failure in the fact window,
   compute both:
   - recorded failure month;
   - last month before failure with plan op-days/production > 0.
   Re-measure idle vs producing hazards after last-producing-month re-attribution.
   Decision rule:
   - if Mc idle hazard collapses toward producing or near zero, treat Mc idle failures
     as misdated numerator and shift them for history only; do not add forward idle
     exposure for Mc;
   - if Mc remains materially idle-hazard-positive after re-attribution, fit a capped
     Mc idle fraction and include it as a separate sensitivity before promoting it;
   - Ya/Vt idle fractions may be tested, but must pass the same re-attribution audit.
3. **Scenario ladder.** D acceptance is evaluated on a transparent ladder, not one
   blended line:
   - `A_only`: censoring-corrected survival, legacy scalar exposure, all manual factors
     set to 1.0;
   - `A_plus_time_placement`: known-`ННО` intervals scaled to run total using the time
     map as placement weights;
   - `A_plus_idle_reattributed`: numerator shifted to last producing month where D0 says
     misdating dominates;
   - `A_plus_idle_hazard`: fitted idle exposure only for fields that pass D0 as real
     idle hazard;
   - `A+B+C`: optional hazard layer after C, primary only if it wins OOS.

**D1. Acceptance criteria (the whole point).** With the selected D scenario
(`esp_models.csv` plus optional `esp_time_map.csv`/idle policy/`esp_cox_coeffs.csv`) and
ALL manual calibration factors set to 1.0:

| Check | Target |
| --- | --- |
| fact/model 2024-01..2026-06, base line | Мирнинский, Ярактинский, Верхнетирский each within **0.90–1.10**; ГЛОБАЛЬНО within 0.90–1.10 |
| survival curves | A6 per-stratum KM overlay passes |
| monthly shape | per-field monthly MAE ≤ current calibrated line's MAE (we must not trade level for shape) |
| time-map claim | placement MAE improvement reported from true holdout; if not material, label as near-null and do not cite as level fix |
| idle policy | Mc idle policy chosen only after last-producing-month re-attribution; no forward Mc idle exposure unless the real-idle gate passes |
| forecast sanity | 2026-07+ expected failures continuous at forecast_start (no seam), fleet-rate within historical envelope; planned-idle exposure must not create a Mc step-up |

Minor fields (Кийский, Большетирский, Марковский…) stay chart-suppressed; small
reporting-field residual factors are permitted there ONLY, re-derived and re-dated.

**D2. Rebuild + regression.** Rerun `production_risk` full-tables for every scenario
in the D0 ladder. Verification plan (`production_risk_ql_hazard_verification_plan.md`)
Tiers 0–3 apply as-is after C; for D0/D1 additionally emit:

- `D_time_map_holdout.csv`: true train/holdout scalar vs time-map metrics;
- `D_idle_reattribution.csv`: recorded-month vs last-producing-month failure counts and
  idle/producing hazard by field;
- `D_scenario_ladder_fact_model.csv`: fact/model and MAE for A-only, placement-only,
  reattributed-idle, real-idle-hazard, and optional hazard layer.

Retire stale manual factors (Мирнинский 0.929, Ya/Vt factors, reporting-field factors)
only after one scenario passes D1 without them; otherwise report the residual explicitly
instead of recomputing factors per run. Update workbook/launcher notes (drop "Mc
survival-weight" from `build_production_risk_launcher.py`); rebuild EXE last.

**D3. Presentation artifacts.** Per УН (Mc/Ya/Vt + ГЛОБАЛЬНО): fact vs A-only vs
selected-D scenario vs hazard(Ql+Kpod, if promoted) monthly-rate chart, KM-vs-model
survival overlay, and one slide-ready table: old fact/model → A-only → selected-D,
B50 old→new, what changed. Say plainly that the time-map scalar was near-null if the
true holdout confirms it; do not sell B as a level fix unless the D0 evidence supports
that. Russian labels.

---

## Order, dependencies, effort

```text
A (Weibull+censoring refit)  ──┐
B (time map, near-null audit) ──┼─→ D0 disambiguation → D1 gate → C (hazard redo) → D2/D3
   idle-month provisional ──────┘
```

- A and B1 are independent and can run in parallel; B's idle-month result is provisional
  and must be re-decided in D0 with last-producing-month re-attribution, especially for Mc.
- C is deliberately AFTER the baseline freezes — hazard βs against a moving baseline
  are throwaway work.
- Fast path for presentation: if A-only or A+placement already brings Ya/Vt into
  0.90–1.10 and D0 resolves Mc via re-attribution or a justified idle policy, C can ship
  as stress-only sensitivity with re-screened Ql and need not block presentation.

Estimated sessions: A ≈ 1–2 (builder + fit + acceptance), B ≈ 2 (dataset, model
selection, integration), C ≈ 1–2 (TV-Cox + OOS), D ≈ 1. Suggested first commit:
Workstream A builder + A6 charts, no deployment.

## Known traps (from project memory — do not rediscover)

- Big «Дата отказа» includes ГТМ/ППР workovers — always gate events with
  `_is_failure_pull` (commit 57b lesson).
- `esp_run_covariates.csv` is keyed by within-well ordinal, not v03 `Нспуска`
  (log_run_seq 10× bug).
- Big join key: `(well_key, install_date)` with norm_well; MR→Mc kept, NE/AM/ZYI/YAY
  stay in `EXPLICIT_GLOBAL_FALLBACK`.
- lifelines TV-Cox predict gotchas (Phase D memory) — validate hand-rolled partial
  hazard against lifelines on a sample.
- Recomputing calibration factors per run forces ratio→1 and hides drift — factors, if
  any survive, are derived once and dated.
- numpy-repr in .bas exports; ChrW() for Cyrillic in VBA.
