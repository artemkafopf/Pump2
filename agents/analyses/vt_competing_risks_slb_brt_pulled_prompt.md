# Vt competing-risks: slb+brt strata & pulled-runs accounting, H2S from LAB only — implementation prompt

> Handoff prompt for an implementation agent. Self-contained. Repo: `d:\GitHub\Pump2`.
> Sibling analysis (port, don't rebuild): `backend/analysis/workflows/production_risk/mc_competing_risks.py`
> and its prompt `agents/analyses/mc_competing_risks_running_pumps_prompt.md`. Mc verdict for contrast:
> flat λ_fail, latent≈observed (+7%, not ×2), ГТМ ≈ independent censoring (L1 assoc ~0; L2: ГТМ runs
> STABLER than failing runs in their final weeks).

## Mission

Port the Mc failure/ГТМ competing-risks accounting to **Вахитовское (Vt)**, where the priors are
the OPPOSITE of Mc's on every axis, and answer two standing questions:

1. **Vt_slb+brt (contractor axis).** The known "brt/slb look equal on failure-only data" result is
   suspected to be an **informative-censoring collider**, not a real equality. Redo the brt-vs-slb
   comparison the right way: cause-specific hazards + Aalen–Johansen CIF, ГТМ modelled as a
   competing risk — does a real contractor difference emerge once ГТМ pulls stop being free
   censoring? (`oth` is the largest and worst group in sour — keep it as a reported third stratum,
   do not silently drop it, but the headline comparison is slb vs brt vs slb+brt pooled.)
2. **Vt_pulled (ГТМ axis).** On Vt, prior evidence says ГТМ is **rate-correlated** (informative),
   unlike Mc. Quantify it with the same two-level test (association + telemetry precursors in the
   final weeks) and state what it does to the latent no-ГТМ counterfactual: on Vt the latent
   column is expected to be **understated** — report the direction and a rough bound, do not
   present the counterfactual as clean.

**The key data change: H2S class comes from LAB data ONLY.** Every previous Vt sour/nonsour split
used the Свод «Кислый/Некислый» flag (`esp_population`'s `h2s_class`). Do NOT use that flag here,
and do NOT use the V03 workbook's `h2s_mg_l` column (raw__v03_runs / raw__v03_failures — it has a
known mixed-units trap, values 754 vs <1 mg/L, and informative missingness). The only admissible
H2S source is `lab.sqlite` (`analysis.paths.resolve_lab_db_path()`), table `lab_samples`, column
`svb_h2s_indicator` — qualitative: `'+'` = H2S/СВБ detected, null = not recorded. Measured facts
about it (verified 2026-07-23): Vt has 745 lab rows on 197 wells, only 67 rows carry `'+'`, and
`'+'` is the ONLY non-null value — absence of a `'+'` is NOT a measured negative.

## Standing repo rules (non-negotiable)

- Paths via `from analysis.paths import results_dir, resolve_lab_db_path, resolve_telemetry_db_path`;
  outputs → `results_dir("production_risk_vt_competing_risks")` (`tables/`, `figures/`).
- Reusable logic → `backend/analysis/workflows/production_risk/`; scripts thin, with the
  sys.path bootstrap (`sys.path.insert(0, str(REPO_ROOT / "backend"))`).
- **Do NOT commit to git.**
- Vt uses **all history** (the installs-2024+ cohort rule is Mc-only).
- **Never pool sour and nonsour** — they differ ~3× in life (RMST 132 vs 409) AND in shape.
  Every fit, CIF and decomposition is per H2S class.
- Reporting: RMST(0, 730) + MRL(0) for any life quantity; median reference only; KM and
  Aalen–Johansen control next to any parametric curve. `Σt/N` is a rate, not a life.
- Bootstrap CIs ≥200 resamples, resampled by WELL (cluster), not by run.
- Windows: run Python with `PYTHONIOENCODING=utf-8`.
- Cross-check fitted event counts vs `esp_models.csv` n_failures per Vt stratum before trusting
  any fit (the registry bundle: `results/esp_survival_vba_models/2026-07-15-mc2023plus/`).

## Reuse, don't rewrite

`mc_competing_risks.py` already has: the cause classifier (failure=1 / ГТМ=2 / censored=0 on
`pull_reason`), single-clock population builder with the t_cal assertion, cause-specific Weibull +
well-cluster bootstrap, piecewise band hazards, AJ CIF vs naive 1−KM, the two-level
informative-censoring test (incl. the telemetry final-window precursor comparison), the
deterministic daily renewal engine with the 5-column decomposition, and Russian figures.
**Generalize/parameterize it (field, strata iterator, H2S classifier injection) or extract the
shared engine — do not copy-paste 800 lines into a Vt clone.** Mc behaviour must stay
byte-identical (its tests must still pass).

## Data

Population via `analysis.workflows.production_risk.esp_population` (`build` + `add_time_scales`),
`field == "Vt"`. Outcome classification identical to Mc (`WORKOVER_REASONS` → ГТМ,
`GENUINE_FAILURE_REASONS`/failed-unit → failure).

### H2S class from lab — the acceptance-critical part

Build a **well-level** three-way class from `lab_samples`:

- `sour_lab` — any sample of the well (any date) has `svb_h2s_indicator == '+'`;
- `nonsour_lab` — the well HAS lab samples but never a `'+'` (a weak negative: the indicator is
  only ever recorded as `'+'`, so treat this as "not detected", not "measured zero");
- `no_lab` — the well has no lab samples at all. **Never default these to nonsour.** Report them
  as their own column everywhere; fits on `no_lab` only if events allow, otherwise excluded and
  said so.

Join key: `lab_samples.well_key` ↔ population `code` via `crosswalk.norm_well` (case-normalize —
telemetry/lab keys are lowercase `vt_XXXX`, population codes are uppercase `VT_XXXX`).

Required diagnostics before anything else (`h2s_lab_coverage.csv`):
per class — wells, runs, failures, ГТМ, open runs; plus the **reclassification cross-tab** lab
class × Свод flag class (how many runs move, in which direction, how many open runs move). This
is the direct successor to the known defect #4 (30 running pumps on sour Vt wells misfiled
nonsour by the flag): a well-level lab class fixes open-row inheritance BY CONSTRUCTION — assert
that (all runs of a well share one lab class, open included).

Then re-measure the headline the flag produced: sour RMST(0,730) with the lab class vs the flag
class — does the "+30% when the 30 open pumps were re-filed" story survive a lab-based split?

### Clock — one clock for everything

Same rule as Mc: **`t_cal` for events, pulls AND open runs** — on Vt the `tte` mixed-clock trap
hit 138/138 open runs. Keep the single-clock assertion (open `t_cal == as_of − install`, as_of =
`config.SVOD_OPEN_ASOF`). «Наработка» is calendar-like (0.95–0.99), not op-time; the sour infant
signal (beta0 ~0.68–0.89) was measured on other clocks — if the calendar clock flattens it, state
that as a clock property (as the Mc note does), don't claim the spike is gone.

### Running pumps

Open Vt runs must be present with correct current ages (expect O(140); the Свод open-row and
dedup fixes are already in `esp_population` — verify counts, report them). Each running pump
enters any forward simulation with conditional hazards given survival to its current age.

## Tasks

**A. Lab-based strata and cause-specific fits.** For each cell of
{sour_lab, nonsour_lab} × {slb, brt, slb+brt pooled, oth}: cause-specific Weibull for λ_fail
(ГТМ censored) and λ_gtm (failure censored), well-cluster bootstrap CIs, plus the piecewise
band hazards (the shape evidence). Pool cells with <15 events into the nearest sensible parent
and say so — do not fit 5-event Weibulls and present them as strata. Report both shapes
prominently per class: sour is expected infant-heavy (β<1), nonsour flatter.

**B. The collider resolution (slb vs brt).** Within each H2S class, compare slb vs brt three
ways: (i) failure-only naive 1−KM (reproduces the old "equal" result), (ii) cause-specific
λ_fail, (iii) AJ CIF of failure with ГТМ competing. Bootstrap the CIF difference
(`cif.bootstrap_cif_difference` exists) at 90/180/365 d. The deliverable sentence: does the
brt/slb equality survive competing-risks treatment, or was it manufactured by ГТМ censoring —
and does the answer differ between sour and nonsour?

**C. CIF vs naive KM.** Per H2S class (and pooled contractors): AJ CIF of failure and of ГТМ vs
naive 1−KM at 90/180/365 d — the honest overstatement table, as on Mc.

**D. Informative-censoring / preventive-ГТМ test (the Vt_pulled core).** Same two levels as Mc,
per H2S class:
1. Association: ГТМ hazard vs rate (mean Qliq from telemetry), age, contractor. Expect
   rate-correlated on Vt — quantify it (this is where Vt should diverge from Mc's null).
2. Telemetry precursors: final 2–4 weeks of ГТМ-ended vs failure-ended runs (frequency
   instability, подача slope, restarts if cheap). Vt is the gassiest, best-instrumented field —
   coverage should allow it; if a class is too thin, stop at level 1 for that class and say so.
   If ГТМ pulls look preventive (precursors elevated), the latent no-ГТМ hazard is understated:
   give the direction and a rough bound (e.g. re-fit λ_fail with the precursor-positive ГТМ runs
   recoded as failures = upper bound), never a silent absorption.

**E. Decomposition with an honesty label.** 12-month five-column decomposition (observed / ГТМ /
total pulls / latent no-ГТМ / infant-renewal effect) per H2S class, seeded from the running
fleet, exactly as on Mc — but the latent column carries the Task-D verdict: if censoring is
informative on Vt, print the bound from D next to it and label the point value "нижняя оценка".

**F. Reconcile.** Fitted event counts vs `esp_models.csv` Vt strata; RMST/shape vs the shipped
2026-07-20 sour refit (3 sour rows, β≈1 memoryless) and the sour/nonsour scan
(`project_vt_weibull_scan`) — flag any headline that moves under the lab-based class, with the
reclassification cross-tab as the explanation.

## Deliverables

1. Generalized competing-risks module (Mc unchanged) + `vt` entry point + thin CLI script.
2. Tables: `h2s_lab_coverage.csv` (+ cross-tab), `cause_specific_params.csv` (per stratum),
   `collider_slb_brt.csv`, `cif_vs_km.csv`, `informative_censoring.csv` (+ telemetry level-2),
   `monthly_decomposition.csv` (per H2S class), `reconciliation.csv`.
3. Figures (Russian labels): CIF stack vs naive 1−KM per class; cause-specific hazard shapes per
   class; slb-vs-brt CIF comparison; decomposition bars.
4. Findings note `docs/notes/vt_competing_risks_findings.md` (Russian): up front — (а) выжила ли
   «равность» brt/slb, (б) информативно ли ГТМ-цензурирование на Vt и что это делает с латентной
   колонкой, (в) что лаб-класс H2S поменял против флага; then evidence.
5. Tests in `backend/tests/`: lab classifier (three-way, well-level inheritance to open runs,
   no default-to-nonsour), single clock on Vt, sour/nonsour never pooled in any fit call,
   decomposition identity, and the existing Mc tests still green.

## Known gotchas

- `svb_h2s_indicator` is `'+'`-or-null — **absence is not a negative measurement**; that's why
  the class is three-way. Missingness is plausibly MNAR (labs sample where trouble is suspected):
  report event rates for `no_lab` next to the classified groups so the reader can judge.
- Do NOT reach for `raw__v03_runs.h2s_mg_l` / `proc__h2s_proxy` as a fallback classifier — the
  continuous column has the 754-vs-<1 mixed-units trap and informative missingness (its NaN
  pattern predicted life better than its values). Diagnostic cross-tab against it is fine;
  classification from it is not.
- Vt sour cells × contractor will be thin — pool per the rule in Task A, report widths, don't
  hide them. brt is the longest-lived group in both classes per the old (flag-based) scan;
  re-check under the lab class before repeating that claim.
- Contractor ≈ rate proxy was a **Mc** finding; on Vt check `support_overlap`-style balance
  (rate, GLF) before attributing a brt/slb gap to the contractor.
- Day-0 failures are kept (clipped to 0.5 d); pull-reason decides event vs censor, not the
  presence of a fail date.
- Telemetry/lab well keys are lowercase; population codes uppercase — normalize on join (the Mc
  module already had this bug once).
- The ПП-plan universe / ВНС entries are NOT part of this analysis (running fleet only) — if
  comparing against any canonical Vt forecast, state the fleet-universe difference explicitly,
  as the Mc note does.
