# Mc failure/ГТМ competing-risks accounting — implementation prompt

> Handoff prompt for an implementation agent. Self-contained. Repo: `d:\GitHub\Pump2`.

## Mission

On Мирнинский (Mc), make the failure-count forecast's bookkeeping explicit: separate
**observed failures under current ГТМ practice** from the **latent no-ГТМ counterfactual**,
model failure and ГТМ as **competing risks**, and get the **running (open) pumps** into the
at-risk accounting correctly — with current ages and conditional hazards. Background: the
current forecast matches actual monthly failure counts; the question raised was "shouldn't
it predict MORE, with the surplus consumed by workovers?" The answer depends on hazard
shape (flat hazard ⇒ pulls don't consume failure mass; infant spike ⇒ each ГТМ renewal
ADDS infant-risk exposure) and on whether ГТМ is de-facto preventive (pulls target pumps
about to fail). This analysis quantifies all of it on Mc.

## Standing repo rules (non-negotiable)

- Paths via `from analysis.paths import results_dir, ...`; outputs →
  `results_dir("production_risk_mc_competing_risks")` (`tables/`, `figures/`).
- Reusable logic → `backend/analysis/workflows/production_risk/`; scripts thin, with the
  sys.path bootstrap (`sys.path.insert(0, str(REPO_ROOT / "backend"))`).
- **Do NOT commit to git.**
- **Mc = installs 2024-01-01+ ONLY** — a COHORT filter on install date
  (`config.MC_INSTALL_COHORT_START`, default-on in `esp_population.select` /
  `load_mart_df`), NEVER left truncation (left truncation keeps pre-2024 runs' later
  exposure and hides that recent runs are shorter — tried, wrong).
- Reporting: RMST(0, 730) + MRL(0) for any life quantity; median reference only; KM (and
  here Aalen–Johansen) control next to any parametric curve.
- Windows: run Python with `PYTHONIOENCODING=utf-8`.
- Cross-check fitted event counts vs `esp_models.csv` n_failures before trusting any fit.

## Data

Population via `analysis.workflows.production_risk.esp_population` (`build` +
`add_time_scales`; Mc cohort filter on). Outcome classification: genuine failure vs
ГТМ/workover pull vs still-running, from `pull_reason` (the pipeline already distinguishes
genuine failures — see `failure_rate.py` / Workstream A conventions). Current-pump ages for
running units: the population's open rows as of `config.SVOD_OPEN_ASOF`; `passport.py`
gives real current-pump ages for cross-checking.

### Clock — one clock for everything

Known trap: `tte` mixes clocks (events on Наработка, censorings falling through to
calendar age — on Mc 60/61 open runs did this), biasing survival up. Use **one consistent
clock for events, pulls AND open runs** — `t_cal` (calendar) is acceptable and simplest;
follow the Workstream-B `time_map.py` decisions if a better consistent clock is available.
Document the choice; assert no open run gets a different clock than events.

### Running pumps — the acceptance-critical part

- Open runs MUST be present with correct current ages. Known silent-loss modes to check
  and report: (a) any loader that keeps closed-only (`ingest_v03`-style complete-case —
  do not use); (b) clock fall-through (above); (c) dedup of open rows vs Свод/Big
  (the 171-double-count fix — verify it holds for Mc).
- Reconcile: number of open Mc runs vs the plan's active-producing well count (the fleet
  denominator is the ПП plan's active producers — e.g. 2024-10 = 19, not 36).
- Each running pump enters the forecast with conditional hazards given survival to its
  current age, for BOTH processes.

## Tasks

**A. Cause-specific fits.** Fit cause-specific hazards on the Mc 2024+ cohort:
λ_fail (failure; ГТМ treated as censoring) and λ_gtm (ГТМ pull; failures treated as
censoring). Weibull each (report β, η with bootstrap CIs ≥200 resamples, resampled by
well); if Weibull is a poor fit for ГТМ timing, a piecewise-constant hazard is fine.
Report both shapes prominently — the whole "consumed by workovers" question hinges on
whether λ_fail rises with age (expect: infant-spike-then-flat, no wear-out — that is what
Mc has always shown).

**B. CIF vs naive KM.** Compute the Aalen–Johansen cumulative incidence of failure and of
ГТМ, and compare with the naive 1−KM (ГТМ-censored) failure curve. Quantify how much the
naive curve overstates cumulative failure incidence at 90/180/365 d. This is the honest
"how many failures actually materialise vs how many the net model imagines" gap.

**C. Informative-censoring / preventive-ГТМ test (port from Vt).** Two levels:
1. Association: does the ГТМ hazard correlate with failure-risk factors (rate/Ql,
   contractor, well age)? (On Vt, ГТМ was rate-correlated — expect the same.)
2. Preventive direction: do runs ending in ГТМ show elevated failure precursors in their
   final 2–4 weeks (frequency instability, подача degradation, restarts) vs matched
   survivors at the same age? Use telemetry if coverage allows
   (`resolve_telemetry_db_path`); if coverage is too thin, say so and stop at level 1.
   If ГТМ pulls look preventive, the latent hazard is understated — report the direction
   and a rough bound, do not silently absorb it.

**D. Monthly forecast decomposition.** For a 12-month horizon from `SVOD_OPEN_ASOF`,
simulate the fleet month by month (deterministic expectation is fine; reuse the
production_risk renewal machinery rather than rebuilding it):
1. **Observed-failure forecast** — both processes active; pulls and failures renew the
   well with an age-0 pump (which re-enters the infant window). This column must
   calibrate against actual monthly failure counts (compare on the available actuals
   window).
2. **ГТМ-count forecast** — from λ_gtm, sanity-checked against the ПП plan's workover
   schedule where the plan has one; state which source the column uses.
3. **Total pulls** = failures + ГТМ (the all-cause workshop-load estimand — keep it a
   separate column, never blended into the failure KPI).
4. **Latent no-ГТМ counterfactual** — λ_gtm switched off, pumps run to failure with
   renewal on failure only. Label it clearly as a counterfactual resting on
   non-informative censoring (per C).
5. **Infant-renewal effect of ГТМ** — difference in expected failures between (1) and a
   variant where ГТМ replaces the pump WITHOUT resetting infant risk (age-0 pump but
   hazard continued flat). This isolates "workovers manufacture infant exposure".
Expected picture given flat hazards: (4) ≈ (1), NOT 2×(1); if that's what comes out, say
it plainly — it is the answer to the motivating question.

**E. Reconcile with the canonical forecast.** Compare column (1) against the canonical Mc
build (`results/production_risk_mc_svodprognoz/2026-07-17/`, model cal c0 k2). Does
explicit competing-risks modelling materially move the failure forecast (>5% on the
12-month total)? If not, the current single-hazard forecast is vindicated as an
observed-failure predictor and the new machinery is for the extra columns only —
state that conclusion explicitly.

## Deliverables

1. `backend/analysis/workflows/production_risk/mc_competing_risks.py` + thin CLI script.
2. Tables: `cause_specific_params.csv`, `cif_vs_km.csv`, `informative_censoring.csv`,
   `monthly_decomposition.csv` (columns 1–5 above + actuals), `canonical_comparison.csv`.
3. Figures (Russian labels): CIF stack (failure + ГТМ + running) vs naive 1−KM;
   cause-specific hazard shapes; monthly decomposition bars (observed forecast vs actuals
   vs latent counterfactual).
4. Findings note `docs/notes/mc_competing_risks_findings.md` (Russian): the
   observed-vs-latent answer in one paragraph up front, then evidence.
5. Tests in `backend/tests/`: at-risk accounting sanity (open runs present, single clock,
   cohort filter on), decomposition identity (failures + ГТМ + still-running = fleet
   exposure), and a synthetic two-hazard simulation recovering known cause-specific
   parameters.

## Known gotchas

- Mc is thin (order ~100 runs 2024+, ~60 open) — bootstrap CIs will be wide; report them,
  don't hide them. Any pre-2026-07-17 Mc figure in docs is from the wrong population —
  re-measure, don't cite.
- «Наработка» is calendar-like (corr 0.95–0.99), not op-time; don't present it as op-time.
- Contractor strata on Mc are mostly a rate proxy — don't stratify the cause-specific fits
  by contractor unless events allow; pool thin cells.
- `Σt/N` is a rate, not a life — never report it as ННО/МРП; use RMST off the all-cause KM
  for any МРП-style number.
- The ПП plan drives the well universe and ВНС entries in the canonical forecast — when
  comparing (task E), keep the same universe; do not silently drop plan-driven wells.
