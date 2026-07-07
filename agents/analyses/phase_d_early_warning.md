# Phase D — Within-Run Dynamics: Time-Varying Hazard + Early-Warning Layer

## Purpose

Phase C ended with the program's defining negative result: static baseline covariates,
however real in-sample, **do not transport** to unseen installs (merged θ ≈ chance,
C-index 0.47–0.53). The predictive channel that remains untapped is the one the static
layer cannot reach by construction: **how a run is behaving right now** — trends,
variability, cycling, productivity decline — from the 1.25M-row daily table.

Phase D builds that layer and answers one operational question:

> **Can we flag, H days ahead, which running pumps will fail — materially better than
> "stratum + current age" alone?**

Everything in Phase D is judged by that out-of-sample question. In-sample hazard ratios
are the *diagnostic* view (D2); the *deliverable* is the validated alarm performance (D4)
and, if it earns its keep, a fleet watchlist scorer (D5). If dynamics do not beat the
age+stratum baseline either, that is a reportable structural finding about telemetry
quality — the honest-null clause applies exactly as it did in C5.

Context (read first):

```
results/phase_abc_synthesis/2026-07-07/README.md      — what stands, what didn't transport
results/phase_c_report/2026-07-06/reports/…           — C5 holdout method + why θ failed
results/phase_b_report/2026-07-06/reports/…           — mode physics (restarts/freq → electro-thermal)
agents/analyses/phase_a_cox_foundation.md              — design decisions, registry
backend/analysis/models/survival/temporal_holdout.py   — IPCW Brier / C-index machinery (reuse)
CLAUDE.md                                              — path rules
```

---

## §0 — CONFIRM WITH USER BEFORE IMPLEMENTING

### 0.1 The alarm product definition (drives every design choice downstream)

| Parameter | Default (confirm or change) | Why it matters |
|---|---|---|
| Prediction horizon H | **60 operating days** (30 and 90 as secondary) | What "early" means for interventions |
| Landmark cadence | every **30 operating days** of run age | How often the watchlist refreshes |
| Actionable capacity | evaluate at **top 5% and top 10%** of running fleet flagged | Ops can only inspect/act on so many wells; precision@k must match reality |
| Scope | all-cause primary; hydraulic & electro-thermal secondary | Phase B says the precursors differ by mode |

### 0.2 Guard gap (leakage against "predicting" an already-dying pump)

Daily data immediately before `stop_date` is often the failure in progress (qliq → 0,
freq → 0) or the shutdown that *is* the event. Features at landmark t therefore predict
events in **(t + g, t + g + H]**, with guard gap **g = 7 operating days** by default.
**Confirm g with ops reality**: how far ahead of the recorded stop does a failing pump
typically show terminal behavior, and is `stop_date` the stop or the pull? Sensitivity at
g ∈ {3, 7, 14} is mandatory regardless.

### 0.3 Eligibility

Runs enter the landmark frame when they have ≥ **30 valid telemetry days** before their
first landmark (else skip until they do). Confirm; state the resulting population size
(~81% of runs have some telemetry; per-landmark coverage will be reported, not assumed).

---

## Design decisions (inherited — not re-litigated)

- Clock: **operating days** (`ttf_mix` convention). Daily rows are calendar-dated → D0
  builds the calendar→operating-day mapping (operating day = qliq > 0, consistent with
  `proc__ttf_true`); every landmark and horizon is in operating days.
- Strata (`stratum_key`), global β, `cluster_col='well_key'`, robust — for the D2
  inference view. Time-varying coefficients only via the sanctioned episode-split module.
- Cause-specific machinery from Phase B for the mode views.
- Intervals on everything; ΔAIC for nested choices; multiple-testing grid discipline;
  §0.2-style causal disclaimer on all D2 outputs (dynamics are *even more* confounded by
  indication than static settings — a rising load trend may be the operator responding).
- **Prediction-first discipline (new, the C5 lesson codified):** no Phase D feature or
  model earns a place in the report's conclusions on in-sample significance alone; every
  conclusion cites its out-of-sample delta.

---

## Leakage rules (named, tested — the phase fails if any of these is violated)

1. **Past-only features:** a feature at landmark t may use daily rows with operating-day
   index ≤ t only. Enforced by construction and by a unit test: recomputing features at
   landmark t after truncating all data > t must be bit-identical.
2. **Guard gap:** events in (t, t + g] are excluded from the target window (neither
   positive nor at-risk credit); runs failing there contribute their earlier landmarks only.
3. **Temporal split before feature selection:** the train/test split (D3) is fixed
   *before* any feature screening; all screening happens on train only.
4. **No outcome-clock features:** nothing derived from run_days/ttf totals (the
   `idle_frac` lesson — HR 24 of pure reverse causation). Cycling enters only as
   trailing-window counts.

---

## Tasks

### D0 — Landmark frame + leakage-proof feature builder (output: `phase_d_landmarks`)

`backend/analysis/data/landmark_features.py`:

- Calendar→operating-day mapping per run (from `proc__daily_merged` qliq > 0, reconciled
  against `proc__ttf_true`; discrepancies > 10% flagged per run).
- Landmark frame: one row per run × landmark (30, 60, 90, … operating days while alive),
  with target columns per horizon and guard gap (event-in-window / censored-in-window /
  survived-window; cause codes from Phase B loader).
- Trailing-window features (windows 14d and 30d, past-only), all with `_missing`
  indicators and per-landmark coverage:

| Family | Features | Prior expectation |
|---|---|---|
| Productivity decline | `kprod` level, slope (Theil–Sen), % drop vs run best; `qliq` slope | the classic precursor; hydraulic |
| Electrical | `load` level, slope, std; excursion count (>90% days) | electro-thermal |
| Frequency regime | mean, std, step count (\|Δf\|>1 Hz), days >55 Hz | electro-thermal (Phase C corroborated) |
| Cycling | restarts (qliq 0→>0) per window, days since last restart, longest idle spell in window | electro-thermal (inrush) |
| Hydraulic regime | intake pressure `rpump_intake` level+slope, `rzab` slope, drawdown proxy, watercut level+jump, gas_factor level+slope | hydraulic |
| Age & static context | operating age at landmark, stratum, run_seq, is_sour_flagged | the baseline to beat carries these |

- Tests: synthetic daily frames for every feature; the truncation-invariance leakage test
  (rule 1); guard-gap target correctness test.

### D1 — Descriptive precursor atlas (output: `phase_d_precursors`)

Before any model: for failures vs survivors matched on stratum × age band, plot the
distribution of each feature in the last pre-guard window vs earlier windows. This is the
"is there anything here at all?" look — cheap, and it caps expectations honestly.
Deliverable: one grid figure + a ranked table of standardized differences (train split
only, per rule 3).

### D2 — Inference view: time-varying Cox (output: `phase_d_tv_cox`)

Episode-split the landmark frame (start = landmark, stop = next landmark/event/censor)
and fit stratified time-varying Cox — all-cause, then hydraulic / electro-thermal
cause-specific — on a screened subset (cascade adapted from `screening_cox.py`, run on
train only). Purpose: which dynamics carry hazard and on which mode (engineering
knowledge, §0.2-style disclaimer). Check the Phase B/C priors: cycling and frequency
instability should land electro-thermal; kprod/qliq decline hydraulic.

### D3 — Prediction view: landmark risk models (output: `phase_d_landmark_models`)

- Primary split: **install cohort** (train installs ≤ 2023-12-31, test after — the C5
  convention); secondary: landmark-calendar-date split (train landmarks before mid-2025,
  test after) as a robustness read.
- Models, deliberately modest and in increasing order of complexity — stop at the first
  that clearly beats baseline: (a) **baseline = stratum K=2 hazard at current age**
  (age+stratum only — the null to beat); (b) Cox landmark model on D0 features;
  (c) gradient boosting on the same frame (only if (b) fails to use its features —
  interpretability is secondary here, this is the prediction view).
- Per §0.1: horizons 30/60/90d, guard-gap sensitivity g ∈ {3,7,14}.

### D4 — Alarm evaluation: the phase verdict (output: `phase_d_alarm_eval`)

On the test split only, per horizon and landmark:

- Dynamic AUC / IPCW C-index and Brier vs baseline (reuse `temporal_holdout.py` parts);
- **Operational metrics at top-5% / top-10% flag rates**: precision, recall,
  **lead-time distribution** (operating days from first alarm to failure),
  **false alarms per pump-year**;
- Calibration of predicted window-risk;
- Well-cluster bootstrap CIs on every delta vs baseline.

The verdict line the report must contain: *"At H=60d and top-10% capacity, the dynamic
model catches X% of failures with median lead time L days at F false alarms per
pump-year, vs Y% for age+stratum alone"* — with intervals, or the honest null.

### D5 — Fleet watchlist scorer (output: `phase_d_watchlist`; **gated on D4 success**)

Only if D4 shows a material, interval-backed improvement: a thin script that scores every
currently-running pump at its latest landmark and emits a ranked CSV
(well, stratum, age, risk in next H days, top drivers per pump, data-freshness flag).
This is the operational payoff artifact. If D4 is null → skip, and instead write the
telemetry-improvement recommendation (which channels/coverage would most raise the
ceiling, from D1's atlas + coverage tables).

### D6 — Report (output: `phase_d_report`)

The story: precursor atlas → what carries hazard (D2, mode-checked) → does it predict
(D3/D4, the verdict with intervals) → watchlist or telemetry recommendation (D5).
§0 decisions in the header; leakage rules attested (test names cited); prediction-first
discipline visible (every conclusion carries its out-of-sample delta).

---

## Rules

- Paths via `analysis.paths`; reusable logic in `backend/analysis/` (`data/landmark_features.py`,
  model additions under `models/survival/`); scripts thin (`scripts/run/phase_d_*.py`).
- No VBA changes; no stratum changes; θ stays on the shelf (Phase C verdict stands).
- `cd backend && python -m pytest tests/ -q` green including the new leakage tests; the 4
  pre-existing `analysis.plotting` collection errors: do not touch, do not add to.
- Compute note: landmark frame ≈ 2,000 telemetry runs × ~5–15 landmarks ≈ 10–30k rows —
  everything here is laptop-scale; if something is slow, the design is wrong.
- Commit only when the user asks.

## Definition of done

- [ ] §0.1 product parameters, §0.2 guard gap, §0.3 eligibility confirmed and recorded
- [ ] D0 frame built; truncation-invariance and guard-gap tests pass
- [ ] Precursor atlas published (train only)
- [ ] Time-varying Cox: all-cause + two mode views, screened on train, disclaimered
- [ ] Landmark models vs age+stratum baseline, both splits, three horizons, g-sensitivity
- [ ] Alarm verdict line with CIs (or the honest null) in the report
- [ ] Watchlist scorer (if earned) **or** telemetry-improvement recommendation (if not)
- [ ] Phase D report; open questions carried forward
