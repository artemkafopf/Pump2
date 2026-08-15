# Pump sizing vs operating life — findings

**Date:** 2026-07-28 · **Slug:** `production_risk_pump_sizing_landmark`
**Code:** `backend/analysis/workflows/production_risk/pump_sizing_landmark.py`,
`backend/analysis/features/free_gas.py` · **CLI:** `scripts/run/pump_sizing_landmark.py`

---

## The question

Two strategies, both present in the fleet:

* **A — oversize.** Big pump, take the rate early, let pressure and rate fall away; Kpod
  drifts down and its average ends up low.
* **B — right-size.** Smaller pump, hold the rate, produce at roughly constant Kpod.

Does A cost pump life, and how much?

Mean Kpod cannot answer it: `corr(Qnom, Kpod_mean) = −0.04`. The strategies are invisible
in the level and live in the trajectory. That is why the analysis is built on a landmark
rather than a run-average.

## Design in one paragraph

Every feature is computed on a **fixed window of the first 90 operating days**, identical
for every run. The outcome is **residual life measured from that landmark**, right-censored,
on the `t_cal` clock. Runs that never reach the landmark are excluded and counted. Pump
size is read two ways: between-well with deliverability controls (`kprod`, `rpl`,
drawdown), and **within-well** via a Cox model stratified on `well_key`, which uses only
wells that ran different pump sizes at different times, so reservoir quality cancels.

Fixes three defects in the naive version, each of which produces a plausible wrong answer
rather than an error:

| Defect | Naive result | Fix |
|---|---|---|
| Slope fitted over each run's own length | correlates with run length semi-definitionally (ρ=+0.35) | fixed 90-op-day window |
| 41 % of runs reach 90 op-days; the rest are infant failures | silent survivor selection | explicit conditional estimand + selection audit |
| Features and outcome overlap | leakage | windows disjoint by construction |

---

## Results

### 1. Oversizing costs life — and the confound was hiding it, not creating it

| Estimate | HR per e-fold Qnom | 95 % CI | p |
|---|---|---|---|
| Between-well + deliverability controls | 1.46 | — | LR < 1e-7 |
| Ya | 1.58 | — | 2e-5 |
| Vt | 1.41 | — | 0.014 |
| **Within-well (stratified Cox)** | **2.01** | **[1.59, 2.54]** | **4e-9** |

The within-well estimate is **larger** than the between-well one. That is the expected
direction given `corr(Qnom, kprod) = +0.64` — big pumps are chosen for productive wells,
so the naive comparison is biased *toward* "sizing is harmless."

Stable across landmarks and thresholds: within-well HR = 2.06 / 2.01 / 1.98 at landmarks
60 / 90 / 120, and 2.01 vs 2.01 at ≥1.25× vs ≥1.5× size change.
`run_seq` HR = 0.90 (p = 0.001) — later runs on a well live *longer*, so the size effect
is not reading depletion order.

**Priced:** doubling nameplate flow costs

| | RMST lost | of baseline | per m³/d gained |
|---|---|---|---|
| between-well | ~52 d | 11 % | 0.33 d |
| within-well | ~101 d | 22 % | 0.62 d |

against a 465 d residual RMST baseline, buying ~161 m³/d more liquid (fitted elasticity).

### 2. The trajectory hypothesis — right sign, but it does not replicate

Kpod slope enters with the hypothesised sign — flatter Kpod → lower hazard, HR 0.89/SD
pooled, 0.84 on Ya — and the trajectory block is significant pooled (LR p = 0.038) and on
Ya (p = 0.017).

**But on Vt it reverses sign (HR 1.12) and loses out-of-sample** (CV −6.9073 → −6.9378);
Mc is worse. Per the standing replication gate, a covariate that dies on replication is
dead: **the trajectory layer is not shippable.** It is a Ya result, not a fleet result.
Sizing, by contrast, replicates on Ya *and* Vt *and* within-well.

### 3. Free gas at intake — an honest NULL as the mediator

This was the mechanism the analysis was built to test, and it fails.

* Adding β attenuates the Qnom coefficient by **0.4 %** (Ya 0.8 %, Vt 2.0 %).
* LR test for β non-significant everywhere: pooled p = 0.36, Ya 0.26, Vt 0.065.
* Zero attenuation at landmarks 60, 90 and 120 alike.

β is well built, so this is a real null rather than a broken feature — see below.
**The size effect does not run through gas interference at the intake.**

---

## Two dead ends worth recording

**`P_intake / P_bubble` does not work.** Median 0.246; **99.0 % of runs sit below bubble
point.** "Is there free gas?" is a constant on this fleet, so any threshold indicator —
including the existing `frac_pzab_below_1` — carries almost no cross-run information.

**Raw Standing does not transfer here.** The gate `Rs(P_b) ≈ GOR` fails: median ratio
**0.55** (Za 0.24, Mc 0.28, Az 0.30, Vt 0.43, Ya 0.76), and no PVT set in the sweep gets
past 0.78. Reported producing GOR is 1.3-4× what the fluid can hold at its own reported
bubble point — the signature of gas-cap/coning contribution, possibly plus a m³/t vs m³/m³
convention (~15 %).

The fix is `solution_gor_anchored`: keep Standing's *shape*, pin the endpoint to the
warehouse's own `(P_b, GOR)`. Ordinary correlation tuning, and it makes β
**rank-invariant to the PVT assumptions — Spearman 1.0000** across API 28-36, T 60-100 °C,
γg 0.70-0.90 (median moves 4.1 %, vs 10.7 % unanchored). Since β is consumed as a ranking
covariate, the assumed API/temperature/gas gravity are provably irrelevant to the result.

⚠ **β's level is not usable, only its ordering.** Median β = 0.586 with 66 % of daily rows
above 0.40 — a pump truly ingesting 59 % free gas would gas-lock continuously, and these
wells run. The gap is the separation credit the warehouse cannot supply (natural
separation alone is commonly 50-90 %). Never report "β = 0.59 exceeds the 0.15 tolerance."
This also keeps the null in §3 honest: if separator fitting correlates with pump size, a
true mediation could be masked, and the warehouse records no separator equipment.

---

## What this cannot see

**Half of all failures happen before the landmark** — 50.8 % at L90 (41.6 % at L60, 57.2 %
at L120) — and are excluded by construction. The estimand is residual life *given survival
to day 90*; it is silent about infant mortality.

The selection audit confirms this is not size-selective: pooled p = 0.078 at L90, no field
significant; at L60 the dropped runs are *smaller* (p = 0.005), which biases against
finding a size effect. **So the sizing result is conservative, not inflated.** But "does an
oversized pump die young?" is a separate, open question — and given that half the fleet's
failures live there, it is probably the more valuable one.

Also open: the within-well design does not fully close the confound, because successive
runs on a well are ordered in time and pump size is entangled with cumulative depletion.
`run_seq` controls it and points the other way, but observational data cannot finish the
job. β was the intended mechanistic tie-breaker and it came back null.

## Next

1. **Infant-band companion analysis** — where the other half of the failures are.
2. **The economic close** — days lost vs m³ gained is computed here per-run; turning it
   into a sizing rule needs the oil price/workover cost side, same shape as the
   fleet-frequency result (+5 Hz ≈ 327 000 m³ per extra workover).
3. Do **not** ship the Kpod-trajectory layer until it replicates on Vt.
