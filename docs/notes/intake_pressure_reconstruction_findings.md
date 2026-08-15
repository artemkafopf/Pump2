# Рприем reconstruction and free gas at intake — input-data findings

Measured 2026-08-05 against `data/warehouse/pump2.db` (1 251 826 daily rows) and
`data/sqlite/telemetry.sqlite`.  Code: `backend/analysis/features/intake_pressure.py`,
`backend/analysis/models/ml/intake_pressure_ml.py`,
`backend/analysis/workflows/intake_pressure/`.

---

## 1. The missing-data headline is misleading

`rpump_intake` is positive on **56.4 %** of daily rows.  Acting on that number would
waste most of the effort, because the pressure is missing chiefly *because the pump is
off*:

| subset | rows | `rpump_intake` > 0 |
|---|---|---|
| all rows | 1 251 826 | 56.4 % |
| **operating** (`qliq>0 AND freq>0`) | 584 531 | **95.7 %** |
| idle / stopped | 667 295 | ~20 % |

On an idle day there is no intake flow and no gas crossing the intake, so β is not
defined there in any useful sense.  **The real target is the 25 336 operating days
(4.33 %) that lack the pressure.**  Imputing idle days is declared out of scope rather
than silently attempted.

## 2. The gap is bimodal, and that decides the method

Consecutive missing days form 1 608 blocks within wells:

* **8.1 %** of gap days sit in blocks ≤7 days (median block = 1 day) — telemetry dropouts
  inside an otherwise instrumented well.
* **85.4 %** sit in 126 blocks longer than 30 days, concentrated in **17 wells** that are
  0-50 % covered.  Two — `vt_5608`, `au_315` — have *no* observed intake pressure at all
  (only `vt_5608` reaches the modelling panel; `au_315` has 16 operating days and is cut
  by the 30-day minimum).  Because two wells cannot support a score, the cold regime is
  measured by leaving *whole wells* out fleet-wide.

Longest single block: 1 442 days (`ya_418`).  So the problem is not interpolation; it is
extrapolation across years, in a handful of wells.

## 3. Рзаб is *calculated from* Рприем — do not use it

The telemetry column map names the source field **«Расчетное забойное давление»**.  The
warehouse behaves exactly as that implies:

* correlation with Рприем **0.933**;
* present on **93.3 %** of days where Рприем exists, and on **10.3 %** of days where it
  does not.

Two consequences:

1. **It is co-missing with the target**, so a model that leans on it scores well in
   cross-validation and has nothing to run on in production — the train/apply covariate
   mismatch this project has already been bitten by.  It is excluded from every feature
   set and `intake_pressure_ml.FORBIDDEN` exists so it cannot come back by accident.
2. **Calibrating the annulus leg against it is circular.**  Doing so gives median
   absolute error 0.7 atm, R² 0.744 and a physically sane median `h_eff` of 94 m with
   96 % of wells inside 0-600 m — all of which measures the warehouse's own conversion
   formula, not the wellbore.

## 4. Рпл carries no within-well drawdown signal

Within a well, `corr(rpl, rzab)` has median **0.28**, lower quartile **0.00**, minimum
−0.94.  Reservoir pressure here behaves like a periodically-restated map value, not a
daily measurement.  Any IPR driven off it inherits that — the Vogel inflow leg scores
**R² −2.395** against Рзаб.

Note this is a different failure from `pbubble_atm`, which this project established is a
*field* label.  `rpl` does vary (median 52 distinct values per well, within-well sd
33 atm) — it just does not vary *with* the thing it should.

## 5. Vogel is the right IPR on paper and wrong here

Reservoir pressure (median 174 atm) sits below the bubble point carried as a field label
(230-260 atm on Ya), so the reservoir is saturated and solution-gas drive — Vogel's own
regime.  It still fails: reaching the observed 174 → 51 atm drop needs `q/q_max ≈ 0.87`,
i.e. the well pinned near absolute open flow where the curve is steepest.  Halving the
rate from there swings predicted P_wf by ~67 atm against a measured within-well sd of
Рприем of **19 atm** — too stiff by roughly 3×.

## 6. Nodal analysis through the pump curve also fails

Static column at `vg_m` minus affinity-scaled catalog head has the merit of running on
`freq` (99.9 % present in the gap).  Cross-well it correlates **+0.20** with mean Рприем,
within-well |r| = 0.33.  Wellhead pressure and tubing friction are not in the mart, and
without them the level cannot be closed.

## 7. Collinearity trap in the surviving linear form

The annulus term `ρ·g·h/101325` moves only over ρ = 850-1010 kg/m³ — an **18.8 %** span —
so it is nearly collinear with the datum term.  Fitted uncentred, `a` and `h_eff` trade
off about 1:1, land at large offsetting values, and clipping either to its physical bound
destroys the balance the other was holding.  That bug scored **MAE 90.2 atm** on held-out
blocks against ~18 for the same form fitted centred on `RHO_REF = 930`.  The fix is
centring plus an intercept re-solve whenever a bound bites.

## 8. `gas_factor` is m³ per **tonne**, confirmed

`free_gas.py` flagged a suspected "m³/t vs m³/m³ convention … worth ~15 %".  The column
map settles it: **«Газовый фактор, м3/т»**.  Standing's `Rs` is m³/m³ of stock-tank oil,
so the column must be scaled by oil density (0.865 at the default API 32) before use.
`gas_at_intake.gor_m3_per_m3` does that conversion; it is applied there rather than inside
`free_gas` so existing callers of that module keep their documented behaviour.

## 9. Columns present in telemetry but absent from `proc__daily_merged`

The raw telemetry carries **«Давление буферное»** (wellhead, 24.4 % populated) and
**«Давление затрубное»** (casing/annulus, 22.5 %), neither of which reaches the daily
mart.  They are the missing inputs for a true nodal closure.  Measured on 43 729 rows
where both exist alongside Рприем, their marginal correlations with it are weak
(+0.20 and +0.08), so promoting them is not obviously worth it — but the option exists
and finding 6 is the reason it might matter.

---

## Validation design that follows from the above

Two regimes, scored separately, because a pooled number would average a nearly-solved
problem with a genuinely hard one:

* **warm** — the well has observed Рприем either side of the gap (15 of the 17 structural
  wells, and all short dropouts).  Scored by hiding a contiguous middle 40 % block.
  Scattered-day holdout is rejected: with a 1-day hole the neighbours pin the answer, so
  it measures interpolation while the real gaps run 30-1 442 days.
* **cold** — no observed Рприем at all.  Scored leave-whole-wells-out.

The **well-median baseline is not a strawman** — it is the bar.  A random row split is
also reported, purely to size the leakage it invents.

## Results (760 wells, 558 456 observed operating days)

| model | warm MAE / R² | cold MAE / R² | random split MAE / R² |
|---|---|---|---|
| **`cb_plain`** | **13.61 / +0.675** | **15.81 / +0.597** | 9.87 / +0.800 |
| `cb_physics` | 13.73 / +0.665 | 19.52 / +0.503 | 9.74 / +0.802 |
| `cb_residual` | 13.85 / +0.659 | 19.37 / +0.502 | 9.82 / +0.800 |
| `well_median` | 17.89 / +0.421 | 25.92 / −0.079 | 17.20 / +0.432 |
| `phys_linear` | 18.22 / +0.403 | 26.27 / −0.034 | 16.65 / +0.457 |
| `phys_vogel` | 29.35 / −0.289 | 45.59 / −1.445 | 26.63 / −0.100 |

**The random split overstates accuracy by 28 % of MAE** (9.87 vs 13.61) — that is what an
ungrouped fold would have invented.

**Physics-informing the ML is an honest null, and a liability cold.**  `plain` wins both
regimes; the hybrids are within 2 % warm and **23 % worse cold**.  Mechanism: for an
unseen well the physical model falls to field-pooled constants, and `residual` adds that
baseline back by construction while `physics` has learned to lean on `phys_pred_atm` —
its single most important feature (importance 21.8, ahead of `rpl` at 13.4).  The
*feature's meaning* shifts between fit and apply even though its availability does not.

**The physical model degenerates to the baseline.**  Across 757 calibrated wells the
inverse productivity index `b` has median **0.00** and `h_eff` rails to a bound in
**88.6 %** of wells (only 11.4 % inside 0-600 m).  Both terms switch off; the datum `a`
carries the prediction, which is why it scores level with the well median — *it has become
the well median*.  Do not quote `b` as a productivity index or `h_eff` as geometry.

It is still worth keeping as the second realization: the two arms disagree by a median of
**4.94 atm** per row, which is a usable uncertainty signal, and `phys_pred_atm` is the
ML's strongest feature.

## What this does to β

All **25 402** gap rows are imputed.  β error measured out-of-sample on the held-out
blocks:

| true Рприем band, atm | n | mean β | P MAE, atm | β MAE | dβ/dP per atm |
|---|---|---|---|---|---|
| (0, 20] | 3 793 | 0.704 | 16.64 | 0.094 | 0.0057 |
| (20, 40] | 69 689 | 0.655 | 11.32 | 0.062 | 0.0055 |
| (40, 60] | 68 007 | 0.584 | 8.95 | 0.041 | 0.0046 |
| (60, 100] | 46 675 | 0.404 | 15.59 | 0.054 | 0.0035 |
| (100, 400] | 25 644 | 0.147 | 28.79 | 0.058 | 0.0020 |

Sensitivity is **2.8× steeper below 20 atm than above 100** — the pressure error matters
most exactly where free gas is worst, so a pooled β-MAE would be the wrong summary.

**The ordering survives, which is what β is licensed for.**  Well-level Spearman between
β on true and β on reconstructed pressure is **0.980** over 613 wells, with the level
nearly unbiased (mean β 0.541 true vs 0.534 reconstructed).  Since `free_gas` admits β
only as a ranking variable — its level is an upper bound with no separation credit — that
is the relevant acceptance test, and it passes.

### ⚠ Imputed β is range-compressed — do not compute exposure shares on it

The reconstruction regresses to the mean (a boosted tree cannot leave the convex hull of
its leaf means, and MAE loss pulls to the centre): reconstructed pressure has **sd ratio
0.860** against observed, minimum 0.5 → 10.5 atm, p99 199 → 184.  β is monotone and convex
in pressure, so that compression maps into a monotone bias gradient — from **−0.091** in
the (0, 20] atm band to **+0.050** above 100 atm (column `beta_bias` above).

**β is pulled toward its middle: understated by ~0.09 in the gassiest wells, overstated by
~0.05 in the least gassy.**  Rank statistics survive (Spearman 0.980).  What does *not*
survive is anything reading β's spread or tails — in particular
`free_gas.free_gas_window`'s `frac_beta_above_*` exposure shares, which are threshold
counts in exactly the tails being shrunk.  Compute those on observed rows only.

The mandatory PVT sweep still passes on the reconstructed pressure: across all 27
assumption sets (API 28-36, T 60-100 °C, γg 0.70-0.90) the median β moves **5.0 %** and
Spearman against the default-PVT ranking never drops below **0.99988**.  So neither the
PVT assumptions nor the pressure reconstruction disturbs the ordering — the two error
sources are independently small in the dimension that matters.

⚠ Imputation does not upgrade β from a ranking variable to an absolute one.  Every caveat
in `free_gas.py` still stands: no separation credit, level is an upper bound, ordering
only.
