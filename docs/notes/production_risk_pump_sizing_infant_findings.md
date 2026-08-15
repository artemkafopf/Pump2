# Pump sizing in the infant band — findings

**Date:** 2026-07-28 · **Slug:** `production_risk_pump_sizing_infant`
**Code:** `backend/analysis/workflows/production_risk/pump_sizing_infant.py`
**CLI:** `scripts/run/pump_sizing_infant.py` · **Companion to:**
[`production_risk_pump_sizing_findings.md`](production_risk_pump_sizing_findings.md)

---

## The question

The landmark analysis priced oversizing at ~101 d of lost residual RMST per doubling of
nameplate — but it reached that by excluding every run that failed before 90 operating
days, i.e. **50.8 % of all failures**. So it measured a cost *conditional on surviving
infancy*, leaving open: **does an oversized pump also die young?** If yes, the true cost
is higher than 101 d and the sizing rule needs re-pricing.

## Design

A fixed 90-op-day feature window is impossible for a run that fails at day 12, and any
window short enough would be contaminated by the failing pump's own collapsing `qliq`.
So telemetry features are dropped entirely in favour of what is known **at install**:

* `log_qnom` — chosen by an engineer before the run starts, so it cannot be contaminated
  by how the run ended (the same argument that made Qnom the preferred rate variable in
  `vt_v4`);
* **prior-run deliverability** — `kprod`/`rpl` over the well's last 90 operating days
  *strictly before* this run's install, leakage-free by construction.

Payoff: **4 139 runs / 2 136 events**, against 1 496 / 770 for the landmark — 2.8× the
events, and **no conditioning at all**.

Rather than defend a boundary, the size effect is allowed to vary over follow-up: a Cox
model in counting-process form with a heaviside `log_qnom` coefficient per interval.

---

## Result: an oversized pump does **not** die young

Within-well (`strata='well_key'`, so reservoir quality cancels), HR per e-fold of Qnom:

| band (calendar days) | events | HR [95 % CI] | p |
|---|---|---|---|
| **[0, 30)** | 372 | **1.13 [0.93, 1.37]** | **0.23** |
| [30, 90) | 361 | 1.38 [1.12, 1.69] | 0.0026 |
| [90, 365) | 762 | 1.66 [1.41, 1.95] | <1e-9 |
| [365, ∞) | 641 | 1.87 [1.49, 2.36] | <1e-7 |

Monotone, and the bands genuinely differ: χ² = 14.02, df = 3, **p = 0.0029**. A single
fleet-wide HR is the wrong summary.

**Replicates on Ya** (within-well 1.04 ns → 1.42 → 1.53 → 1.79) and points the same way on
Vt (1.38 ns → 1.78 → 1.86; the [365,∞) band is flagged uninterpretable, CI spans 1400×).

**Cross-design check:** the mature band's 1.87 sits alongside the landmark's within-well
2.01 — two different estimands on two different clocks agreeing where they overlap.

### Reading

**Oversizing is a duty/wear effect that accumulates with running time, not an
installation effect.** Infant mortality on this fleet is size-indifferent, consistent with
commissioning/installation defects — which is what an infant band usually consists of.

**Consequence for the landmark result:** the excluded half of the failures carries no size
signal, so **~101 d per doubling is not an underestimate**. The landmark's conditioning
turns out to be benign for this exposure. That was not knowable in advance, which is the
whole reason this analysis was worth running.

---

## The confound, caught in the act

Between-well (unstratified), the infant band shows a *significant* effect —
HR 1.25 [1.08, 1.45], p = 0.003 — which **vanishes within-well** (1.13, p = 0.23).

That gap is `corr(Qnom, kprod) = +0.64` surfacing as a spurious infant effect. Anyone
running this without well stratification concludes that oversizing kills young pumps. It
does not. Same lesson as the landmark analysis, opposite direction: there the confound
*hid* a real effect, here it *manufactures* one.

Unadjusted survival by size tercile separates from day 30 (S(30) = 0.927 small vs 0.881
large) — that is the confound, not causation.

## A bug worth remembering

The first run of this analysis reported a flat, null profile (1.02 / 1.04 / 1.12 / 1.15).
Cause: `penalizer=0.01` on the `CoxTimeVaryingFitter`. With 795 well strata and one
mostly-zero `log_qnom` column per band, even that tiny ridge crushes the estimates —
turning a clean monotone gradient into a non-result.

The fix is a **validity gate**, now run before any band estimate:
`_check_reproduces_plain_cox` collapses the per-band columns back into one and requires
the counting-process fit to reproduce a plain (stratified) Cox exactly. Measured:
0.284517 vs 0.284517 unstratified, 0.394190 vs 0.394190 stratified. Default penalizer is
now 0; a penalty is applied only as a convergence fallback and is **reported in the output
row** so a shrunk coefficient is never read as a null. Regression-tested.

## Non-findings, recorded

* **Failure-mode shift: does not replicate.** Node composition of infant failures varies
  with size pooled (χ² p = 0.018) but not on Ya (p = 0.25) or Vt (p = 0.90) — the pooled
  signal is field-mix. No mechanistic support either way.
* Mc is too thin throughout (11-21 events); its stratified fits are penalised to converge
  and are flagged, not reported.

## What remains open

* **Why** the effect accumulates with running time is still unidentified. Free gas at
  intake was tested in the companion analysis and came back an honest null (0.4 %
  attenuation), so the mechanism is neither installation quality nor intake gas.
  Candidates not yet tested: cumulative thrust/abrasive wear, motor thermal duty.
* The economic close still needs the oil-price/workover-cost side to become a sizing rule.
  Both halves of the life side are now measured.
