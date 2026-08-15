# v5.3 — gas covariates on 3-month telemetry windows (summary)

Date: 2026-08-06. **Uncommitted, nothing shipped, no calculator or VBA touched.**

Full log, every table, the figure and the runnable code:
`results/production_risk_field_v53/2026-08-06/` (start at `FINDINGS.md`).
Predecessor: `docs/notes/v52_model_and_layer_procedure.md`.

**No constraints were imposed anywhere.** Free piecewise-linear polylines in log θ, no
monotone/tent shapes, no anchors, no rational deployment forms, no operator scenarios. The
only fixed quantity is the normalisation point, which divides a curve by a constant and
changes no relative risk.

## The one-paragraph version

Widening the counting-process window from 30 to 90 days and adding a gas block to v5.2 turns
up **one new layer that passes every gate — θ(GLF / номинал газосепаратора)** — and
**disqualifies all three pressure quantities as design covariates**. The separator nameplate
covers **90.6 % of runs** (`Сепараторы mapped.xlsx` → `raw__equipment_big.gassep_type`), and it
is not pump size restated: corr(log Qг_ном, log Qном) = 0.831, ratio p10 1.12 / p90 5.00.

## Verdicts on what was asked

| covariate | verdict |
|---|---|
| **GLF / Qг_ном** | **layer.** ΔCV +12.01 (5 seeds, 5/5) at lag 3 mo, **rising** to +12.28 at 6 mo. θ span 1.70×, worth +74 d ННО on Ya to +106 d on Za |
| GLF raw | real but weaker (+8.05) and **lean-field concentrated** — only +2.28 on the gassy strata |
| **P_приём / P_нас** | **nowcast, not a cause.** +19.01 at lag 0 → +5.74 at 6 mo. Within 0.7 CV of raw P_приём at every lag ⇒ the ratio adds nothing |
| **P_заб / P_нас** | **worse at every lag, dead at 6 mo** (+0.91, 1 seed negative). `rzab` is computed from `rpump_intake` — not an independent measurement |
| β free gas at intake | Ya-only; its gassy-fitted shape scores **−0.06 on gassy**. Not shippable |
| GOR, Qг/Qг_ном, Qг_ном alone | nulls (+1.00 / −0.30 / +0.40) |

## The five things worth not re-deriving

1. **`pbubble_atm` is a field label** — within-field CV 0.002 (Ic) to 0.174 (Mc), 5 distinct
   values across 386 Vt wells. Measured consequence: **corr(P_приём, P_приём/P_нас) = 0.98
   within stratum.** Dividing by it rescales fields and does nothing within one. Always fit
   the raw pressure alongside any P/P_нас ratio, or the ratio's claim is unmeasurable.
2. **The lag profile is the verdict, not the CV level.** Pressures decay ~70 % from lag 0 to
   6 months; GLF/Qг_ном *rises*. `θ_P_приём` is monotone **increasing** (0.742 at 25 atm →
   1.292 at 163) — backwards for gas, forwards for a pump that has stopped drawing the level
   down because it is already dying.
3. **The ratio beats both its ingredients AND both of them as two free arms** (+12.01 vs
   +8.49 on fewer parameters). Two arms are additive in log θ; the ratio is a function of
   their difference. ⇒ it is the **separator's gas loading** that matters, not gas rate and
   not separator size. This is why the previously-null GLF now scores.
4. **v4's "GLF is water cut wearing a ratio" does not hold here.** `GLF = GOR×(1−ХВ)×ρ`, GOR
   is a null, and the ХВ arm is a polyline in raw percent whose last knot is 95 — it cannot
   express 95→99.9 % (8.3 % of intervals are above 95 %, 4.2 % above 99 %). Refitted with
   knots to 99.5 and again as `log(1−ХВ)`: neither improves ХВ's own CV, and **neither
   absorbs any of GLF/Qг_ном** (+12.19 / +11.89).
5. ⚠ **A defect in v5.2's own frame.** Its builder seeded pre-telemetry windows with the
   run-level water cut **as a fraction** while converting telemetry to percent — 945 rows
   (2.7 %) across 96 runs entered at 0.1–0.4 % ХВ, carrying exposure and **zero events**.
   Corrected, **θ_ХВ(0) = 0.956, not 0.768**: v5.2's *"low water cut is protective"* does not
   survive. With the gas layer in, the whole ХВ layer shrinks to span **1.18×** against the
   1.38× v5.2 reported.

## Ya as the donor — it works, but it understates

Ya is the **second least gassy** field (median GLF 68 vs Vt_sour 223, Mc 194, Za 179) while
carrying 52 % of the events. Transfer test (fit the shape on one group, score held-out on the
other): the LEAN-fitted curve is worth **+5.57 on the gassy strata** against +3.21 on its own
donors, and the GASSY-fitted curve is worth +5.22 on lean. **The shape transfers both ways;
the depth does not** — θ at the top knot is 0.720 lean-fitted, 0.657 gassy-fitted, 0.588
pooled, and the G6 fact panel's top bin is 0.574 [0.412, 0.779] gassy against 0.784
[0.517, 1.141] lean. Use Ya for the shape; expect a fleet-shared curve to understate Vt, Mc,
Az and Za.

## Not established

The **mechanism**. More gas per unit of separator capacity comes out *protective*, which is
the opposite of the naive gas-damage story. Fit, lag profile, fact panel and transfer test all
agree on the sign, so the association is solid — but candidates (gas-lift assistance to the
lift; selection, i.e. a big separator is fitted precisely where gas is known to be trouble, so
Qг_ном encodes an operator's risk assessment; reservoir quality) are untested. Treat it as a
**predictive** layer until one is separated. Ship it disabled until G6 and G7 confirm its
magnitude on the workbook side, per the v5.2 procedure.
