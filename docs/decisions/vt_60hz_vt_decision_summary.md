# Vt 60 Hz Decision Summary

## Executive Answer

**Short answer:** `60 Hz` should **not** be treated as a default safe operating mode for ESPs in `Vt` if the goal is to maximize pump life.

The evidence does **not** support a blanket statement that `60 Hz` is universally bad across the whole portfolio. But for `Vt`, the balance of evidence points in one direction: pumps run at `>58 Hz` tend to fail **earlier in calendar time**, and the early-failure burden is higher. The subgroup is still small, so the exact size of the penalty is uncertain, but the direction is consistent enough that `60 Hz` in `Vt` should be treated as a **caution / exception mode**, not a normal default mode.

## Direct Answer To The Operating Question

### Is it safe to run ESP at 60 Hz in Vt?

Not as a default life-maximizing setting.

Operationally, the best current recommendation is:

- `Do not ban 60 Hz outright portfolio-wide.`
- `Do not assume 60 Hz is safe-by-default in Vt.`
- `Use 60 Hz in Vt only when there is a clear production reason, and pair it with tighter monitoring.`

## Will 60 Hz shorten TTF in Vt, and by how much?

The evidence says **probably yes in calendar time**, but the exact size depends on how TTF is defined and how censoring is handled.

Best practical range from this study:

- Using the raw best-available duration summary, the median in `Vt` moves from `141.5` d at `50–55 Hz` to `111.5` d at `>58 Hz`: about **30 days shorter** (21%).
- Using the `TTF_true` sensitivity pass, the median moves from `151 d` to `117 d`: about **34 days shorter** (23%).
- The adjusted hazard ratio inside `Vt` is in the **1.34–1.54x** range depending on the time axis, but the Vt-only interval is still wide because the sample is small.

So the responsible answer is:

- `Expected effect in Vt:` likely shorter calendar TTF
- `Best direct-median estimate:` roughly **30 to 35 days shorter**
- `Model-based reading:` higher failure hazard in the **1.34–1.54x** range, so the effective life penalty may be larger in some censoring-adjusted views
- `Confidence in exact magnitude:` moderate-to-low
- `Confidence in direction:` moderate

## Does 60 Hz increase early-failure risk in Vt?

Yes, directionally it does.

- At `30` days, early-failure share among Vt failures is `0.17` at `50–55 Hz` vs `0.21` at `>58 Hz`.
- At `60` days, it is `0.23` vs `0.29`.
- At `90` days, it is `0.34` vs `0.57`.

The `90`-day gap in `Vt` is the most decision-relevant one: **0.23 higher early-failure share** at `>58 Hz`. The same pattern also appears globally, where the `90`-day gap is `0.11` and statistically clearer.

## Is the 60 Hz signal in Vt just because those wells were harsher?

The available evidence says **no, not mainly**.

- Effective `H2S` in `Vt >58 Hz` wells is actually **lower**, not higher: median `0.083` vs `0.518` in the rest of `Vt`.
- `GLF` is also not higher in the `>58 Hz` Vt subgroup: median `289.6` vs `326.3`.

So the worse `Vt >58 Hz` outcome cannot be dismissed as a simple 'those were just the harshest wells' story based on the measured harshness proxies we have.

## What equipment is most likely to become problematic at 60 Hz in Vt?

The first risk signal is **working-organ / clogging-related failure**, and the second is the **motor / electrical package**.

- In the `Vt >58 Hz` failure mix, `Засорение РО` is the largest single category: `35.3%` of failures at `>58 Hz` versus `13.1%` in the rest of `Vt`.
- The next important risk cluster is `ПЭД (R-0)`: `29.4%` at `>58 Hz` versus `20.3%` in the rest of `Vt`.

When we separate early failures from mature failures, the pattern becomes clearer:

- Earlier `Vt` failures are relatively more concentrated in `Засорение РО` and some fast mechanical/electrical events.
- More mature `Vt` failures lean relatively more toward `ПЭД (R-0)`, `НКТ`, and seal/protection-related categories.

**Practical interpretation:** if `Vt` wells are pushed to `60 Hz`, the first equipment family to watch is the **working organs / clogging path**. The second family to watch is the **motor / electrical package (ПЭД)**.

## Is there evidence of a fixed cumulative-work capacity (TRF budget)?

Only partially.

- On the `TTF_true` axis, the global `>58 Hz` penalty is stronger: adjusted HR `1.31`.
- On the `TRF` axis, the same penalty shrinks materially: adjusted HR `1.06` globally and `1.17` in `Vt`.

This suggests the pump may **not** simply be wearing out after less total accumulated frequency work. Instead, at higher frequency it appears to get through that usable work capacity faster, so it fails sooner in calendar time.

But the data do **not** support a single clean universal `TRF` failure threshold, either globally or in `Vt`. If a cumulative-duty budget exists, it is more likely **failure-mechanism-specific** than universal.

## Stronger Non-Frequency Driver

`H2S` remains the stronger and more stable risk signal than `60 Hz` itself. In the adjusted Vt model, high `H2S` has an estimated hazard ratio of `2.42`.

That means the most conservative operational rule is not 'frequency only'. It is:

- `avoid 60 Hz especially when H2S is high`
- `avoid 60 Hz especially when failure history already points to clogging / working-organ problems`

## Final Recommendation

### Recommended operating policy for Vt

- `Default mode:` stay with the normal `50–55 Hz` regime when life maximization matters.
- `60 Hz mode:` use only when justified by production need, and document it as an exception mode.
- `Monitoring priority at 60 Hz:` watch for working-organ clogging / solids / scale symptoms first, then motor / electrical package stress.
- `Highest-risk combination:` `60 Hz` plus elevated `H2S`.

### One-sentence decision answer

**Running ESPs at `60 Hz` in `Vt` is not the safest default choice for TTF. The most likely effect is shorter calendar life, with direct-median estimates around `30–35 days` shorter, and the main additional risk appearing first in working-organ / clogging failures and then in the motor (`ПЭД`) package.**

## Key Figures

### Vt TTF_true survival

![Vt TTF_true survival](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/ttf_true_days_vt_km.png)

### Early-failure share at 30 / 60 / 90 days

![Threshold sensitivity](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/true60_threshold_early_share_ttf_true.png)

### Are >58 Hz Vt wells simply harsher?

![Vt environment check](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/vt_true60_environment_check.png)

### Which Vt failures become more common at >58 Hz?

![Vt failure mix](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/vt_true60_failure_mix.png)
