# Vt High-Frequency (>55 Hz Majority Days) Decision Summary

## Executive Answer

**Short answer:** under the new definition, `Vt` high-frequency wells do **not** show a clear TTF penalty versus the `50-55 Hz` baseline.

This is a very different answer from the earlier `mean >58 Hz` proxy. The reason is that the new rule creates a much broader high-frequency population. That broader group does not behave like the old near-60-Hz subgroup.

## Definition Used

- High-frequency well = `days(freq > 55 Hz) / duration_best_days > 0.5`.
- Baseline = `50 < freq_w_mean <= 55 Hz`.

## What Happens To The Population?

- In `Vt`, the old `mean >58 Hz` proxy had `24` runs / `17` failures.
- The new high-frequency definition has `57` runs / `42` failures.
- Globally, the old proxy had `176` runs / `115` failures, while the new definition has `328` runs / `219` failures.

So yes, this redefinition changes the study population a lot.

## Does The New High-Frequency Group In Vt Have Shorter TTF?

Not in this rerun.

- Best-available median in `Vt`: `141.5` d at baseline versus `162.0` d in the new high-frequency group.
- `TTF_true` median in `Vt`: `151.0` d at baseline versus `171.0` d in the new high-frequency group.
- Adjusted Vt hazard ratio: `0.96` on `TTF_true` and `0.98` on best-available duration.

So under this definition, the high-frequency Vt subset is **not** failing sooner than the baseline subset.

## Does It Increase Early-Failure Risk?

Not clearly in `Vt`.

- At `90` days, the Vt early-failure share difference is `-0.07`.
- Globally, the same `90`-day difference is `0.04`.

That means the global high-frequency burden is still slightly less favorable, but the Vt-specific signal is weak under this broader definition.

## Is The New HF Group Simply Harsher?

Not on the measured chemistry proxies we checked.

- Vt median `H2S`: `0.317` in the high-frequency group versus `0.408` in the rest of `Vt`.
- Vt median `GLF`: `321.5` versus `324.3`.

## What Is Most Likely To Become Problematic?

Under the new definition, the Vt failure pattern is less concentrated than in the old near-60-Hz proxy, but two areas still deserve attention:

- `Слом вала` is the most overrepresented category in the new high-frequency Vt group: `23.8%` versus `14.6%`.
- `Засорение РО` is the next strongest difference: `21.4%` versus `13.3%`.

## TRF Interpretation

- The Vt adjusted hazard ratio on the TRF axis is `0.89`.
- The global `TTF_true` hazard ratio is `1.11`.

This supports a practical conclusion: the broader majority-days-above-55-Hz definition is not isolating the same high-risk operating mode as the old near-60-Hz proxy.

## Final Decision Interpretation

- If management wants to know whether **majority-of-days above 55 Hz** is unsafe in `Vt`, this rerun does **not** show a clear TTF penalty.
- If management wants to know whether **near-60-Hz mode** is unsafe in `Vt`, the older narrower proxy remains more aligned with that question and gives a more cautionary answer.

## Key Figures

### Vt KM on TTF_true

![Vt HF share55 TTF_true](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_ttf_true_days_vt_km.png)

### Early-failure share at 30 / 60 / 90 days

![HF share55 thresholds](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_thresholds.png)

### Vt environment check

![HF share55 Vt environment](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_vt_environment.png)

### Vt failure mix

![HF share55 Vt failure mix](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_vt_failure_mix.png)
