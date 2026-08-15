# Workstream E - Mc Variant Validation Findings

Date: 2026-07-15  
Runner: `scripts/run/production_risk_mc_variant_validation.py`  
Outputs: `results/production_risk_mc_variant_validation/2026-07-15/`

## Purpose

Workstream D showed that A + placement map + gated idle hazard fixes Ya/Vt/global
but leaves Мирнинский high:

- Mc observed failures: 41
- Mc predicted failures: 31.99
- Mc fact/model: 1.282

E tests whether Mc is a baseline/vintage problem by swapping only the
`Mc_nonsour_Pooled` survival row while keeping the D replay setup unchanged:
placement map + gated idle hazard, manual calibration disabled, no shipping bundle.

## Result

| Mc window | Mc observed | Mc predicted | Mc fact/model | Mc gate | Global fact/model | A/KM note |
|---|---:|---:|---:|---|---:|---|
| all_vintage | 41 | 31.99 | 1.282 | fail | 0.975 | passes A/KM |
| install_2023plus | 41 | 41.27 | 0.994 | pass | 0.962 | fails strict `dS < 0.05` |
| install_2024plus | 41 | 37.48 | 1.094 | pass | 0.967 | fails strict `dS < 0.05` |

Both recent Mc baselines land inside the D fact/model band. The 2023+ variant is
best centered on the 2024-01..2026-06 fact window; 2024+ is also acceptable but
near the upper edge.

## Interpretation

This strongly supports D's diagnosis: the remaining Mc miss is a Mc baseline/vintage
choice, not the time map, idle exposure, or manual calibration removal.

However, this is not yet a production promotion decision. The recent Mc variants
failed the strict A6 KM acceptance threshold (`accept_dS_lt_0p05 = False`), although
their B50 remains inside the KM median confidence interval. E therefore creates a
clear model-governance tradeoff:

- all-vintage is KM-clean but under-predicts deployed Mc failures;
- recent-vintage matches the production fact window but needs an explicit acceptance
  rationale or an updated Mc-specific KM gate.

## Recommendation

Decision update, 2026-07-15: `install_2023plus` is accepted for the Mc production
baseline. The reason is explicit: D/E fact/model acceptance on the deployed 2024-2026
window is allowed to override the strict per-window `dS < 0.05` screen for Mc because
the all-vintage curve is KM-clean but demonstrably too long for the current deployed
Mc population.

Implemented follow-up:

1. Standard candidate bundle created:
   `results/esp_survival_vba_models/2026-07-15-mc2023plus/`.
2. Default production bundle date changed to `2026-07-15-mc2023plus`.
3. Legacy manual calibration factors retired in `failure_rate.py`.
4. Bundle-level `esp_observed_failures.csv` added so the production chart uses the same
   A-population observed numerator as D validation.
5. Normal production CLI verified with `python scripts/run/production_risk.py --no-excel --full-tables`.

Production-path fact/model over 2024-01..2026-06:

| field | observed | predicted | fact/model |
|---|---:|---:|---:|
| Мирнинский УН | 41 | 41.27 | 0.99 |
| Ярактинский УН | 212 | 223.65 | 0.95 |
| Верхнетирский УН | 223 | 218.52 | 1.02 |
| ГЛОБАЛЬНО | 680 | 707.00 | 0.96 |

The accepted line passes the D1 0.90-1.10 gate in the normal production path.
