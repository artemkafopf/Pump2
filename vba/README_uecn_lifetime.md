# UECN lifetime VBA module

The module [vba/mdlUecnLifetime.bas](vba/mdlUecnLifetime.bas) implements the requested ready-to-use formulas for UECN lifetime metrics from pre-fitted Weibull parameters.

## Public functions

- UECN_Median(debit, frequency, kpod, h2sClass, contractor, Optional tau)
- UECN_Mean(debit, frequency, kpod, h2sClass, contractor, Optional tau)
- UECN_RMST(debit, frequency, kpod, h2sClass, contractor, Optional tau)
- UECN_ResidualResource(debit, frequency, kpod, h2sClass, contractor, age, Optional tau)
- UECN_Metrics(...)

## Notes

- The implementation uses the requested pattern selection:
  - all-three when the row is non-sour and debit, frequency and kpod are available
  - only-debit otherwise, including acidic rows
- Missing covariates are dropped from the log-linear term exactly as requested.
- The module clamps debit to the stated applicability bounds 47..823.
- The default horizon is 730 days.

## Example checks

The formulas reproduce the three provided checks approximately:

- non-sour / all-three / debit=400 / frequency=52 / kpod=0.90 / slb -> median ≈ 247, mean ≈ 304, RMST(730) ≈ 292
- non-sour / only-debit / debit=400 / slb -> median ≈ 285, mean ≈ 373, RMST(730) ≈ 335
- acidic / slb -> median ≈ 120, mean ≈ 143, RMST(730) ≈ 143
