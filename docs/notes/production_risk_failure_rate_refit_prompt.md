# Production Risk Failure-Rate Refit & Timeline Extension — Handoff Prompt

You are working in `D:\GitHub\Pump2` (branch `data_storage_and_processing`) on the production-risk workflow for ESP pump failure forecasting. This continues the work described in `docs/notes/production_risk_failure_rate_handoff_prompt.md` — read that first for context on what was already shipped (failure-rate sheets, fleet denominator fix, Big-backed history replay).

## Diagnosis you are fixing (already established — do not re-derive)

A review of `results/production_risk_forecast/2026-07-13/tables/F_failure_rate_monthly.csv` over the fact window 2024-01..2026-04 found per-УН fact/predicted failure ratios:

| УН | fact | pred | ratio |
|---|---|---|---|
| ГЛОБАЛЬНО | 1090 | 986 | 1.11 |
| Верхнетирский | 338 | 300 | 1.13 |
| Ярактинский | 307 | 311 | 0.99 |
| **Мирнинский** | **99** | **52.6** | **1.88** |
| Кийский | 11 | 3.4 | 3.26 |
| Большетирский | 4 | 0.6 | 6.48 |

Root causes for the Мирнинский (Mc + Mr) 2× underprediction:

1. **`FIELD_PREFIX_MAP` gaps** (`backend/analysis/workflows/production_risk/config.py`): plan-fleet prefixes `MR` (20 wells), `BT` (35), `MSH` (14), `KI` (9), `NE`, `AM`, `YAY`, `ZYI` are absent → `crosswalk.map_model_field_from_well` returns `None` → wells silently resolve to `Global_Pooled`. Resolution is also inconsistent: in `G_big_install_validation.csv` 13/30 MR rows got `Mc_nonsour_*` (via Свод run matching) while 17 got `Global_Pooled`.
2. **Stale, under-powered Mc fit**: bundle `results/esp_survival_vba_models/2026-07-08/esp_models.csv` row `Mc_nonsour_Pooled` is `k1_degenerate` (w1=0, no early-failure mode), β=0.88, η=586, b50=386 op-days, CI 283–889 — fit on only 41 Свод failures. `WellsArtificialLiftBig.xlsx` holds **207 Mc+MR failure records** (156 MC + 51 MR) the fit never saw.
3. **Vintage regime shift**: censoring-aware Kaplan-Meier on Big Mc+MR runs (312 runs, 206 events, censor at 2026-06-30): median TTF installs ≤2023 = **479 cal days** (consistent with current model at ~0.74 uptime), installs 2024 = **247 d**, installs 2025+ = **236 d**. The fleet ramped 12→58 producing wells over the window, so it is dominated by the fast-failing new vintage. Empirical Mc+MR monthly failure rate = 0.0985 vs model ~0.05.

## Agreed scope (user decisions — follow exactly)

### Task 1 — Suppress minor/new-field chart output

In the `Интенсивность отказов` sheet (`failure_rate.py::write_sheets`), only render per-УН charts for **major fields**: ГЛОБАЛЬНО plus УН that pass a materiality threshold (suggest: ≥ 20 observed+predicted failures over the displayed window, or ≥ 10 average fleet size — pick one, make it a module constant with a comment). Minor/new УН (Юраченский, Потаповская, Северо-Марковский, Иктехский, Западно-Усть-Кутский, Аянская площадь, Большетирский, Кийский, Марковский, Верхненепский Северный…) must still appear in the long-form `Отказы_данные` audit sheet and the wide rate table — only their charts are suppressed. Global totals must not change.

### Task 2 — Fix prefix→stratum mapping

- Add `"MR": "Mc"` to `FIELD_PREFIX_MAP`.
- For the remaining unmapped prefixes (`BT`, `MSH`, `KI`, `NE`, `AM`, `YAY`, `ZYI`): investigate what fields they are (check `producer_meta.license_area` / plan УН per prefix, and whether Свод/Big has runs for them). Map them to an existing model field when there is a defensible physical match (e.g. `YAY`→`Ya` if it is a Yarakta pad); otherwise leave them on `Global_Pooled` fallback **explicitly** — add them to a documented `EXPLICIT_GLOBAL_FALLBACK` set so the fallback is a decision, not an accident.
- Add a coverage diagnostic: per УН, the share of predicted failures resolved via `Global_Pooled`. Export it in the `coverage` dict of `FailureRateResult` and print/log it in the run summary so silent fallback can never hide again.
- Make stratum resolution consistent: a given well must resolve through one path (prefix map first, Свод-match refinement only for sour/contractor within the same model field), not land in different strata depending on data path.

### Task 3 — Refit Mc stratum on Свод + Big with a recency window

The user chose refit over a calibration-factor patch. Requirements:

- Build the Mc(+MR) fitting dataset by merging Свод runs with Big intervals (dedupe by well + install date, tolerance ~7 days; Big fields: well=col0, Датамонтажа=7, Датаотказа=9, Датадемонтажа=10, ННО=11 — see `analysis/data/equipment_big.py` loader, use `nno_days` as the op-day clock where present, otherwise calendar×uptime_factor consistent with the existing `ttf_mix` clock convention).
- Respect right-censoring (running pumps censored at data cutoff 2026-06; pulled-without-failure censored at pull).
- **Recency window**: fit on runs installed ≥ 2023-01-01 (captures the new-vintage regime, ~150+ events). Also produce the all-history fit as a comparison row in the report, but the shipped registry row uses the recency window.
- Try the standard K=2 mixture under the existing constraint design (β₂>max(1,β₁), η₂/η₁≥2 — see `project_esp_survival_em` conventions / `backend/analysis/` EM code) — with the 25th percentile of failed Mc runs at ~76 cal days, an early mode is likely to fit now. Fall back to k1 if the mixture degenerates, same `model_kind` labeling as the existing registry.
- Write a **new bundle** `results/esp_survival_vba_models/<today>/` (copy the 2026-07-08 bundle, replace/add the refit rows: `Mc_nonsour_Pooled`, `Mc_nonsour_brt`, and — if MR stays mapped to Mc — no separate Mr row). Update `C.BUNDLE_DATE`. Do NOT edit the 2026-07-08 bundle in place.
- Sanity target: the refit Mc should imply a monthly rate near the empirical 0.09–0.10/well·month at typical fleet ages (expect b50 to land roughly 180–220 op-days). If the refit lands far from that, stop and report rather than shipping.
- Other strata: leave untouched in this pass unless the per-УН ratio table above shows them broken AND they gain material events from Big (check Вт/Я are fine: ratios 1.13/0.99).

### Task 4 — (deferred, do NOT do) vintage/calendar covariate in Cox

Explicitly out of scope this pass.

### Task 5 — Extend the historical timeline (full history OK), display window 2024+

The chart currently starts at 2024-01 because `months = plan.months` (ПП master columns). The user wants the model warmed up well before the display window so the 2024 starting level is meaningful.

- **Compute** the failure-rate series from early history — full history is acceptable; Big has solid global coverage from ≥2018 (failures/yr: 2021=233, 2022=417, 2023=639).
- **Display** window stays 2024-01..2027-12 in the charts (the extra history is for correct state at 2024-01, and may be written to the data sheet / CSV in full).
- **Denominator before 2024-01**: the plan master has no columns there. Use interval-based fleet counting: wells with an active ESP interval (install ≤ M ≤ end from Big/Свод) intersected with master wells. **Important**: to avoid a level step at the 2024-01 seam, either (a) use interval-based counting for the entire timeline, or (b) cross-validate interval-based vs plan-based counts on 2024-01..2026-04 and only keep the plan-based denominator if they agree within ~5%; document the choice in the sheet header.
- **Model line pre-plan**: `_segment_month_slices` gates on the plan active mask and plan op-days — for pre-plan months fall back to `nno_days` scaling (the existing `total_op` branch) or `uptime_factor × calendar days`. Also note the known asymmetry: observed failures count in any month but predictions are zeroed where the plan shows the well inactive — when extending, apply the same population rule to both lines.
- `FACT_COMPLETE_THROUGH = "2026-04"` is hardcoded in `failure_rate.py` — move it to `RunConfig` (CLI flag `--fact-through`, default derived or current value).
- Replace the bare `except Exception: return {}` in `_big_runs_by_well` with a logged warning + a `coverage` flag, so a Big load failure in the frozen EXE cannot silently revert the history line to Свод-only.

## Verification / acceptance

1. `cd backend && python -m pytest tests/ -v` — all pass (was 300); add/adjust tests for: prefix map additions, fallback-share diagnostic, chart suppression, extended-timeline denominator, refit registry loading.
2. Rerun the pipeline (dev is fine; rebuild EXE only at the end):
   ```powershell
   python scripts/run/production_risk.py `
     --pp-master "D:\Projects\Pumps\data\pp\ТМ-06_2026_2027_Р50_Базовый_Мастер файл.xlsx" `
     --gtm-schedule "D:\Projects\Pumps\data\pp\М08 2025 (СД 2026 - 2027)\ДФ_04.xlsx" `
     --prediction-workbook "D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_simple_prediction_2026-07-09_samples_ru_90d_compact.xlsm" `
     --techregime-workbook "D:\Projects\Pumps\data\tr\ТР_НЕФТЬ 30.06.2026.xlsm" `
     --equipment-big "D:\Projects\Pumps\data\big\WellsArtificialLiftBig.xlsx" `
     --forecast-start 2026-07-01 --horizon-end 2027-12-01 --full-tables
   ```
3. Recompute the per-УН fact/pred ratio table from the new `F_failure_rate_monthly.csv` (2024-01..2026-04) and report it in your summary. Acceptance: **Мирнинский ratio in 0.8–1.3** (was 1.88); ГЛОБАЛЬНО stays in 1.0–1.2; Верхнетирский/Ярактинский do not regress beyond ±0.1 of their current ratios.
4. Confirm the `Интенсивность отказов` sheet shows only major-field charts, the Мирнинский model line level is ~0.09–0.10/well·month in 2025 history, and no visible level step at the 2024-01 denominator seam.
5. Rebuild the frozen EXE (`powershell -ExecutionPolicy Bypass -File scripts\build_production_risk_exe.ps1`), rerun it, spot-check the same numbers in the packaged output, and verify the new bundle date is baked in (PyInstaller bundles `results/esp_survival_vba_models/` via resource root — confirm the new date folder ships).

## Files to touch

- `backend/analysis/workflows/production_risk/config.py` — prefix map, `EXPLICIT_GLOBAL_FALLBACK`, `BUNDLE_DATE`
- `backend/analysis/workflows/production_risk/failure_rate.py` — chart suppression, timeline extension, denominator, `FACT_COMPLETE_THROUGH`, Big-load warning
- `backend/analysis/workflows/production_risk/crosswalk.py` — consistent stratum resolution
- Refit script: new `scripts/run/refit_mc_stratum.py` (thin CLI) + logic in `backend/analysis/` (follow repo rule: heavy logic in `backend/analysis/`, outputs via `analysis.paths.results_dir("esp_survival_mc_refit")`)
- `backend/tests/test_production_risk.py`, `backend/tests/test_failure_rate_*`
- New bundle: `results/esp_survival_vba_models/<today>/esp_models.csv` (+ carry over `esp_cox_coeffs.csv`, `esp_run_covariates.csv`, `esp_mode_mix.csv` unchanged)

## Repo rules reminder

- All paths via `analysis.paths`; never hardcode `D:\Projects\...` in library code (CLI args as above are fine).
- Results via `results_dir("<workflow>_<step>_<variant>")`.
- sys.path bootstrap for scripts: `sys.path.insert(0, str(REPO_ROOT / "backend"))`.
- Write a completion summary including the before/after ratio table and the refit Weibull parameters (with CIs) to `docs/notes/` when done.
