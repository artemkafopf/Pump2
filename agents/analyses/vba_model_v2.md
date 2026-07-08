# VBA Prediction System v2 — Align the Workbook with Phases A–D

## Purpose

Bring the Excel VBA prediction system (same input file:
`D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsx` saved as `.xlsm`,
modules in `D:\GitHub\Pump2\vba\`) in line with everything Phases A–D established.

The currently deployed system is **stale on three axes**, one of them silently biasing
every prediction:

| # | What the workbook does today | What Phases A–D established |
|---|---|---|
| 1 | `ESP_Models` carries the 2026-07-01 Excel-era parameters — **failures-only population, calendar clock** (e.g. Ya_nonsour_Pooled w1=0.155, η₂=540d, n=702) | Failures-only fits understate B50 ~2×; calendar clock inflates it 15–53%. Current standard: **full population (2,634 runs) + `ttf_mix` operating clock**, post-label-hygiene (Ic gained 7 runs), with bootstrap CIs and ΔAIC-verified K=1/K=2 choice per stratum |
| 2 | `mdlCoxHR` + `ESP_*_Cox` UDFs apply a 3-covariate θ (delta_bep, p_bot, n_stages_ratio) from the pre-Phase-C 5e fit | That θ was superseded by the merged 18-term θ — **which failed the out-of-sample gate** (C-index ≈ chance, both cutoffs; Phase D's dynamic layer failed the same gate). Program verdict: **no covariate θ ships**; strata + baselines are the forecasting engine |
| 3 | No failure-mode outputs | Phase B: electro-thermal is 33–55% of incidence by stratum; per-mode CIF quantiles exist for repair planning (`b5_incidence_quantiles`), with a frozen node→mode map as single source of truth |

Read first:

```
results/model_report/2026-07-07/README.md          — the consolidated model report (esp. §1, §3.8, §8)
agents/analyses/cox_hr_vba_integration.md           — current VBA architecture, UDFs, sheet layout
vba/README.md                                       — setup, import procedure, column map
C:\Users\alexe\.claude\projects\d--GitHub-Pump2\memory\feedback_vba_encoding.md — VBA bug patterns
CLAUDE.md                                           — repo rules
```

---

## §0 — CONFIRM WITH USER BEFORE IMPLEMENTING

### 0.1 Clock policy — the decision that matters most

The new baselines live on **operating time**; the workbook's age column (I) is **calendar
days since install**. Feeding calendar age into an operating-time model overstates a
pump's age (biases RUL down, S(t) down); reporting operating-day RUL as if it were
calendar days biases planning dates the other way. Options:

- **(a) Uptime-factor conversion (recommended).** Per-stratum uptime factor
  `u_s = mean(ttf_mix / run_days)` computed from the mart (typical range ≈ 0.6–0.9).
  Input: `age_op = age_cal × u_s`; output: `RUL_cal = RUL_op / u_s`. Factors live in a new
  registry column, are shown in an audit column (`ESP_Uptime`), and are overridable per
  row if the user knows a well's actual uptime.
- (b) Document-only: state that all ages/RULs are operating days and leave conversion to
  the user. Honest but guarantees silent misuse.
- (c) Keep calendar-clock models in the workbook. Rejected by default — it re-imports the
  bias Phase A removed — but listed for completeness.

**Confirm (a), or choose.** Whatever the choice, every output column header must carry the
clock (`_op_d` / `_cal_d` suffix).

### 0.2 θ layer decommission — deprecate or delete?

- **(recommended) Deprecate-keep:** `ESP_Theta` and `ESP_*_Cox` remain callable but return
  the baseline result with θ≡1 unless an `ESP_CoxCoeffs` sheet exists AND carries an
  explicit `enabled=TRUE` flag cell; `RunPredictions` drops the Cox columns (CI–CK); the
  old coefficient sheet is deleted. Rationale: existing spreadsheets referencing the UDFs
  don't #NAME?, and Phase E can re-enable via the same switch if a future θ passes the
  out-of-sample gate.
- Or delete outright (cleaner project, breaks any sheet that references them).

### 0.3 Registry representation for single-Weibull strata

ΔAIC prefers K=1 for Mc and Da; several strata are degenerate on K=2 (TTF-mix:
Mc_nonsour_brt, Ya_nonsour_oth). Representation options for the `ESP_Models` sheet:

- **(recommended)** store as `w1=0` with component 2 = the single Weibull (β, η from the
  TTF-mix Phase 1 fit) and a `model_kind` column (`k2` / `k1_aic` / `k1_degenerate`) so
  the existing mixture math works unchanged and the audit trail is explicit;
- or keep K=2 numbers and only flag — rejected: Phase A showed those numbers are
  unreliable precisely there.

### 0.4 Scope of the new planning outputs

Mode-mix planning layer (Phase B): include now or defer? Recommended: include the sheet +
two UDFs (cheap, read-only; see T4). Bootstrap CI display: recommended as two extra
columns on B50 outputs (`ESP_B50_lo/hi`); confirm they won't clutter the operational view.

### 0.5 Sour caution outside Vt

Az/Za/Ya flagged-sour runs fail materially faster in raw KM but the adjusted effect was
borderline → strata stay Vt-only (Phase B decision). Optional display-only flag
(`ESP_SourCaution` = TRUE when field ∉ Vt and the well's H₂S proxy > 10 mg/l) — requires
an H₂S column in the workbook; confirm whether one exists (BW is *class*, not
concentration) or skip.

---

## Part 1 — Python side: one export, one source of truth

Extend `scripts/run/export_model_csv_for_vba.py` to emit a **single versioned bundle**
under `results_dir("esp_survival_vba_models")`:

1. **`esp_models.csv`** — one row per registry stratum:
   `stratum | model_kind | w1 | beta1 | eta1 | beta2 | eta2 | b10 | b50 | b90 | b50_lo | b50_hi | uptime_factor | n_runs | n_failures | pct_mixed_clock | clock | fit_date`
   - **Regenerate the fits first** (do not consume the 2026-07-03 CSVs: label hygiene
     moved 7 runs into Ic, and the contractor-level CSVs predate the ΔAIC columns):
     field-level + contractor-level Phase 1/Phase 2 on `ttf_mix`, full population, current
     code. Pooled rows = field-level fits; contractor rows where ≥20 failures; Global_Pooled
     fallback recomputed on the full fleet.
   - `model_kind` per §0.3; B50 CIs from the Phase A bootstrap machinery
     (`models/survival/bootstrap_ci.py`, wells resampled — rerun on the regenerated fits);
     `uptime_factor` per §0.1 from the mart.
2. **`esp_mode_mix.csv`** — per field-level stratum × mode group: incidence share at
   90/365d and time-to-10%/25% incidence with CIs (regenerate via the Phase B B5/B2
   machinery, post-hygiene), plus the frozen node→mode map rows (from
   `analysis/data/failure_modes.py` export) for the audit sheet.
3. A `bundle_manifest.txt` — row counts, git commit, clock, date — imported into a hidden
   `ESP_Version` sheet so any workbook can state exactly which model build it runs.

Python acceptance: a pytest that reads the bundle back and asserts schema, `clock ==
"ttf_mix"` everywhere, every stratum has a usable row (no NaN in the params the VBA math
touches), and B50 recomputed from (w1,β,η)-params by bisection matches the `b50` column
within 1%.

## Part 2 — VBA side

### T1. Registry v2 (`mdlModelRegistry.bas`, `mdlBatchProcess.ImportModelCSV`)

- Extend the `ESP_Models` sheet schema with the new columns (`model_kind`, `b50_lo/hi`,
  `uptime_factor`, `pct_mixed_clock`, `clock`); `LoadModelRegistry` reads them; keep the
  3-priority cascade unchanged (exact → `{field}_{h2s}_Pooled` → `Global_Pooled`).
- `ImportModelCSV` v2: header-driven (no positional column assumptions), validates
  `clock="ttf_mix"` on every row and **refuses the import with a written cell message**
  (never `MsgBox` — see rules) if a calendar-clock or schema-mismatched file is offered.
- `CreateModelSheet` (the hardcoded bootstrap sheet) is regenerated from the new bundle —
  a small Python helper should *generate the VBA literal block* from `esp_models.csv` so
  the hardcoded fallback can never drift from the CSV (write it to
  `results_dir(...)/vba/CreateModelSheet.generated.bas` and paste-import).

### T2. Clock layer (per §0.1 decision)

- New registry-backed function `UptimeFactor(stratumKey)`; conversions applied inside the
  UDF layer, not the math layer: `ESP_RUL`/`ESP_SF`/`ESP_Hazard`/`ESP_Predict` convert the
  incoming age, and RUL-type outputs convert back to calendar for the planning columns.
- New audit UDF `ESP_Uptime(field, h2s, ctr)`; `RunPredictions` writes it next to
  `ESP_Stratum`.
- Optional per-row override column (user-supplied uptime), taking precedence when present.
- Every output header renamed with the clock suffix; the old headers must not survive the
  upgrade (silent unit confusion is the failure mode this task exists to kill).

### T3. θ decommission (per §0.2 decision)

Implement the chosen option. Either way: remove Cox columns from `RunPredictions`; delete
the stale `ESP_CoxCoeffs` values; leave a comment block in `mdlCoxHR.bas` pointing at
`results/model_report/2026-07-07/README.md` §3.8 (why θ is shelved) and at
`theta_final_coeffs.csv` (what would be re-enabled if a future θ passes the OOS gate —
the η-rescaling math in `ApplyCoxToEta` stays correct and tested for that day).

### T4. Mode-mix planning layer (per §0.4)

- New sheet `ESP_ModeMix` imported from `esp_mode_mix.csv` (+ the node→mode map on an
  audit sheet — the VBA must never re-hardcode node lists).
- Two UDFs:
  `ESP_ModeShare(field, h2s, ctr, mode, horizon_days)` — share of incidence by that mode;
  `ESP_ModeIncidenceTime(field, h2s, ctr, mode, pct)` — operating days to pct% cumulative
  incidence (CIF-based, i.e. honest absolute risk under competing modes).
- One optional `RunPredictions` column: dominant expected failure mode at current age
  (max CIF increment over the next 90 operating days) — planning input for spares/crew.

### T5. Risk flags (display-only — never multipliers)

- `ESP_RunSeq(wellRef)` — count of prior rows for the same well in `Свод` (well name
  normalized the same way as the Python `normalize_well_key`: trim + casefold);
  `RunPredictions` writes `run_seq` and a caution mark when ≥3. Rationale: `log_run_seq`
  was the largest θ coefficient — as a *flag* it is defensible; as a multiplier it failed
  the OOS gate with everything else.
- `ESP_IsDegenerate` extended to read `model_kind` (reports `k1_aic` / `k1_degenerate`
  distinctly).
- §0.5 sour-caution flag if confirmed.

### T6. Validation harness (`mdlValidation.bas`, new)

A `ValidateRegistry` macro writing a pass/fail table to a scratch sheet (again: cell
output, not MsgBox):

1. Registry row count and stratum coverage vs `ESP_Version` manifest.
2. For every stratum: B50 recomputed by VBA bisection vs the imported `b50` column,
   tolerance 1% — this catches math/parameter drift in one shot.
3. θ≡1 equivalence: `ESP_B50_Cox(...) = ESP_B50(...)` exactly (if deprecate-keep chosen).
4. Clock spot-checks with known answers from the regenerated fits, e.g.
   `ESP_B50("Ya","nonsour","brt")` ≈ the regenerated Ya_nonsour_brt B50 (of order 313 op-d
   pre-regeneration — assert against the imported value, not a hardcoded number),
   `ESP_B50("Vt","sour","slb")` ≈ its imported value; `ESP_Stratum("Bt",…) = "Global_Pooled"`.
5. A deliberate failure probe: import refusal on a doctored CSV with `clock=run_days`.

Acceptance for the whole phase: `ValidateRegistry` all-green in the target workbook, plus
a screenshot/paste of the validation sheet in the results write-up.

---

## Rules (hard-earned — violations have each cost a session before)

- **`ChrW()` not `Chr()`** for any code point > 255; source `.bas` files contain no
  Cyrillic literals (sheet/name matching via `ChrW` sequences as done today).
- `mLoaded`, `mCount`, `mModels` are **Private to `mdlModelRegistry`** — other modules go
  through `EnsureRegistryLoaded()` / public accessors only.
- Never a name that collides case-insensitively with a function (`h2sClass` vs
  `H2SClass()` — VBA is case-insensitive).
- **Never `MsgBox` in any code path reachable from a UDF** (→ `#VALUE!`); errors surface
  as return strings / validation-sheet rows.
- Module re-import procedure: remove the old module, save, then import — stale-module
  ghosts otherwise.
- Keep `Option Explicit` everywhere; no cross-module calls to Private functions.
- Repo side: paths via `analysis.paths`; the export script and its pytest live with the
  other `scripts/run/` + `backend/tests/` conventions; nothing hardcodes
  `D:\Projects\...` in Python.
- The workbook is production for planning staff: **back up the `.xlsm` before module
  surgery**, and stage the upgrade in a copy first.

## Definition of done

- [ ] §0.1–0.5 decisions recorded at the top of the results write-up
- [ ] Bundle exported (models + mode-mix + manifest), pytest green, fits regenerated
      post-hygiene on `ttf_mix` full population with ΔAIC/`model_kind` and B50 CIs
- [ ] Workbook registry imported; headers carry clock suffixes; `ESP_Version` populated
- [ ] Old calendar/failures-only parameters gone from `ESP_Models` **and** from the
      regenerated `CreateModelSheet` fallback
- [ ] θ decommissioned per §0.2; Cox columns out of `RunPredictions`
- [ ] Clock conversions live per §0.1 with audit column and override
- [ ] Mode-mix sheet + UDFs (if §0.4 confirmed)
- [ ] `ValidateRegistry` all-green, including the doctored-CSV refusal probe
- [ ] Results write-up under `results_dir("vba_model_v2")` with the validation sheet
      contents and a short "what changed for users" section (old vs new B50 for 3–4
      familiar strata, with the population/clock explanation — expect questions, since
      several B50s will move a lot)
