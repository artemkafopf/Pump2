# Handoff Prompt — Joint Hazard-Aware Refit (Weibull + time-map + hazard), Мирнинский/Ya/Vt

You are implementing a refit of the ESP production-risk survival model so that ONE bundle
reproduces BOTH the stratum survival curves AND the 2024-01..2026-06 historical monthly
failure rates — without the manual calibration factors. Primary fields: **Мирнинский (Mc),
Ярактинский (Ya), Верхнетирский (Vt)**. The goal is presentation-ready results.

Read these first, in order:

1. `docs/notes/production_risk_joint_refit_plan.md` — the full plan (Workstreams A–D). This
   prompt is the executable brief for **Workstream A**; B/C/D follow after the A baseline
   freezes.
2. `docs/notes/weibull_big_censoring_refit_findings.md` — the original censoring audit.
3. `docs/notes/production_risk_ql_hazard_verification_plan.md` — verification tiers to run
   after each change.
4. Project memory `project_production_risk.md` (loaded via MEMORY.md) — traps and history.

Follow `CLAUDE.md` / `agents/AGENTS.md`: paths from `analysis.paths`, outputs via
`results_dir()`, reusable logic in `backend/analysis/`, scripts are thin CLI wrappers.

---

## Background established this session (do not re-derive)

**Why survival curves and failure rates disagree today.** On genuine failures the model
over-predicts: global fact/model ≈ 0.75 (Мирнинский 0.60, Ya 0.76, Vt 0.84). The manual
`_CALIBRATION_FACTORS` (Ya 0.756, Vt 0.837) and reporting-field factors in
`backend/analysis/workflows/production_risk/failure_rate.py` currently paper over this;
the goal is to remove them. Two root causes were confirmed, both fixable in Workstream A,
both push predictions DOWN toward fact:

**Lever 1 — missing censored tail.** The Weibull fit (`mart__weibull_input` from V03 runs)
omits currently-running pumps. Recheck (2026-07-15):

- The prior audit's "810 Big-only censored rows" used a wrong ESP filter and was polluted.
- Corrected filter keeps **501** rows = **one current pump per well across 501 distinct
  wells** (each is the max-install-date run on its well; no fail date, no demount; ZERO
  superseded by a later run). This matches the company's ~500-well fleet. Big is a
  workover/failure ledger with no "running-well" category, but the last run per well IS the
  live pump → legitimate right-censoring, exactly the tail the fit is missing.
- Strata: Ya 165, Vt 131, Mc 56, Za 51, Az 45, Ic 36, Vt_sour 26, Da 17. Median ННО 333
  op-days, installs mostly 2024–2026.
- List for reference: `results/big_open_runs_501.csv`.

**Lever 2 — workovers mis-booked as failures.** `WellsArtificialLiftBig.xlsx` stamps «Дата
отказа» on workover runs too (established in commit 57b). Serial evidence this session
(pump stage-1 serial, Big column 31 «1 ступ. зав. номер») settles the treatment:

- After a **ГТМ** pull the SAME physical pump returns only **4.4%** of the time (after a
  genuine failure 5.6%, after ППР 12.5% but n=24). So a ГТМ pull is a real run termination;
  ~95% of the time a different pump goes in.
- The `нов/рем` flag is unusable — 98.5% empty (119 of 8,287 rows).
- Therefore **ГТМ/ППР pulls are CENSORING events, not failures — even when a fail_date is
  present.** мех.добыча ГТМ ≈ 1,469 runs, ППР ≈ 21: a large censored mass currently likely
  booked as failures.

**Corrected ESP row definition** (the user's correction — three gates, all required):

1. Big column E **«Цель спуска» == «Мех. добыча»** (drops водозаборная ВД/НД,
   нагнетательная, пьезометр, фонтанная, ГРП, техн. операции, консервация, ликвидирована).
2. **`Насос (50Гц)/Тип ГНО`** confirms an ESP by a POSITIVE vocabulary: `ЭЦН*` families PLUS
   REDA/SLB model designations (`D####N`, `G####N`, `S####N`, `GN####`, `ESP 5xx-…`,
   `MT5A-…DP`). **Exclude `ВНН`** (screw pumps, 242 мех.добыча rows) and ШГН-type rows. The
   literal-«ЭЦН» test alone loses 167 genuine SLB units.
3. **`Насос (50Гц)/Собственник оборудования`** = contractor (`Борец`→brt, `Шлюмберже`→slb,
   else oth) — already loaded as `contractor` in `equipment_big.py`.

**Canonical sources now:** events from the fresh Свод
`data/inputs/Отказы свод с анализом.xlsx` (sheet «Свод», 2,891 rows, factual through
2026-06 — the same workbook the packaged EXE pins). Dedup Big-only rows against BOTH Свод
and V03 by `(well_key, install_date ± 7 days)` using `norm_well`. The exact-date dedup was
already sound (only 10/810 matched V03 at ±7d).

---

## Task — Workstream A: censoring-corrected Weibull refit

### A0. Loader change — `backend/analysis/data/equipment_big.py`

- Add `«Цель спуска»` to `_SELECT` (e.g. output column `purpose`, single-level header).
- Add a strict ESP classifier alongside the existing `is_esp_row`: `is_esp_strict` =
  (`purpose` == «Мех. добыча») AND (`Тип ГНО` matches the positive ESP vocabulary above,
  excluding ВНН/ШГН). Keep the old `is_esp` for backward compat; do not break existing
  callers. Expose the pump stage-1 serial (column «1 ступ. зав. номер» under «Насос (50Гц)»)
  as `pump_serial` for the workover audit.
- Emit the ~25 distinct non-ЭЦН `Тип ГНО` model families under мех.добыча as an audit CSV so
  the user can confirm the ESP vocabulary once. Do NOT silently include unknown families.
- Extend `backend/tests/test_equipment_big.py` for the strict filter (ВНН excluded, REDA
  models included, водозаборная excluded).

### A1. Fit-population builder

Rebuild the survival input as: fresh-Свод events + Big-only censored ESP rows (corrected
A0 filter, absent from Свод and V03 by `(well_key, install_date±7d)`, ННО > 0, prefix-
mappable via `C.FIELD_PREFIX_MAP`; MR→Mc kept; NE/AM/ZYI/YAY stay in
`C.EXPLICIT_GLOBAL_FALLBACK`). Produce an audit table reconciling mart-vs-Свод event counts
per stratum. Reuse/extend `scripts/run/refit_weibull_big_censoring.py` rather than starting
fresh, but replace its `is_esp` selection with `is_esp_strict` and its V03-only dedup with
the Свод+V03 dedup.

### A2. Event vs censoring by PULL REASON (applies to the WHOLE population)

- pull reason ∈ {ГТМ, ППР} → **censored at ННО**, even when fail_date is filled.
- genuine-failure pull reasons (Снижение изоляции/R-0, Отсутствие подачи, Нет звезды, Клин
  ЭЦН, Нет подачи-токи х.х., Снижение подачи, Прочие-if-failure…) → **event at ННО**.
- last run per well with no pull → **censored** at current ННО (the 501).
- Do NOT stitch same-pump-back runs (~5%); the survival unit stays one run (install→pull).
- Report, per stratum: event/censor counts vs the old fit AND the number of runs that FLIP
  event→censored under this rule. That flip-count is the expected downward pressure.

### A3. Sour/contractor inheritance
`h2s_proxy_mg_l` for Big-only rows from mart by `well_key`, then by `pad`; unknown →
nonsour + audit count (Vt sour/nonsour split is the one that matters).

### A4. Fit
Existing K=2 EM machinery, same constraints (β₂ > max(1, β₁), η₂/η₁ ≥ 2), same cascade
`field_sour_ctr → field_sour_Pooled → Global_Pooled`. Clock stays **op-days (ttf_mix)** —
the calendar mapping is Workstream B, not a clock change here.

### A5. Mc special path
Integrate `backend/analysis/workflows/production_risk/mc_refit.py` (Свод+Big, MR→Mc,
install-window) into the same builder rather than a post-hoc row swap. Re-test the vintage
window on the censoring-corrected population — the 2024+ window (b50=224) was chosen when
censored mass was missing; with censoring + workover-reclassification the shift may move.
Fit 2023+, 2024+, all-vintage; pick by A6 + D1, document the choice.

### A6. Acceptance (survival side)
For Mc/Ya/Vt strata: model S(t) overlaid on IPCW-KM of the corrected population; max |ΔS| <
0.05 on t ∈ [0, b80]; b50 inside the KM CI; no degenerate-stratum regression. Charts to
`results/esp_survival_big_censored_refit/<date>/figures/`. Also emit the old-vs-new B50
comparison — **the prior audit's ×1.5–3.1 shifts were computed on the polluted 810 and are
overstated; re-measure and report the corrected shifts.**

---

## Deliverables & gates

- Corrected refit bundle candidate + audit tables + KM-overlay figures under
  `results/esp_survival_big_censored_refit/<date>/`.
- The non-ЭЦН vocabulary audit CSV for user confirmation.
- Do NOT ship the bundle, rebuild the EXE, or touch calibration factors yet — Workstream A
  produces the frozen baseline that B and C build on. Stop after A6 with a summary of: event/
  censor/flip counts per stratum, corrected B50 shifts, and whether Mc/Ya/Vt KM acceptance
  passes.
- Run `cd backend && python -m pytest tests/ -q` before finishing; the refit script must run
  end-to-end from repo root.

## Traps (from project memory — do not rediscover)

- Big «Дата отказа» includes ГТМ/ППР workovers — classify by pull reason, not fail_date.
- `esp_run_covariates.csv` is keyed by within-well ordinal, not v03 `Нспуска` (the
  log_run_seq 10× bug). Not directly in scope for A but relevant when C touches covariates.
- Big join key `(well_key, install_date)` with `norm_well`; ±7d tolerance.
- Recomputing calibration factors per run forces ratio→1 and hides drift — factors, if any
  survive in D, are derived once and dated.
- numpy-repr in .bas exports; ChrW() for Cyrillic in VBA (only relevant at D packaging).

## After A (context for continuity, not this task)

- **B**: data-based ttf_true→calendar map (`esp_time_map.csv`) replacing the scalar
  `uptime_factor`; plus the idle-month decision (26% of failures / 34% Mc in plan-non-
  producing months) — measure idle vs producing empirical hazard, choose mis-dating-
  attribution vs a fitted idle-hazard fraction.
- **C**: redo the Cox hazard layer on the frozen A+B baseline — keep Kpod
  (`frac_kpod_below_0p7`, `kpod_freq_mean`), re-screen Ql (do NOT carry β=−0.587/γ=+0.093
  forward), TV-Cox monthly Ql, history replay on ACTUAL Ql; OOS time-split gate; stress-only
  unless it wins OOS.
- **D**: acceptance = fact/model 0.90–1.10 for Mc/Ya/Vt + global with all manual factors =
  1.0; retire stale factors (incl. Мирнинский 0.929, which was derived with the now-retired
  Mc survival-weight); presentation charts (fact vs base vs hazard, KM-vs-model overlay,
  old→new ratio table).
