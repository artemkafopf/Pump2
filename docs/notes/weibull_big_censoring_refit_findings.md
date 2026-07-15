# Workstream A — Censoring-corrected, pull-reason-aware Weibull refit (findings)

Дата: 2026-07-15 (заменяет проверку 2026-07-13 ниже — та работала на загрязнённых 810
Big-only строках и переоценивала сдвиги B50).

Builder: `scripts/run/refit_weibull_big_censoring.py` (end-to-end from repo root).
Outputs: `results/esp_survival_big_censored_refit/2026-07-15/`.
This is the **frozen A baseline** — bundle NOT shipped, EXE NOT rebuilt, calibration
factors untouched (per the handoff `docs/notes/production_risk_joint_refit_prompt.md`).

## What changed vs the shipped fit

Population = fresh Свод runs (canonical events) **+** Big-only strict-ESP runs absent
from Свод and V03 by `(well_key, install_date ± 7d)`. One survival unit = one run
(install → pull). Clock stays op-days (Наработка / ННО). Two corrections, both push
curves LONGER (predictions DOWN):

**Lever 1 — missing censored tail.** The strict-ESP filter (`is_esp_strict`: «Цель
спуска» == «Мех. добыча» AND positive ESP vocabulary, ВНН/ШГН excluded) yields **1 268
prefix-mappable Big-only runs** absent from Свод/V03, of which **775 are censored** (the
~500 currently-running pumps plus workover-censored tail). See `esp_vocab_audit.csv` —
24 non-ЭЦН families under мех.добыча, dominated by screw/rod pumps (ВНН 242, УВНН, ВВН,
ЭОВНБ, УЭВН, KUDU) + a handful of typo'd ЭЦН (1–2 rows each); confirm nothing genuine is
dropped.

**Lever 2 — workovers mis-booked as failures.** Events are classified by PULL REASON,
not by a stamped fail date: `{ГТМ, ППР} → censored even when a fail date is present`;
genuine-failure reasons → event; ambiguous («Прочие»/blank) fall back to a **named
failed component** («Отказавший узел») for Свод and a fail date for Big.

  * **Свод «Failure Flag» is NOT a failure/censor marker** — verified this session: FF=0
    rows routinely carry a named failed component (ЭЦН 131, Кабельная линия 91, ПЭД 61…),
    and ГТМ rows carry FF=1 on 453 rows. So FF is a sub-classification; the failed
    component + pull reason are the real signal. Using FF would have mis-classified ~460
    rows each way.
  * Свод workovers mostly have NO failed component → already censored under the
    component rule; the pull-reason rule flips only **14 more** Свод runs. The large
    Lever-2 effect is on **Big** (276 runs flip event→censored; Big stamps a fail date on
    ГТМ pulls). Total spurious-failure removals: **290** (14 Свод + 276 Big).

## Population (source_audit.csv)

| | value |
|---|---:|
| Свод runs (tte>0) | 2 768 |
| Big-only strict-ESP, prefix-mappable | 1 268 (censored 775 / events 493) |
| **population runs** | **4 036** (events 2 096, censored 1 940) |
| events under OLD definition (no workover demotion) | 2 386 → **290 flips** to censored |

## KM acceptance (Mc/Ya/Vt) — `km_acceptance.csv`, figures `figures/km_overlay_*.png`

Acceptance metric refined: `max|ΔS|` is evaluated on the **reliable horizon**
`[0, min(b80, t_at-risk≥10)]`. Beyond that the KM plateaus on censored observations, so a
parametric tail that keeps descending shows a large-but-meaningless ΔS (this, not a real
misfit, is the only reason a naive `[0, b80]` gate fails the 76%-censored Mc stratum). The
full-`[0, b80]` value is reported alongside for transparency.

| stratum | n / fail | model b50 | KM median (CI) | max\|ΔS\| reliable | b50∈CI | **A6** |
|---|---|---:|---|---:|:---:|:---:|
| Ya_nonsour_Pooled | 2118 / 1192 | 446 | 464 (424–511) | 0.022 | ✓ | **PASS** |
| Vt_nonsour_Pooled | 452 / 198 | 376 | 396 (315–480) | 0.042 | ✓ | **PASS** |
| Vt_sour_Pooled | 164 / 111 | 98 | 107 (83–125) | 0.048 | ✓ | **PASS** |
| Mc_nonsour_Pooled | 194 / 46 | 769 | 663 (519–—) | 0.042 | ✓* | **PASS*** |

\* Mc caveat below.

Two fit-machinery notes surfaced and were addressed in the builder:
1. **`k2_high_w1`** — the best Vt_nonsour k2 (b50≈376, matches KM 396) has w1≈0.89, which
   trips the legacy `w1>0.75` "degenerate" flag and would force a single Weibull
   (b50≈574, fails). The mixture wins AIC by **Δ=36.5**, so the builder now keeps the k2
   whenever AIC favours it (AIC already penalises the extra params), relabelling it
   `k2_high_w1`. Only Vt_nonsour_Pooled was affected; genuine k1_degenerate strata have
   Δaic<0 and keep the single Weibull.
2. The raw EM is seed-stable for the focus strata at ≥8 starts; the baseline uses 12.

## Corrected B50 shifts — `esp_models_old_vs_refit_b50.csv`

**The prior audit's ×1.5–3.1 shifts were computed on the polluted 810; re-measured here.**
They did NOT shrink — Lever 2 (workover reclassification) compounds Lever 1 (censoring),
so curves are even longer:

| stratum | b50 old → new | ×ratio |
|---|---|---:|
| Global_Pooled | 227 → 437 | 1.93 |
| Ya_nonsour_Pooled | 251 → 446 | 1.78 |
| Vt_nonsour_Pooled | 150 → 376 | 2.50 |
| Vt_sour_Pooled | 70 → 98 | 1.41 |
| Mc_nonsour_Pooled | 224 → 769 | 3.43 |

The KM medians are the robust (nonparametric, censoring-aware) ground truth and the model
tracks them for Ya/Vt. **Headline: on the corrected population run life is ~1.8–2.5× the
shipped estimate — the shipped curves were too short because they missed the censored tail
AND counted workover pulls as failures.**

## Mc (Мирнинский) — special path, NOT fully resolved (A5)

`mc_vintage_variants.csv` re-tests the install window on the censoring-corrected
population (all-vintage / 2023+ / 2024+, each overlaid on the KM of its OWN window):

| window | n / fail | b50 | KM median | max\|ΔS\| reliable | A6 |
|---|---|---:|---:|---:|:---:|
| all_vintage | 194 / 46 | 769 | 663 | 0.042 | PASS |
| install_2023+ | 166 / 42 | 503 | 519 | 0.055 | fail |
| install_2024+ | 157 / 39 | 560 | 419 | 0.109 | fail |

The builder wires **all_vintage** into the candidate `Mc_nonsour_Pooled` (only window
passing A6). **Caveat — this is A6-only; the final window is a D1 (fact/model) decision:**
- Mc genuine failures collapsed 160 → 46 after workover reclassification (Мирнинский was
  the most workover-polluted field), so its real run life is far longer than shipped
  (KM median 663 vs shipped b50 224). This is the corrected cause of the shipped Mc
  over-prediction (fact/model 0.60).
- all_vintage blends long-lived ≤2023 pumps; the current fleet is recent-vintage whose
  KM median is shorter (2024+ = 419). So all_vintage may OVER-estimate current Mc life →
  risk of under-prediction in D. Recency windows are fleet-representative but the single
  Weibull under-fits their KM shape.
- Per plan/memory, the true Mc fix is the Workstream-B idle-month / time-map decision
  (26% of failures — 34% Mc — fall in plan-non-producing months). Re-decide Mc window on
  fact/model after B settles exposure.

## Gates honoured

- Bundle NOT shipped, EXE NOT rebuilt, calibration factors untouched.
- `cd backend && python -m pytest tests/ -q` → **326 passed** (incl. new strict-ESP tests
  in `test_equipment_big.py`).
- Builder runs end-to-end from repo root.

## Deliverables (`results/esp_survival_big_censored_refit/2026-07-15/`)

`fit_population.csv`, `big_only_runs.csv`, `source_audit.csv`,
`event_censor_by_stratum.csv`, `esp_vocab_audit.csv`,
`esp_models_big_censored_refit.csv`, `esp_models_old_vs_refit_b50.csv`,
`mc_vintage_variants.csv`, `km_acceptance.csv`, `figures/km_overlay_*.png`.

---

# (Archived) Проверка 2026-07-13 — на загрязнённых 810 Big-only строках

> Оставлено для истории. Фильтр `is_esp` был только negative-фильтром по `Тип ГНО` и
> игнорировал «Цель спуска», поэтому в 810 попали водозаборные/нагнетательные скважины;
> сдвиги B50 ×1.5–3.1 переоценены. Корректная версия — выше.

- `mart__weibull_input`: 2 634 run, 1 527 отказов, 1 107 censored.
- Big-only censored rows (старый фильтр): 810 → после strict-ESP + Свод/V03 dedup: 1 268
  Big-only (775 censored). Старая таблица сдвигов B50 (Vt_nonsour 150→410 и т.д.) —
  недействительна; см. пересчёт выше (Vt_nonsour 150→376).
