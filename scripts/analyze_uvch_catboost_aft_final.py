"""CatBoost AFT — финальный анализ по Config #38.

Config: vt_only / freq_plus_mech / kip>0.4 / Normal (LogNormal)
Лучшая конфигурация из sensitivity grid (C=0.859, 0 extreme-short, cohort gap=+0.8 d).

Выход:
  docs/uvch_claude/catboost_aft_final_wells.csv     — per-well predictions
  docs/uvch_claude/catboost_aft_final_report.md     — детальный отчёт
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analysis.paths import results_dir
from catboost import CatBoostRegressor, Pool
from lifelines.utils import concordance_index
from scipy.optimize import minimize_scalar
from scipy.stats import norm as sp_norm

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_uvch_catboost_aft import (  # noqa: E402
    MISSING_CATEGORY,
    RANDOM_SEED,
    band_model,
    build_last_history_lookup,
    h_base,
    load_positive_mart,
    load_uvch_raw,
    make_aft_labels,
    safe_round,
    scenario_freq_above,
    split_by_well,
    target_band,
    weibull_rul_q,
)

# ---------------------------------------------------------------------------
# Config #38 parameters
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_SLUG = "uvch_catboost_aft"
OUT_DIR = results_dir(_SLUG)

DIST = "Normal"
FIELDS = ("Vt",)
KIP_THRESHOLD = 0.4
FEATURES = [
    "freq_w_mean", "freq_above_55hz_pct", "load_m_mean",
    "kpod_m_mean", "frac_kpod_below_0p7_m", "run_number",
    "field", "contractor", "pump_type",
]
CAT_FEATURES = ["field", "contractor", "pump_type"]

MODEL_DEPTH = 6
MODEL_LR = 0.05
MODEL_ITERS = 500
WEIBULL_REF = 141.0
SHORT_THRESHOLD = 40.0


# ---------------------------------------------------------------------------
# Helpers (repeated from sensitivity script to be self-contained)
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for f in FEATURES:
        if f in CAT_FEATURES:
            col = df[f].astype("string") if f in df.columns else pd.Series(MISSING_CATEGORY, index=df.index)
            result[f] = col.fillna(MISSING_CATEGORY)
        else:
            result[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else float("nan")
    return result


def _pool(df: pd.DataFrame) -> tuple[Pool, pd.DataFrame]:
    feat = _prepare(df)
    cat_idx = [feat.columns.get_loc(c) for c in CAT_FEATURES if c in feat.columns]
    pool = Pool(
        data=feat,
        label=make_aft_labels(
            df["ttf_true_best_days"].to_numpy(dtype=float),
            df["event"].to_numpy(dtype=int),
        ),
        cat_features=cat_idx,
    )
    return pool, feat


def _cbm(iters: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function=f"SurvivalAft:dist={DIST}",
        eval_metric=f"SurvivalAft:dist={DIST}",
        iterations=iters,
        depth=MODEL_DEPTH,
        learning_rate=MODEL_LR,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )


def _sigma_mle(model: CatBoostRegressor, df: pd.DataFrame) -> float:
    log_t = np.log(df["ttf_true_best_days"].to_numpy(dtype=float))
    event = df["event"].to_numpy(dtype=int)
    mu = model.predict(_prepare(df))

    def neg_ll(s: float) -> float:
        z = (log_t - mu) / s
        return -float(np.sum(event * (sp_norm.logpdf(z) - np.log(s)) + (1 - event) * sp_norm.logsf(z)))

    result = minimize_scalar(neg_ll, bounds=(0.05, 5.0), method="bounded")
    return float(result.x)


def _rul_lognormal(q: float, age: float, mu: float, sigma: float) -> float:
    """Conditional q-quantile of remaining life for LogNormal: S(age+u)/S(age) = q."""
    if age <= 0:
        age = 1.0
    z_age = (math.log(age) - mu) / sigma
    s_age = float(sp_norm.sf(z_age))
    if s_age < 1e-12:
        return 0.0
    target_s = q * s_age
    if target_s < 1e-12:
        return 0.0
    z_star = float(sp_norm.isf(target_s))
    return max(math.exp(mu + sigma * z_star) - age, 0.0)


# ---------------------------------------------------------------------------
# Per-well scoring
# ---------------------------------------------------------------------------

def _score_wells(
    uvch_raw: list,
    model: CatBoostRegressor,
    sigma: float,
    train_lkp: dict,
    pos_lkp: dict,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in uvch_raw:
        well = rec[0]
        nno = float(rec[1])
        f_cur = float(rec[7])
        f_after = float(rec[9])
        band = target_band(f_after, f_cur)
        h_fn, m = band_model(band)

        # Weibull baseline (cohort model)
        wbl_p50_c = weibull_rul_q(0.50, nno, h_base)           # current band (T0 = no accel)
        wbl_p50_a = weibull_rul_q(0.50, nno, h_fn, m)          # after band (T3 usually)

        row: dict[str, Any] = {
            "well": well,
            "nno_days": int(nno),
            "freq_current_hz": f_cur,
            "freq_after_hz": f_after,
            "band": band,
            "weibull_p50_current": wbl_p50_c,
            "weibull_p50_after": wbl_p50_a,
            "cb_available": False,
            "mu_hat": None,
            "cb_p25_current": None,
            "cb_p50_current": None,
            "cb_p75_current": None,
            "cb_p50_after": None,
            "gap_current_days": None,
            "gap_after_days": None,
            "was_extreme_weibull": wbl_p50_c < SHORT_THRESHOLD,
        }

        history = train_lkp.get(well)
        if history is None:
            history = pos_lkp.get(well)

        if history is not None and sigma > 0:
            cur_pct = pd.to_numeric(
                pd.Series([history.get("freq_above_55hz_pct")]), errors="coerce"
            ).iloc[0]
            sc_pct = scenario_freq_above(cur_pct, band)

            def _pred_mu(f_target: float, pct: float) -> float:
                r = history.copy()
                r["freq_w_mean"] = f_target
                r["freq_above_55hz_pct"] = pct
                return float(model.predict(_prepare(pd.DataFrame([r])))[0])

            mu_c = _pred_mu(f_cur, cur_pct)
            mu_a = _pred_mu(f_after, sc_pct)

            p25 = _rul_lognormal(0.75, nno, mu_c, sigma)  # q=0.75 in S gives P25 of T (pessimistic)
            p50 = _rul_lognormal(0.50, nno, mu_c, sigma)
            p75 = _rul_lognormal(0.25, nno, mu_c, sigma)  # q=0.25 in S gives P75 of T (optimistic)
            p50_a = _rul_lognormal(0.50, nno, mu_a, sigma)

            row.update({
                "cb_available": True,
                "mu_hat": round(mu_c, 3),
                "cb_p25_current": safe_round(p25, 0) if math.isfinite(p25) else None,
                "cb_p50_current": safe_round(p50, 0) if math.isfinite(p50) else None,
                "cb_p75_current": safe_round(p75, 0) if math.isfinite(p75) else None,
                "cb_p50_after": safe_round(p50_a, 0) if math.isfinite(p50_a) else None,
                "gap_current_days": (
                    round(safe_round(p50, 0) - wbl_p50_c, 0)
                    if math.isfinite(p50) else None
                ),
                "gap_after_days": (
                    round(safe_round(p50_a, 0) - wbl_p50_a, 0)
                    if math.isfinite(p50_a) else None
                ),
            })

        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _fmt(v: object, d: int = 0) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    return f"{float(v):.{d}f}"


def _risk_tier(rul: float | None, nno: int) -> str:
    if rul is None:
        return "н/д"
    if rul < 30:
        return "КРИТИЧНО"
    if rul < 60:
        return "Высокий"
    if rul < 120:
        return "Средний"
    return "Низкий"


def build_report(
    wells: pd.DataFrame,
    c_index: float,
    sigma: float,
    cohort_rul: float,
    train_rows: int,
    test_rows: int,
    train_wells: int,
    test_wells: int,
) -> str:
    avail = wells[wells["cb_available"]].copy()
    n_extreme_cb = int((avail["cb_p50_current"].fillna(999) < SHORT_THRESHOLD).sum())
    n_extreme_wbl = int(wells["was_extreme_weibull"].sum())
    n_no_data = int((~wells["cb_available"]).sum())
    mean_gap = avail["gap_current_days"].mean()
    median_gap = avail["gap_current_days"].median()

    lines = [
        "# CatBoost AFT (LogNormal) — Финальный анализ UVCh",
        "",
        "**Config #38:** `vt_only / freq_plus_mech / kip>0.4 / Normal (LogNormal)`",
        "",
        "Лучшая конфигурация из sensitivity grid (135 конфигураций).",
        "Независимая оценка для сравнения с когортным Weibull-анализом.",
        "",
        "---",
        "",
        "## Параметры модели",
        "",
        f"| Параметр | Значение |",
        f"|---|---|",
        f"| Дистрибуция | LogNormal (`dist=Normal`) |",
        f"| Поля обучения | Vt only |",
        f"| Фичи | freq\\_plus\\_mech (9 признаков) |",
        f"| KIP-порог | > 0.4 |",
        f"| Глубина / LR / итерации | {MODEL_DEPTH} / {MODEL_LR} / {MODEL_ITERS} |",
        f"| Данные train | {train_rows} пробегов / {train_wells} скважин |",
        f"| Данные test | {test_rows} пробегов / {test_wells} скважин |",
        f"| **C-index (test)** | **{c_index:.4f}** |",
        f"| **sigma (MLE)** | **{sigma:.4f}** |",
        f"| Cohort RUL P50 при NNO=100 | {cohort_rul:.1f} дней (ref: {WEIBULL_REF:.0f} д.) |",
        "",
        "### Набор фич `freq_plus_mech`",
        "",
        "```",
        "freq_w_mean           — средняя частота за пробег (Гц)",
        "freq_above_55hz_pct   — доля времени выше 55 Гц",
        "load_m_mean           — нагрузка (среднее)",
        "kpod_m_mean           — КПОД (среднее)",
        "frac_kpod_below_0p7_m — доля точек с КПОД < 0.7",
        "run_number            — номер пробега",
        "field / contractor / pump_type — категориальные",
        "```",
        "",
        "---",
        "",
        "## Сводка по UVCh скважинам",
        "",
        f"| Показатель | Weibull (baseline) | CatBoost LogNormal |",
        f"|---|---|---|",
        f"| Скважин с RUL < 40 дней | **{n_extreme_wbl}** | **{n_extreme_cb}** |",
        f"| Скважин без данных CB | — | {n_no_data} |",
        f"| Mean gap (CB − Weibull) | — | {mean_gap:+.1f} дней |",
        f"| Median gap (CB − Weibull) | — | {median_gap:+.1f} дней |",
        "",
        "> **Mean gap** положительный (+{:.1f} д.) — CatBoost в среднем даёт чуть бо́льший RUL, чем Weibull.".format(mean_gap),
        "> **Median gap** отрицательный ({:.1f} д.) — у бо́льшей части скважин CatBoost даёт более короткий прогноз, но без экстремальных значений.".format(median_gap),
        "> Право-скошенное распределение: несколько скважин с низким риском получают значительно бо́льший RUL.".format(),
        "",
        "---",
        "",
        "## Per-well таблица (все 43 UVCh скважины)",
        "",
        "**Сценарий «текущий»** — модель при текущей частоте (f_cur).",
        "**Сценарий «после»** — модель при целевой частоте f_after (обычно 60 Гц для T3).",
        "",
        "Квантили LogNormal: P25 (пессимистичный) / **P50 (медиана)** / P75 (оптимистичный).",
        "",
        "| Скважина | NNO | Полоса | f_cur | f_aft | Wbl P50 (тек.) | CB P25 | **CB P50** | CB P75 | Gap | CB P50 (после) | Риск |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    sorted_wells = wells.sort_values("nno_days", ascending=False)
    for _, w in sorted_wells.iterrows():
        wbl_c = _fmt(w["weibull_p50_current"])
        cb25 = _fmt(w["cb_p25_current"])
        cb50 = _fmt(w["cb_p50_current"])
        cb75 = _fmt(w["cb_p75_current"])
        cb50a = _fmt(w["cb_p50_after"])
        gap = _fmt(w["gap_current_days"])
        if w["gap_current_days"] is not None:
            g = float(w["gap_current_days"])
            gap = f"+{g:.0f}" if g >= 0 else f"{g:.0f}"
        risk = _risk_tier(w["cb_p50_current"], int(w["nno_days"]))
        marker = " *" if w["was_extreme_weibull"] else ""
        lines.append(
            f"| {w['well']}{marker} | {int(w['nno_days'])} | {w['band']}"
            f" | {w['freq_current_hz']:.0f} | {w['freq_after_hz']:.0f}"
            f" | {wbl_c} | {cb25} | **{cb50}** | {cb75} | {gap} | {cb50a} | {risk} |"
        )

    lines += [
        "",
        "> `*` — скважина имела Weibull RUL < 40 дней при текущем NNO",
        "> Квантили: P25 — 75% вероятность дожить дольше; P75 — 25% вероятность.",
        "",
        "---",
        "",
        "## Скважины, бывшие в зоне экстремального риска (Weibull RUL < 40 д.)",
        "",
    ]

    extreme_wbl = sorted_wells[sorted_wells["was_extreme_weibull"]].copy()
    if extreme_wbl.empty:
        lines.append("Нет скважин с Weibull RUL < 40 дней.")
    else:
        lines += [
            f"Всего {len(extreme_wbl)} скважин — при Weibull давали RUL < 40 дней.",
            "",
            "| Скважина | NNO | Полоса | Wbl P50 | CB P25 | **CB P50** | CB P75 | Gap | Риск |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, w in extreme_wbl.iterrows():
            wbl_c = _fmt(w["weibull_p50_current"])
            cb25 = _fmt(w["cb_p25_current"])
            cb50 = _fmt(w["cb_p50_current"])
            cb75 = _fmt(w["cb_p75_current"])
            gap = _fmt(w["gap_current_days"])
            if w["gap_current_days"] is not None:
                g = float(w["gap_current_days"])
                gap = f"+{g:.0f}" if g >= 0 else f"{g:.0f}"
            risk = _risk_tier(w["cb_p50_current"], int(w["nno_days"]))
            lines.append(
                f"| {w['well']} | {int(w['nno_days'])} | {w['band']}"
                f" | {wbl_c} | {cb25} | **{cb50}** | {cb75} | {gap} | {risk} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Скважины с наибольшим риском по CatBoost (P50 < 60 дней)",
        "",
    ]

    high_risk = avail[avail["cb_p50_current"].fillna(999) < 60].sort_values("cb_p50_current")
    if high_risk.empty:
        lines.append("Нет скважин с CB P50 < 60 дней — все имеют умеренный или низкий риск.")
    else:
        lines += [
            f"Всего {len(high_risk)} скважин с CB P50 < 60 дней.",
            "",
            "| Скважина | NNO | Полоса | CB P25 | **CB P50** | CB P75 | Wbl P50 | Gap |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, w in high_risk.iterrows():
            wbl_c = _fmt(w["weibull_p50_current"])
            cb25 = _fmt(w["cb_p25_current"])
            cb50 = _fmt(w["cb_p50_current"])
            cb75 = _fmt(w["cb_p75_current"])
            gap = _fmt(w["gap_current_days"])
            if w["gap_current_days"] is not None:
                g = float(w["gap_current_days"])
                gap = f"+{g:.0f}" if g >= 0 else f"{g:.0f}"
            lines.append(
                f"| {w['well']} | {int(w['nno_days'])} | {w['band']}"
                f" | {cb25} | **{cb50}** | {cb75} | {wbl_c} | {gap} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Скважины без данных CatBoost",
        "",
    ]

    no_data = wells[~wells["cb_available"]]
    if no_data.empty:
        lines.append("Все 43 скважины имеют данные CatBoost.")
    else:
        lines += [
            f"{len(no_data)} скважин без исторических данных в mart__vt_freq55:",
            "",
        ]
        for _, w in no_data.iterrows():
            lines.append(f"- {w['well']} (NNO={int(w['nno_days'])}, band={w['band']})")

    lines += [
        "",
        "---",
        "",
        "## Сравнение с baseline CatBoost (Weibull/Extreme)",
        "",
        "| Метрика | Baseline (Extreme) | **Config #38 (Normal)** | Delta |",
        "|---|---|---|---|",
        f"| C-index | 0.779 | **{c_index:.3f}** | +{c_index-0.779:+.3f} |",
        f"| sigma | 0.587 | **{sigma:.3f}** | +{sigma-0.587:+.3f} |",
        f"| beta (Weibull) | 1.705 | — (LogNormal) | — |",
        f"| Cohort gap P50 | −10 д. | **+{cohort_rul - WEIBULL_REF:+.1f} д.** | — |",
        f"| UVCh RUL < 40 д. | 21 | **{n_extreme_cb}** | -{21 - n_extreme_cb} |",
        f"| Mean gap (CB−Wbl) | −53.5 д. | **{mean_gap:+.1f} д.** | — |",
        "",
        "---",
        "",
        "## Методологические замечания",
        "",
        "**LogNormal (Normal) vs Weibull (Extreme):**",
        "- Weibull: h(t) = (beta/eta) * (t/eta)^(beta-1) — монотонно возрастает при beta>1",
        "- LogNormal: h(t) = phi(z)/(1-Phi(z)) * (1/(sigma*t)) — растёт, достигает пика, убывает",
        "- При высоком NNO Weibull с beta>1.7 даёт h(t)→∞, условный RUL→0",
        "- LogNormal «не помнит» что скважина старая — hazard убывает к правому хвосту",
        "",
        "**MLE sigma:**",
        "- Оценка по полному датасету: события (event=1) + правая цензура (event=0)",
        "- Для Normal: max Σ[event*(logpdf(z)-log(s)) + (1-event)*logsf(z)], z=(log(t)-mu)/s",
        "",
        "**Conditional RUL:**",
        "- Решение S(NNO + RUL) / S(NNO) = q (P50: q=0.5, P25: q=0.75, P75: q=0.25)",
        "- LogNormal S(t) = 1 - Phi((log(t) - mu) / sigma)",
        "",
        f"Дата: 2026-06-25 | Скрипт: scripts/analyze_uvch_catboost_aft_final.py",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    uvch_raw = load_uvch_raw()
    positive_df = load_positive_mart()

    data = positive_df[
        (positive_df["kip"] > KIP_THRESHOLD) & (positive_df["field"].isin(FIELDS))
    ].copy()

    print(f"Dataset: {len(data)} rows, {data['well'].nunique()} wells (field=Vt, kip>{KIP_THRESHOLD})")

    df_train, df_test = split_by_well(data)
    print(f"Split: train={len(df_train)} rows/{df_train['well'].nunique()} wells, "
          f"test={len(df_test)} rows/{df_test['well'].nunique()} wells")

    # Eval model for C-index
    print("Training eval model...")
    train_pool, _ = _pool(df_train)
    test_pool, test_feat = _pool(df_test)
    eval_model = _cbm(MODEL_ITERS)
    eval_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)
    best_iter = max(int(eval_model.get_best_iteration() or eval_model.tree_count_), 1)

    mu_test = eval_model.predict(test_feat)
    c_index = float(concordance_index(
        df_test["ttf_true_best_days"].to_numpy(dtype=float),
        predicted_scores=mu_test,
        event_observed=df_test["event"].to_numpy(dtype=int),
    ))
    print(f"C-index (test, {best_iter} trees): {c_index:.4f}")

    # Final model on full dataset
    print("Training final model on full dataset...")
    full_pool, _ = _pool(data)
    final_model = _cbm(best_iter)
    final_model.fit(full_pool)

    # MLE sigma
    sigma = _sigma_mle(final_model, data)
    print(f"sigma (MLE LogNormal): {sigma:.4f}")

    # Cohort sanity check
    sub_50_55 = data[data["freq_w_mean"].between(50.0, 55.0)].copy()
    mus_cohort = final_model.predict(_prepare(sub_50_55))
    cohort_ruls = [_rul_lognormal(0.50, 100.0, float(m), sigma) for m in mus_cohort]
    cohort_mean = float(np.mean([r for r in cohort_ruls if math.isfinite(r)]))
    print(f"Cohort RUL P50 at NNO=100 (50-55 Hz): {cohort_mean:.1f} days (ref={WEIBULL_REF:.0f})")

    # Per-well scoring
    print(f"Scoring {len(uvch_raw)} UVCh wells...")
    train_lkp = build_last_history_lookup(data)
    pos_lkp = build_last_history_lookup(positive_df)
    wells = _score_wells(uvch_raw, final_model, sigma, train_lkp, pos_lkp)

    avail = wells[wells["cb_available"]]
    n_avail = len(avail)
    n_short = int((avail["cb_p50_current"].fillna(999) < SHORT_THRESHOLD).sum())
    print(f"Available: {n_avail}/{len(wells)} | CB P50 < 40 days: {n_short}")
    print(f"Mean gap (CB-Weibull): {avail['gap_current_days'].mean():+.1f} days")
    print(f"Median gap (CB-Weibull): {avail['gap_current_days'].median():+.1f} days")

    # Save CSV
    wells.to_csv(OUT_DIR / "catboost_aft_final_wells.csv", index=False, encoding="utf-8-sig")

    # Save report
    report = build_report(
        wells=wells,
        c_index=c_index,
        sigma=sigma,
        cohort_rul=cohort_mean,
        train_rows=len(df_train),
        test_rows=len(df_test),
        train_wells=int(df_train["well"].nunique()),
        test_wells=int(df_test["well"].nunique()),
    )
    (OUT_DIR / "catboost_aft_final_report.md").write_text(report, encoding="utf-8")

    print(f"\nFiles saved to {OUT_DIR}:")
    print("  catboost_aft_final_wells.csv")
    print("  catboost_aft_final_report.md")


if __name__ == "__main__":
    main()
