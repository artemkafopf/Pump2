"""СВОДПРОГНОЗ (Mc) — прогноз отказов «Прогноз_ремонтов», два файла:

* **Сводпрогноз_Mc.xlsx**        — базовый прогноз отказов (cal c0 k2), без слоя Ql.
* **Сводпрогноз_hazard_Mc.xlsx** — тот же прогноз + слой Ql `θ = (Ql/Ql_ref)^β`,
  β = ln(1.253) (landmark; совпал с cross-fit PH на Ya/Vt ~0.25). Форма PH:
  `S(t|Ql) = S0(t)^θ` — работает для смеси k=2 без `beta0` (`esp_forecast.forecast`
  применяет `q_eff = 1-(1-q)^θ`).

Обе книги несут ОБЯЗАТЕЛЬНЫЙ график «Интенсивность отказов» (факт vs модель, % от парка,
история+прогноз одной моделью). **Факт соединён ломаной линией с маркерами** (не только
точки), модель — сплошной линией.

**Слой Ql на Mc — СЦЕНАРНЫЙ.** `c4_gate` (выигрыш MAE вне выборки 2025-01..2026-06) даёт
≈0% при пороге 5% ⇒ на 44 отказах Mc слой НЕ улучшает месячный счёт. Ql — реальный драйвер
на толстых полях (Ya β=0.31 p=1e-11, Vt подтверждён), но на Mc мал по мощности. θ здесь
**перераспределяет** риск (кто раньше), тотал НЕ меняет: перенормирована к mean=1 отдельно
на прогнозной и исторической популяции, иначе выпуклость θ (Йенсен) подняла бы總 ~3.3%.

Охват: только Мирнинский (Mc + Mr). Ql — плановый (ТМ-06 `Добыча жидкости`/`Отработанное
время`); история — t0-окно 30 оп-суток.

Запуск:  python scripts/run/export_mc_svodprognoz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk import crosswalk as cw
from analysis.workflows.production_risk import t0_covariates as t0
from analysis.workflows.production_risk import repair_compat

MC_PLAN_FIELD = "Мирнинский УН"
HIST_FIRST, HIST_LAST = "2024-01", "2026-06"
HORIZON_MONTHS = 18
METHOD = "statistical"
QL_BETA = M.QL_BETA_LANDMARK  # ln(1.253); cross-fit PH on Ya/Vt = 0.25 (agrees)


def _theta_redistributive(ql_by_code: dict[str, float]) -> tuple[dict[str, float], float, float]:
    """θ = (Ql/Ql_ref)^β, ref = геосредний Ql этой популяции, перенормировка к mean=1.

    Возвращает (theta_by_code, Ql_ref, mean_raw). mean_raw — среднее θ ДО перенормировки:
    отклонение от 1 = скрытый сдвиг тотала, который мы убираем (см. модульный docstring).
    """
    codes = [c for c, v in ql_by_code.items() if v is not None and np.isfinite(v) and v > 0]
    vals = np.array([ql_by_code[c] for c in codes], dtype=float)
    if len(vals) == 0:
        return {}, float("nan"), float("nan")
    ref_log = float(np.mean(np.log1p(vals)))
    theta = {c: M.ql_theta(ql_by_code[c], ref_log, beta=QL_BETA) for c in codes}
    mean_raw = float(np.mean(list(theta.values())))
    theta = {c: v / mean_raw for c, v in theta.items()}  # pure redistribution
    return theta, float(np.expm1(ref_log)), mean_raw


def _plan_ql_by_code(codes: set[str]) -> dict[str, float]:
    """Плановый Ql (м3/сут) по скважинам: ТМ-06 «Добыча жидкости»/«Отработанное время»."""
    plan = cw.load_plan()
    pc = t0.plan_covariates(plan)
    idx_cf = {str(i).casefold(): i for i in pc.index}
    out: dict[str, float] = {}
    for c in codes:
        key = idx_cf.get(str(c).casefold())
        if key is None:
            continue
        v = pc.loc[key, "ql"]
        if pd.notna(v) and v > 0:
            out[c] = float(v)
    return out


def _hist_ql_by_code(pop: pd.DataFrame) -> dict[str, float]:
    """Исторический t0-Ql (окно 30 оп-суток) по пробегам Mc — вход для history_match/c4."""
    d, _ = t0.build(P.select(pop, "Mc", h2s="nonsour"))
    d = d[d["ql"].notna() & (d["ql"] > 0)]
    return {r.code: float(r.ql) for r in d.itertuples()}


def _intensity_sheet(hist: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    h = hist.rename(columns={"модель ожидает": "модель"})[
        ["месяц", "в работе", "факт", "модель", "факт, % от парка", "модель, % от парка"]].copy()
    h["период"] = "история"
    f = fc.rename(columns={"ожидаемые события": "модель"})[
        ["месяц", "в работе", "модель", "модель, % от парка"]].copy()
    f["факт"] = float("nan")
    f["факт, % от парка"] = float("nan")
    f["период"] = "прогноз"
    out = pd.concat([h, f[h.columns]], ignore_index=True)
    out["модель"] = out["модель"].astype(float).round(2)
    return out


def _add_intensity_chart(path: Path, intensity: pd.DataFrame, title_suffix: str) -> None:
    """График «Интенсивность отказов»: факт vs модель, % от парка, история+прогноз.

    Факт — ЛОМАНАЯ линия с маркерами (по просьбе пользователя: не только точки), модель —
    сплошная линия. Факт обрывается на границе история/прогноз (в прогнозе факта нет).
    """
    import openpyxl
    from openpyxl.chart import LineChart, Reference
    from openpyxl.chart.axis import ChartLines

    wb = openpyxl.load_workbook(path)
    ws = wb["Интенсивность отказов"]
    cols = list(intensity.columns)
    n = len(intensity)
    c_fact = cols.index("факт, % от парка") + 1
    c_model = cols.index("модель, % от парка") + 1

    chart = LineChart()
    chart.title = "Интенсивность отказов УЭЦН — Мирнинский (cal c0 k2%s): факт vs модель" % title_suffix
    chart.y_axis.title = "% от парка в работе"
    chart.x_axis.title = "Месяц"
    chart.y_axis.majorGridlines = ChartLines()
    chart.height, chart.width = 9, 28
    for col in (c_fact, c_model):
        chart.add_data(Reference(ws, min_col=col, min_row=1, max_row=n + 1), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=n + 1))
    s_fact, s_model = chart.series[0], chart.series[1]
    # Факт: ломаная линия С маркерами (пользователь просил соединить точки линией).
    s_fact.graphicalProperties.line.width = 15000
    s_fact.graphicalProperties.line.dashStyle = "sysDash"
    s_fact.marker.symbol = "circle"
    s_fact.marker.size = 5
    s_fact.smooth = False
    # Модель: сплошная жирная линия.
    s_model.graphicalProperties.line.width = 22000
    s_model.smooth = False
    ws.add_chart(chart, "J2")
    wb.save(path)


def _build_and_write(path: Path, *, hazard: bool, pop, model, live, ent, entrants,
                     codes, fleet, asof, gate_verdict, theta_fc, theta_hist, ref_fc) -> pd.DataFrame:
    """Собрать одну книгу (базовую или hazard) и записать с обязательным графиком."""
    th_fc = theta_fc if hazard else None
    th_hist = theta_hist if hazard else None

    payload = M.repair_payload(live, model, asof=asof, horizon_months=HORIZON_MONTHS,
                               method=METHOD, theta_by_code=th_fc, entrants=ent)
    hist = M.history_match(model, pop, first=HIST_FIRST, last=HIST_LAST,
                           theta_by_code=th_hist, fleet_by_month=fleet)
    fc, _ = M.forecast(model, live, asof=asof, months=HORIZON_MONTHS,
                       theta_by_code=th_fc, fleet_by_month=fleet, entrants=ent)
    intensity = _intensity_sheet(hist, fc)

    if hazard:
        payload["notes"] = payload["notes"] + [
            "СВОДПРОГНОЗ (hazard): базовый прогноз (cal c0 k2) + слой Ql θ=(Ql/Ql_ref)^%.4f." % QL_BETA,
            "Слой Ql ПЕРЕРАСПРЕДЕЛЯЕТ риск (кто раньше), тотал НЕ меняет: θ перенормирована к mean=1.",
            "c4_gate (выигрыш MAE вне выборки): %s" % gate_verdict,
            "Ql плановый (ТМ-06 Добыча жидкости/Отработанное время), Ql_ref(прогноз)=%.1f м3/сут." % ref_fc,
        ]
    else:
        payload["notes"] = payload["notes"] + [
            "СВОДПРОГНОЗ (базовый): прогноз отказов cal c0 k2, БЕЗ слоя Ql.",
        ]

    written = repair_compat.write_excel(path, payload)
    with pd.ExcelWriter(written, engine="openpyxl", mode="a") as xl:
        intensity.to_excel(xl, sheet_name="Интенсивность отказов", index=False)
        hist.to_excel(xl, sheet_name="История_факт_модель", index=False)
        if hazard:
            fc_codes = sorted(theta_fc)
            pd.DataFrame([{"well": c, "Ql_plan_theta": round(theta_fc[c], 4)} for c in fc_codes]
                         ).to_excel(xl, sheet_name="Слой_Ql", index=False)
    _add_intensity_chart(written, intensity, " + Ql" if hazard else "")

    rows = repair_compat.rows_frame(payload)
    print("  %-28s строк=%d событий=%d" % (path.name, len(rows), int(rows["event_count"].sum())))
    return rows


def main() -> None:
    cfg = C.RunConfig()
    asof = pd.Timestamp(str(cfg.forecast_start))
    pop = P.build(str(cfg.forecast_start))
    model = M.fit_mc_model(pop)

    codes = set(pop[pop["field"] == "Mc"]["code"])
    live0 = M.live_fleet(pop, asof, field="Mc")
    live_codes = set(live0["code"])
    entrants = M.plan_entrants(live_codes, codes, asof, HORIZON_MONTHS)

    meta = M.plan_metadata(list(codes | set(entrants["code"])), asof)
    live = M.live_fleet(pop, asof, field="Mc")
    live["license_area"] = MC_PLAN_FIELD
    live = live.merge(meta, on="code", how="left")
    ent = entrants.merge(meta, on="code", how="left")
    ent["license_area"] = MC_PLAN_FIELD
    ent["raw_id"] = ent["code"]
    ent["pred_source"] = ent["code"].str.startswith("NE_").map(
        {True: "survival (Mc-модель, другое м-е)", False: "survival"})
    if live.empty:
        raise SystemExit("Работающие насосы Mc не найдены — проверить популяцию")

    # ── Ql-слой ──────────────────────────────────────────────────────────────
    fc_codes = live_codes | set(ent["code"])
    theta_fc, ref_fc, mean_fc = _theta_redistributive(_plan_ql_by_code(fc_codes))
    theta_hist, ref_hist, mean_hist = _theta_redistributive(_hist_ql_by_code(pop))
    gate = M.c4_gate(model, pop, theta_hist)
    verdict = str(gate.loc[1, "вердикт"])
    print("Ql-слой: β=%.4f | прогноз Ql_ref=%.1f (mean θ raw=%.4f) | история Ql_ref=%.1f (raw=%.4f)"
          % (QL_BETA, ref_fc, mean_fc, ref_hist, mean_hist))
    print("c4_gate:", verdict)

    out = results_dir("production_risk_mc_svodprognoz")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    common = dict(pop=pop, model=model, live=live, ent=ent, entrants=entrants, codes=codes,
                  fleet=M.active_fleet_by_month(codes=codes | set(entrants["code"])), asof=asof,
                  gate_verdict=verdict, theta_fc=theta_fc, theta_hist=theta_hist, ref_fc=ref_fc)

    print("записано:")
    _build_and_write(out / "tables" / "Сводпрогноз_Mc.xlsx", hazard=False, **common)
    _build_and_write(out / "tables" / "Сводпрогноз_hazard_Mc.xlsx", hazard=True, **common)
    # c4_gate тем же гейтом кладём в hazard-книгу отдельным листом.
    with pd.ExcelWriter(out / "tables" / "Сводпрогноз_hazard_Mc.xlsx", engine="openpyxl", mode="a") as xl:
        gate.to_excel(xl, sheet_name="c4_gate_Ql", index=False)


if __name__ == "__main__":
    main()
