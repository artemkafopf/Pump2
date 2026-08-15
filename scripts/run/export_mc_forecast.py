"""Прогноз_отказов по Мирнинскому УН (Mc+Mr) — Excel на ревью.

Запуск:  python scripts/run/export_mc_forecast.py
Выход :  results/production_risk_mc_forecast/<дата>/tables/Прогноз_отказов_Mc.xlsx
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd

from analysis.paths import WAREHOUSE_DIR, results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import esp_forecast as M

ASOF = pd.Timestamp("2026-07-01")
HIST_FIRST, HIST_LAST = "2024-01", "2026-06"
HORIZON_MONTHS = 18


def _ql_by_code(live: pd.DataFrame) -> pd.DataFrame:
    """Ql на скважину: телеметрия (90 сут) -> Свод -> импутация медианой.

    Импутация ставит theta=1, т.е. скважина остаётся на базовой опасности — ровно
    то, что она получает и без слоя. Это не «улучшение», а честное «нет данных».
    """
    with sqlite3.connect(WAREHOUSE_DIR / "pump2.db") as con:
        dm = pd.read_sql("SELECT well_key, dt, qliq FROM proc__daily_merged",
                         con, parse_dates=["dt"])
    dm["code"] = dm["well_key"].map(crosswalk.norm_well)
    dm["qliq"] = pd.to_numeric(dm["qliq"], errors="coerce")
    rec = dm[(dm["dt"] >= ASOF - pd.Timedelta(days=90)) & (dm["qliq"] > 0)].groupby("code")["qliq"].median()

    sv = pd.read_excel(crosswalk.resolve_prediction_workbook_path(), sheet_name="Свод")
    sv["code"] = sv["Скв."].map(crosswalk.norm_well)
    sv["qliq"] = pd.to_numeric(sv["Дебит жидк."], errors="coerce")
    sv["inst"] = pd.to_datetime(sv["Дата монтажа"], errors="coerce")
    svq = sv[sv["qliq"].notna()].sort_values("inst").groupby("code")["qliq"].last()

    live = live.copy()
    live["ql"] = live["code"].map(rec)
    live["ql_src"] = np.where(live["ql"].notna(), "телеметрия", "")
    m = live["ql"].isna()
    live.loc[m, "ql"] = live.loc[m, "code"].map(svq)
    live.loc[m & live["ql"].notna(), "ql_src"] = "Свод"
    med = float(live["ql"].median())
    m2 = live["ql"].isna()
    live.loc[m2, "ql"] = med
    live.loc[m2, "ql_src"] = "импутация (медиана)"
    return live


def main() -> None:
    pop = P.build(str(ASOF.date()))
    model = M.fit_mc_model(pop)
    print("модель Mc c=%g: w1=%.4f beta1=%.4f eta1=%.2f beta2=%.4f eta2=%.2f B50=%.0f"
          % (model.cut_days, model.w1, model.beta1, model.eta1, model.beta2, model.eta2, model.b50()))

    mc = P.select(pop, "Mc")
    live = mc[mc["end"].isna()].copy()
    live["age"] = (ASOF - live["install"]).dt.days.astype(float)
    live = _ql_by_code(live)

    ref_log = float(np.median(np.log1p(live["ql"])))
    theta = {r.code: M.ql_theta(float(r.ql), ref_log) for r in live.itertuples()}
    # импутированные -> ровно 1.0 (нет данных => нет сдвига)
    for r in live.itertuples():
        if r.ql_src.startswith("импутация"):
            theta[r.code] = 1.0

    hist = M.history_match(model, pop, first=HIST_FIRST, last=HIST_LAST)
    gate = M.c4_gate(model, pop, theta, first=HIST_FIRST, last=HIST_LAST)
    promoted = str(gate.loc[1, "вердикт"]).startswith("ПРОШЁЛ")
    use_theta = theta if promoted else None
    print("гейт C4:", gate.loc[1, "вердикт"])

    monthly, per_well = M.forecast(model, live, asof=ASOF, months=HORIZON_MONTHS,
                                   theta_by_code=use_theta)
    wells = M.well_table(model, live, asof=ASOF, theta_by_code=use_theta)
    # сценарный контраст: тот же прогноз со слоем Ql, всегда, для ревью
    monthly_ql, _ = M.forecast(model, live, asof=ASOF, months=HORIZON_MONTHS, theta_by_code=theta)
    cmp = monthly[["месяц", "ожидаемые подъёмы"]].rename(columns={"ожидаемые подъёмы": "базовая"})
    cmp["со слоем Ql"] = monthly_ql["ожидаемые подъёмы"].to_numpy()
    cmp["дельта"] = cmp["со слоем Ql"] - cmp["базовая"]

    summary = pd.DataFrame([
        ("Объект", "Мирнинский УН (Mc + Mr)"),
        ("Дата расчёта", str(ASOF.date())),
        ("Когорта", "монтажи с %s (стандартное правило)" % C.MC_INSTALL_COHORT_START),
        ("Часы", "календарные сутки (Свод «Наработка» ~= календарь, 0.986)"),
        ("Оценка", "ВСЕ причины подъёма (МРП), а не только отказ узла"),
        ("Отсечка c", "%g сут (вырезано ранних отказов: %d)" % (model.cut_days, model.n_cut)),
        ("Пробегов в подгонке", model.n_runs),
        ("Подъёмов", model.n_events),
        ("Цензурировано (живых)", model.n_censored),
        ("w1", round(model.w1, 4)),
        ("beta1", round(model.beta1, 4)),
        ("eta1", round(model.eta1, 2)),
        ("beta2", round(model.beta2, 4)),
        ("eta2", round(model.eta2, 2)),
        ("B50, сут", round(model.b50())),
        ("Живых насосов", len(live)),
        ("Ql: телеметрия", int((live["ql_src"] == "телеметрия").sum())),
        ("Ql: Свод", int((live["ql_src"] == "Свод").sum())),
        ("Ql: импутация", int(live["ql_src"].str.startswith("импутация").sum())),
        ("Опора Ql (медиана парка)", "%.3f  (%.0f м3/сут)" % (ref_log, np.expm1(ref_log))),
        ("Опора Ql из config (НЕ годится)", "%.3f  (%.0f м3/сут)" % (
            C.QL_HAZARD_FIELD_REF_LOG.get("Mc", np.nan),
            np.expm1(C.QL_HAZARD_FIELD_REF_LOG.get("Mc", np.nan)))),
        ("beta_Ql (landmark)", round(M.QL_BETA_LANDMARK, 4)),
        ("Слой Ql в базовом прогнозе", "ДА" if promoted else "НЕТ — гейт C4 не пройден, лист «Ql_сценарий» справочный"),
        ("Факт/модель за историю", round(float(hist["факт подъёмов"].sum() / hist["модель ожидает"].sum()), 3)),
    ], columns=["Параметр", "Значение"])

    out = results_dir("production_risk_mc_forecast")
    path = out / "tables" / "Прогноз_отказов_Mc.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Сводка", index=False)
        wells.to_excel(xl, sheet_name="Прогноз_отказов", index=False)
        hist.to_excel(xl, sheet_name="История_факт_модель", index=False)
        monthly.to_excel(xl, sheet_name="Прогноз_помесячно", index=False)
        per_well.to_excel(xl, sheet_name="Прогноз_по_скважинам", index=False)
        cmp.to_excel(xl, sheet_name="Ql_сценарий", index=False)
        gate.to_excel(xl, sheet_name="Ql_гейт_C4", index=False)
    print("записано:", path)
    print()
    print(hist.to_string(index=False))
    print()
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
