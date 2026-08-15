"""СВОДПРОГНОЗ (Vt) — прогноз отказов «Прогноз_ремонтов» по Верхнетирскому, два файла:

* **Сводпрогноз_Vt.xlsx**        — базовый прогноз отказов, без слоя Ql.
* **Сводпрогноз_hazard_Vt.xlsx** — тот же прогноз + слой Ql `θ = (Ql/Ql_ref)^β`.

Полный аналог `export_mc_svodprognoz.py`, с ОДНИМ принципиальным отличием: у Мирнинского
одна страта, у Верхнетирского — шесть, и они различаются не только масштабом, но и ФОРМОЙ
опасности. Поэтому прогноз считается по стратам и сшивается (`vt_forecast`), а книга несёт
лист `Страты` с резолвом каждой скважины.

Модели берутся из бандла `2026-07-20-vt-refit`:

* `Vt_sour_Pooled` / `Vt_sour_slb` / `Vt_sour_oth` — переподогнаны 2026-07-20 на `t_cal`
  (KS 0.074→0.050, 0.134→0.079, 0.143→0.080), все с beta ≈ 1.01-1.09;
* `Vt_nonsour_*` и `Vt_sour_brt` — реестр не побит переподгонкой, оставлены как были;
* строка `Vt_nonsour_oth` УДАЛЕНА (25 отказов, ни один фит не опустился ниже KS 0.246) —
  эти скважины резолвятся в `Vt_nonsour_Pooled`.

⚠ **Порядок отказов на Vt НЕ такой, как на Mc.** У Mc beta≈1.32 ⇒ «старые первыми»
(ρ = −0.99). У `Vt_sour_*` beta≈1.0 ⇒ порядка по возрасту НЕТ вовсе (накопленный долг
0.844 против 0.857 между возрастами 30 и 750 сут), у `Vt_nonsour_*` beta<1 ⇒ порядок
ОБРАТНЫЙ, первыми идут МОЛОДЫЕ. Это свойство подогнанной опасности, а не дефект
раскладки; см. `agents/analyses/mc_prognosis_debt_ordering_handoff.md` §5.

Запуск:  python scripts/run/export_vt_svodprognoz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.paths import results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk as cw
from analysis.workflows.production_risk import esp_population as P
from analysis.workflows.production_risk import esp_forecast as M
from analysis.workflows.production_risk import repair_compat
from analysis.workflows.production_risk import t0_covariates as t0
from analysis.workflows.production_risk import vt_forecast as V
from analysis.workflows.production_risk.survival import StrataModel

BUNDLE = "2026-07-20-vt-refit"
HIST_FIRST, HIST_LAST = "2024-01", "2026-06"
HORIZON_MONTHS = 18
QL_BETA = M.QL_BETA_LANDMARK   # ln(1.253); cross-fit PH на Vt_nonsour дал 0.246 — согласуется


def _theta_redistributive(ql_by_code: dict[str, float]) -> tuple[dict[str, float], float, float]:
    """θ = (Ql/Ql_ref)^β, ref = геосредний Ql популяции, перенормировка к mean=1.

    Перенормировка обязательна: без неё выпуклость θ (Йенсен) молча поднимает тотал.
    Слой ПЕРЕРАСПРЕДЕЛЯЕТ риск (кто раньше), а не добавляет отказов.
    """
    codes = [c for c, v in ql_by_code.items() if v is not None and np.isfinite(v) and v > 0]
    if not codes:
        return {}, float("nan"), float("nan")
    vals = np.array([ql_by_code[c] for c in codes], dtype=float)
    ref_log = float(np.mean(np.log1p(vals)))
    theta = {c: M.ql_theta(ql_by_code[c], ref_log, beta=QL_BETA) for c in codes}
    mean_raw = float(np.mean(list(theta.values())))
    return ({c: v / mean_raw for c, v in theta.items()},
            float(np.expm1(ref_log)), mean_raw)


def _plan_ql_by_code(codes: set[str]) -> dict[str, float]:
    pc = t0.plan_covariates(cw.load_plan())
    idx = {str(i).casefold(): i for i in pc.index}
    out: dict[str, float] = {}
    for c in codes:
        key = idx.get(str(c).casefold())
        if key is None:
            continue
        v = pc.loc[key, "ql"]
        if pd.notna(v) and v > 0:
            out[c] = float(v)
    return out


def _hist_ql_by_code(pop: pd.DataFrame) -> dict[str, float]:
    vt = pop[pop["field"] == "Vt"]
    d, _ = t0.build(vt)
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


def _add_intensity_chart(path: Path, intensity: pd.DataFrame, suffix: str) -> None:
    """Факт — ломаная с маркерами, модель — сплошная. Через ExcelWriter график не добавить."""
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
    chart.title = "Интенсивность отказов УЭЦН — Верхнетирский (по стратам%s): факт vs модель" % suffix
    chart.y_axis.title = "% от парка в работе"
    chart.x_axis.title = "Месяц"
    chart.y_axis.majorGridlines = ChartLines()
    chart.height, chart.width = 9, 28
    for col in (c_fact, c_model):
        chart.add_data(Reference(ws, min_col=col, min_row=1, max_row=n + 1), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=n + 1))
    s_fact, s_model = chart.series[0], chart.series[1]
    s_fact.graphicalProperties.line.width = 15000
    s_fact.graphicalProperties.line.dashStyle = "sysDash"
    s_fact.marker.symbol = "circle"
    s_fact.marker.size = 5
    s_fact.smooth = False
    s_model.graphicalProperties.line.width = 22000
    s_model.smooth = False
    ws.add_chart(chart, "J2")
    wb.save(path)


def _ordering_sheet(rows: pd.DataFrame, live: pd.DataFrame, resolve: pd.DataFrame) -> pd.DataFrame:
    """ρ(возраст, дата первого отказа) по стратам — обязательная проверка направления.

    На Mc это −0.99 («старые первыми»). На Vt направление ДРУГОЕ и различается по стратам,
    поэтому его печатаем в книгу, а не выводим по аналогии.
    """
    # Только ЖИВЫЕ насосы: у свежих (ВНС / БАЗА между насосами) возраст 0 по построению,
    # они бы искусственно усилили корреляцию, не неся сигнала о возрасте.
    d = rows.merge(resolve[["скв.", "страта"]], left_on="well_name", right_on="скв.", how="left")
    ages = dict(zip(live["code"], live["age"]))
    d["age"] = d["well_name"].map(ages)
    d = d[d["age"].notna() & d["first_event_day"].notna()]
    out = []
    for st, g in d.groupby("страта"):
        days = g["first_event_day"].to_numpy(float)
        ages = g["age"].to_numpy(float)
        if len(g) < 4 or np.ptp(days) == 0 or np.ptp(ages) == 0:
            # Постоянный вход: у всех насосов страты отказ в ОДИН месяц. Это не «нет
            # данных», а прямое следствие beta≈1 — долг у всех растёт одинаково, и
            # раскладка не может их различить.
            rho = float("nan")
            direction = ("порядка нет: отказы в один месяц (beta≈1)" if np.ptp(days) == 0
                         else "мало скважин")
        else:
            rho = float(spearmanr(ages, days).statistic)
            direction = ("старые первыми" if rho < -0.3 else
                         "молодые первыми" if rho > 0.3 else "порядка по возрасту нет")
        first_month = pd.Timestamp.fromordinal(int(np.min(days))).strftime("%Y-%m")
        last_month = pd.Timestamp.fromordinal(int(np.max(days))).strftime("%Y-%m")
        out.append({"страта": st, "скважин с отказом": len(g),
                    "ρ(возраст, дата)": None if np.isnan(rho) else round(rho, 3),
                    "направление": direction,
                    "первый отказ": first_month, "последний отказ": last_month})
    return pd.DataFrame(out).sort_values("страта")


def _rows_with_first_event(payload: dict) -> pd.DataFrame:
    rows = repair_compat.rows_frame(payload)
    firsts = []
    for r in payload["rows"]:
        ev = r.get("event_dates") or []
        firsts.append(pd.Timestamp(ev[0]).toordinal() if ev else np.nan)
    rows = rows.copy()
    rows["first_event_day"] = firsts
    return rows


def _build_and_write(path: Path, *, hazard: bool, pop, models, live, ent, fleet, asof,
                     resolve, theta_fc, theta_hist, ref_fc, gate_verdict) -> pd.DataFrame:
    th_fc = theta_fc if hazard else None
    th_hist = theta_hist if hazard else None

    note = ("Модели по стратам из бандла %s; sour-строки переподогнаны 2026-07-20 на t_cal."
            % BUNDLE)
    payload = V.repair_payload_multi(live, models, asof=asof, horizon_months=HORIZON_MONTHS,
                                     theta_by_code=th_fc, entrants=ent,
                                     fleet_by_month=fleet, model_note=note)
    hist = V.history_match_multi(models, pop, first=HIST_FIRST, last=HIST_LAST,
                                 theta_by_code=th_hist, fleet_by_month=fleet)
    fc, _ = V.forecast_multi(models, live, asof=asof, months=HORIZON_MONTHS,
                             theta_by_code=th_fc, fleet_by_month=fleet, entrants=ent)
    intensity = _intensity_sheet(hist, fc)

    if hazard:
        payload["notes"] += [
            "СВОДПРОГНОЗ (hazard): базовый прогноз + слой Ql θ=(Ql/Ql_ref)^%.4f." % QL_BETA,
            "Слой Ql ПЕРЕРАСПРЕДЕЛЯЕТ риск (кто раньше), тотал НЕ меняет (θ к mean=1).",
            "c4_gate (выигрыш MAE вне выборки): %s" % gate_verdict,
            "Ql плановый (ТМ-06), Ql_ref(прогноз)=%.1f м3/сут." % ref_fc,
        ]
    else:
        payload["notes"] += ["СВОДПРОГНОЗ (базовый): прогноз отказов по стратам, БЕЗ слоя Ql."]

    payload["notes"] += [
        "⚠ ОЧЕРЕДЬ БЛОЧНАЯ ПО СТРАТАМ. Долг растёт пропорционально риску страты, поэтому "
        "sour (≈17%/мес) забирает все ранние месяцы, а nonsour (≈5-8%/мес) идёт следом. "
        "По ожиданию это верно (sour втрое рискованнее), но «сначала все sour, потом все "
        "nonsour» — свойство ПРАВИЛА, а не предсказание: в реальности они чередуются.",
        "⚠ Внутри sour порядка по возрасту НЕТ (beta≈1, долг у всех одинаков) — очередь "
        "там определяется порядком сортировки, а не риском. Внутри nonsour порядок "
        "ОБРАТНЫЙ к Мирнинскому: первыми идут МОЛОДЫЕ насосы (beta<1).",
    ]

    written = repair_compat.write_excel(path, payload)
    rows = _rows_with_first_event(payload)
    ordering = _ordering_sheet(rows, live, resolve)
    with pd.ExcelWriter(written, engine="openpyxl", mode="a") as xl:
        intensity.to_excel(xl, sheet_name="Интенсивность отказов", index=False)
        hist.to_excel(xl, sheet_name="История_факт_модель", index=False)
        resolve.to_excel(xl, sheet_name="Страты", index=False)
        ordering.to_excel(xl, sheet_name="Порядок_отказов", index=False)
        if hazard:
            pd.DataFrame([{"well": c, "Ql_plan_theta": round(theta_fc[c], 4)}
                          for c in sorted(theta_fc)]).to_excel(xl, sheet_name="Слой_Ql", index=False)
    _add_intensity_chart(written, intensity, " + Ql" if hazard else "")

    print("  %-30s строк=%d событий=%d" % (path.name, len(rows), int(rows["event_count"].sum())))
    return ordering


def main() -> None:
    cfg = C.RunConfig()
    asof = pd.Timestamp(str(cfg.forecast_start))
    pop = P.build(str(cfg.forecast_start))

    codes = set(pop[pop["field"] == "Vt"]["code"])
    live0 = V.live_fleet(pop, asof)
    live_codes = set(live0["code"])
    entrants = M.plan_entrants(live_codes, codes, asof, HORIZON_MONTHS, prefixes=V.VT_PREFIXES)

    meta = M.plan_metadata(list(codes | set(entrants["code"])), asof)
    live = live0.merge(meta, on="code", how="left")
    live["license_area"] = V.VT_PLAN_FIELD
    ent = entrants.merge(meta, on="code", how="left")
    ent["license_area"] = V.VT_PLAN_FIELD
    ent["raw_id"] = ent["code"]
    ent["pred_source"] = "survival"
    if live.empty:
        raise SystemExit("Работающие насосы Vt не найдены — проверить популяцию")

    registry = StrataModel(bundle_date=BUNDLE)
    models, resolve = V.resolve_models(list(live["code"]) + list(ent["code"]),
                                       V.stratum_of(pop), registry)
    hist_models, _ = V.resolve_models(sorted(codes), V.stratum_of(pop), registry)
    fleet = M.active_fleet_by_month(codes=codes | set(entrants["code"]))

    print("Vt: пробегов=%d, живых насосов=%d, ВНС/БАЗА свежих=%d, строк формы=%d"
          % (len(pop[pop["field"] == "Vt"]), len(live), len(ent), len(live) + len(ent)))
    print(resolve["страта"].value_counts().to_string())

    fc_codes = live_codes | set(ent["code"])
    theta_fc, ref_fc, mean_fc = _theta_redistributive(_plan_ql_by_code(fc_codes))
    theta_hist, ref_hist, mean_hist = _theta_redistributive(_hist_ql_by_code(pop))
    gate = V.c4_gate_multi(hist_models, pop, theta_hist, first=HIST_FIRST, last=HIST_LAST,
                           fleet_by_month=fleet)
    verdict = str(gate.iloc[1]["вердикт"])
    print("Ql-слой: β=%.4f | прогноз Ql_ref=%.1f (mean θ raw=%.4f) | история Ql_ref=%.1f (raw=%.4f)"
          % (QL_BETA, ref_fc, mean_fc, ref_hist, mean_hist))
    print("c4_gate:", verdict)

    out = results_dir("production_risk_vt_svodprognoz")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    common = dict(pop=pop, models=models, live=live, ent=ent, fleet=fleet, asof=asof,
                  resolve=resolve, theta_fc=theta_fc, theta_hist=theta_hist,
                  ref_fc=ref_fc, gate_verdict=verdict)

    print("записано:")
    ordering = _build_and_write(out / "tables" / "Сводпрогноз_Vt.xlsx", hazard=False, **common)
    _build_and_write(out / "tables" / "Сводпрогноз_hazard_Vt.xlsx", hazard=True, **common)
    with pd.ExcelWriter(out / "tables" / "Сводпрогноз_hazard_Vt.xlsx",
                        engine="openpyxl", mode="a") as xl:
        gate.to_excel(xl, sheet_name="c4_gate_Ql", index=False)

    print()
    print("порядок отказов по стратам (на Mc для сравнения ρ = -0.99, «старые первыми»):")
    print(ordering.to_string(index=False))
    print()
    print("результаты →", out / "tables")


if __name__ == "__main__":
    main()
