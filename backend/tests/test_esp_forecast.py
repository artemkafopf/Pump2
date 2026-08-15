import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import esp_forecast as M


def _model(**kw):
    d = dict(w1=0.0, beta1=1.4, eta1=355.0, beta2=1.4, eta2=355.0,
             n_runs=100, n_events=60, n_censored=40, n_cut=0, cut_days=3.0)
    d.update(kw)
    return M.SurvivalModel(**d)


def _live(codes_ages, install=pd.Timestamp("2025-01-01")):
    return pd.DataFrame([{"code": c, "age": float(a), "install": install,
                          "ql": np.nan, "ql_src": ""} for c, a in codes_ages])


def test_survival_is_monotone_and_starts_at_one():
    m = _model()
    t = np.arange(0, 2000, 25.0)
    s = m.survival(t)
    assert s[0] == pytest.approx(1.0)
    assert np.all(np.diff(s) <= 1e-12)


def test_daily_fail_prob_matches_survival_ratio():
    m = _model()
    q = m.daily_fail_prob(500)
    a = 300
    expected = 1.0 - float(m.survival(a + 1)) / float(m.survival(a))
    assert q[a] == pytest.approx(expected, rel=1e-9)


def test_residual_median_halves_conditional_survival():
    """ОР — это медиана ОСТАТКА, а не возраст: S(age+ОР) == 0.5 * S(age)."""
    m = _model()
    for age in (0.0, 100.0, 400.0):
        orr = M.residual_median(m, age)
        assert float(m.survival(age + orr)) == pytest.approx(0.5 * float(m.survival(age)), rel=1e-6)


def test_nno_column_is_age_plus_resource():
    """ННО = возраст на сегодня + ОР, где ОР = смесь RMST/MRL (НЕ медиана)."""
    m = _model()
    asof = pd.Timestamp("2026-07-01")
    t = M.well_table(m, _live([("MC_1", 120), ("MC_2", 300)]), asof=asof)
    assert (t["ННО (возраст + ОР), сут"]
            == t["возраст на сегодня, сут"] + t["ОР до отказа, сут"]).all()
    # дата отказа должна согласоваться с ОР (по ИМЕНАМ колонок: позиционные r._N
    # разъезжаются, как только в таблицу добавляется столбец)
    for _, r in t.iterrows():
        got = (pd.Timestamp(r["дата отказа"]) - asof).days
        assert got == pytest.approx(r["ОР до отказа, сут"], abs=1.0)


def test_rmst_is_restricted_and_mrl_is_not():
    """RMST режется горизонтом tau и падает к нулю при age->tau; MRL — нет."""
    m = _model(eta1=400.0, eta2=400.0, beta1=1.4, beta2=1.4)
    m = M.SurvivalModel(**{**m.as_dict(), "tau": 475.0})
    assert m.rmst(0.0) < m.mrl(0.0)                  # RMST отрезает хвост за tau
    assert m.rmst(470.0) < 6.0                       # окно (470, 475] почти закрыто
    assert m.rmst(475.0) == 0.0
    assert m.mrl(470.0) > 50.0                       # MRL не знает про tau
    # beta>1 => MRL убывает с возрастом (износ)
    assert m.mrl(400.0) < m.mrl(0.0)


def test_wearout_shortens_residual_life_with_age():
    """beta>1 => остаточный ресурс УБЫВАЕТ с возрастом (в отличие от beta<1)."""
    m = _model(beta1=1.4, beta2=1.4)
    assert M.residual_median(m, 400.0) < M.residual_median(m, 0.0)


def test_ql_theta_is_one_at_reference_and_capped():
    ref = np.log1p(144.0)
    assert M.ql_theta(144.0, ref) == pytest.approx(1.0)
    assert M.ql_theta(144.0 * 1000, ref) == pytest.approx(np.exp(M.QL_BETA_LANDMARK * M.QL_CAP))
    assert M.ql_theta(0.0, ref) == pytest.approx(np.exp(-M.QL_BETA_LANDMARK * min(ref, M.QL_CAP)))
    assert M.ql_theta(np.nan, ref) == 1.0


def test_ql_theta_above_reference_raises_hazard():
    ref = np.log1p(144.0)
    assert M.ql_theta(600.0, ref) > 1.0
    assert M.ql_theta(30.0, ref) < 1.0


def test_forecast_renews_so_counts_can_exceed_one_per_well():
    """Обновление: за длинный горизонт один слот скважины даёт >1 подъёма."""
    m = _model(eta1=60.0, eta2=60.0)
    monthly, pw = M.forecast(m, _live([("MC_1", 0)]), asof=pd.Timestamp("2026-01-01"), months=18)
    assert float(pw["всего за горизонт"].iloc[0]) > 1.0


def test_history_match_does_not_renew():
    """История: даты монтажа известны -> обновления НЕТ. После подъёма пробег не
    порождает новый насос, поэтому ожидание в следующих месяцах строго ноль.

    (Сумма ожиданий по пробегу МОЖЕТ превышать 1: это конвенция Пуассона по
    пробего-месяцам, E[event|run-month]=mu, та же, что в Workstream C —
    Sum mu = 1775.6 против Sum event = 1520 на 39451 пробего-месяце. Прогноз же
    обновление делает, см. test_forecast_renews_...)
    """
    pop = pd.DataFrame([{
        "code": "MC_1", "install": pd.Timestamp("2024-01-01"),
        "end": pd.Timestamp("2024-03-15"), "tte": 74.0, "event": 1,
        "field": "Mc", "h2s_class": "nonsour", "contractor_group": "brt",
    }])
    m = _model(eta1=60.0, eta2=60.0)
    df = M.history_match(m, pop, first="2024-01", last="2026-06").set_index("месяц")
    assert df.loc["2024-04":, "модель ожидает"].sum() == 0.0
    assert df.loc["2024-04":, "в работе"].sum() == 0


def test_history_match_counts_the_pull_in_its_month():
    pop = pd.DataFrame([{
        "code": "MC_1", "install": pd.Timestamp("2024-01-01"),
        "end": pd.Timestamp("2024-03-15"), "tte": 74.0, "event": 1,
        "field": "Mc", "h2s_class": "nonsour", "contractor_group": "brt",
    }])
    df = M.history_match(_model(), pop, first="2024-01", last="2024-06")
    got = df.set_index("месяц")["факт"]
    assert got["2024-03"] == 1
    assert got.sum() == 1
    # после подъёма пробег не должен числиться в работе
    assert df.set_index("месяц")["в работе"]["2024-05"] == 0


def test_history_match_ignores_runs_installed_later():
    pop = pd.DataFrame([{
        "code": "MC_1", "install": pd.Timestamp("2025-06-01"), "end": pd.NaT,
        "tte": 100.0, "event": 0, "field": "Mc", "h2s_class": "nonsour", "contractor_group": "brt",
    }])
    df = M.history_match(_model(), pop, first="2024-01", last="2025-12")
    assert df.set_index("месяц")["в работе"]["2024-05"] == 0
    assert df.set_index("месяц")["модель ожидает"]["2024-05"] == 0.0


def test_history_fleet_counts_wells_not_runs():
    """«В работе» = СКВАЖИНЫ. Скважина с подъёмом и переустановкой в одном месяце
    даёт 2 пробега — знаменатель обязан остаться 1, иначе % несопоставим с прогнозом
    (там парк = len(live) по скважинам)."""
    pop = pd.DataFrame([
        {"code": "MC_1", "install": pd.Timestamp("2024-01-01"),
         "end": pd.Timestamp("2024-03-10"), "tte": 69.0, "event": 1,
         "field": "Mc", "h2s_class": "nonsour", "contractor_group": "brt"},
        {"code": "MC_1", "install": pd.Timestamp("2024-03-12"), "end": pd.NaT,
         "tte": 500.0, "event": 0,
         "field": "Mc", "h2s_class": "nonsour", "contractor_group": "brt"},
    ])
    df = M.history_match(_model(), pop, first="2024-03", last="2024-03")
    assert int(df.iloc[0]["в работе"]) == 1, "две записи одной скважины -> парк 1, не 2"


def test_c4_gate_rejects_a_noise_sized_gain():
    """Гейт не должен штамповать: выигрыш MAE в пределах шума -> НЕ ПРОШЁЛ.

    theta ~ 1 у всех скважин => слой почти ничего не меняет => промотировать нельзя.
    """
    pop = pd.DataFrame([
        {"code": "MC_%d" % i, "install": pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i),
         "end": pd.Timestamp("2025-06-01") + pd.Timedelta(days=20 * i), "tte": 200.0 + i,
         "event": 1, "field": "Mc", "h2s_class": "nonsour", "contractor_group": "brt"}
        for i in range(12)
    ])
    g = M.c4_gate(_model(), pop, {("MC_%d" % i): 1.0005 for i in range(12)},
                  first="2024-01", last="2026-06")
    assert str(g.loc[1, "вердикт"]).startswith("НЕ ПРОШЁЛ")


def _live_dates(codes_ages, asof=pd.Timestamp("2026-07-01")):
    return pd.DataFrame([{"code": c, "age": float(a),
                          "install": asof - pd.Timedelta(days=a),
                          "ql": np.nan, "ql_src": ""} for c, a in codes_ages])


def test_payload_form_is_wellformed():
    m = _model(eta1=200.0, eta2=200.0, beta1=1.4, beta2=1.4)
    live = _live_dates([("MC_1", 50), ("MC_2", 300), ("MC_3", 150)])
    asof = pd.Timestamp("2026-07-01")
    pa = M.repair_payload(live, m, asof=asof, horizon_months=18, method="statistical")
    for p in (pa,):
        assert len(p["rows"]) == 3
        for row in p["rows"]:
            assert len(row["statuses"]) == len(p["dates"])
            assert set(row["statuses"]) <= {0, 1}
            # «Вероятностный прогноз» >= «Факт ННО» (полная наработка, не остаток)
            if row["probabilistic_nno"] is not None:
                assert row["probabilistic_nno"] >= row["actual_nno"]


def test_statistical_places_failures_on_the_first_of_the_month():
    """Модель даёт МЕСЯЧНУЮ вероятность отказа — день внутри месяца не определён,
    поэтому каждый прогнозный отказ ставится на 1-е число (кроме месяца asof, где
    сетка начинается позже). Раньше день размазывался обратной CDF — это была
    нужда РАСПИСАНИЯ РЕМОНТОВ, а здесь прогнозируются ОТКАЗЫ."""
    m = _model(eta1=120.0, eta2=120.0, beta1=1.3, beta2=1.3)
    live = _live_dates([("MC_%d" % i, 30 + 15 * i) for i in range(20)])
    asof = pd.Timestamp("2026-07-01")
    p = M.repair_payload(live, m, asof=asof, horizon_months=18, method="statistical")
    dates = [pd.Timestamp(d) for row in p["rows"] for d in row["event_dates"]]
    assert dates, "должен быть хотя бы один прогнозный отказ"
    assert all(d.day == 1 for d in dates), \
        "все отказы должны стоять на 1-е число: %s" % sorted({str(d.date()) for d in dates if d.day != 1})


def test_statistical_preserves_fleet_event_total():
    """Перенос остатка: Σ событий == ΣE минус хвостовой остаток (0 <= carry < 1).

    НЕ round(ΣE): дробный хвост последнего месяца не может стать событием, поэтому
    сумма = floor по накоплению. Расхождение с ΣE строго меньше 1 события.
    """
    m = _model(eta1=120.0, eta2=120.0, beta1=1.3, beta2=1.3)
    live = _live_dates([("MC_%d" % i, 30 + 15 * i) for i in range(20)])
    asof = pd.Timestamp("2026-07-01")
    monthly, _ = M.forecast(m, live, asof=asof, months=18)
    p = M.repair_payload(live, m, asof=asof, horizon_months=18, method="statistical")
    zeros = sum(r["statuses"].count(0) for r in p["rows"])
    expected = float(monthly["ожидаемые события"].sum())
    assert 0 <= expected - zeros < 1.0, "потеряно/добавлено больше одного события"


def test_monthly_counts_follow_the_carry_rule():
    """Помесячно: n_m = floor(E_m + carry), остаток переносится в следующий месяц."""
    m = _model(eta1=120.0, eta2=120.0, beta1=1.3, beta2=1.3)
    live = _live_dates([("MC_%d" % i, 30 + 15 * i) for i in range(20)])
    asof = pd.Timestamp("2026-07-01")
    monthly, pw = M.forecast(m, live, asof=asof, months=18)
    got = M.allocate_events(monthly, pw)
    by_month = {}
    for periods in got.values():
        for per in periods:
            by_month[str(per)] = by_month.get(str(per), 0) + 1
    carry = 0.0
    for _, row in monthly.iterrows():
        total = float(row["ожидаемые события"]) + carry
        n = int(np.floor(total))
        carry = total - n
        assert by_month.get(str(row["месяц"]), 0) == n, "месяц %s" % row["месяц"]


def test_events_go_to_the_riskiest_wells_first():
    """КТО: старые (рискованные) насосы получают отказ раньше молодых.

    Прежняя раскладка (обратная CDF + хеш-фаза) давала корреляцию возраст↔дата
    -0.177 — шум, из-за чего свежий насос ремонтировался раньше трёхлетнего.
    """
    m = _model(eta1=300.0, eta2=300.0, beta1=1.4, beta2=1.4)   # beta>1 => старые рискованнее
    live = _live_dates([("MC_%d" % i, 20 + 40 * i) for i in range(15)])
    asof = pd.Timestamp("2026-07-01")
    monthly, pw = M.forecast(m, live, asof=asof, months=18)
    got = M.allocate_events(monthly, pw)
    first = {c: min(p.ordinal for p in ps) for c, ps in got.items() if ps}
    ages = {str(r.code): float(r.age) for r in live.itertuples()}
    pairs = sorted((ages[c], first[c]) for c in first)
    corr = pd.Series([p[0] for p in pairs]).corr(
        pd.Series([p[1] for p in pairs]), method="spearman")
    assert corr < -0.5, "дата должна убывать с возрастом, получено %.3f" % corr




def _entrants(specs, asof=pd.Timestamp("2026-07-01")):
    """specs: [(code, месяц_ввода)] -> кадр как у M.plan_entrants."""
    rows = []
    for code, month in specs:
        entry = pd.Period(month, "M").start_time
        rows.append({"code": code, "entry": entry,
                     "start_day": max(0, int((entry - asof).days)),
                     "age": float(max(0, int((asof - entry).days)))})
    return pd.DataFrame(rows, columns=["code", "entry", "start_day", "age"])


def test_entrant_contributes_nothing_before_it_is_commissioned():
    """ВНС не может отказать до месяца ввода: скважины ещё нет."""
    m = _model(eta1=100.0, eta2=100.0, beta1=1.2, beta2=1.2)
    live = _live_dates([("MC_1", 400)])
    asof = pd.Timestamp("2026-07-01")
    ent = _entrants([("MC_NEW", "2026-10")], asof)
    _, pw = M.forecast(m, live, asof=asof, months=18, entrants=ent)
    row = pw[pw["скв."] == "MC_NEW"].iloc[0]
    for mo in ["2026-07", "2026-08", "2026-09"]:
        assert row[mo] == 0.0, "%s: ВНС отказала до ввода" % mo
    assert row["2026-10"] > 0.0, "в месяц ввода отказы обязаны появиться"


def test_entrant_enters_at_age_zero_and_carries_infant_risk():
    """Новый насос входит в возрасте 0, а не наследует возраст парка.

    У cal c0 k2 масса дня-0 ~3%: если бы ВНС входила «взрослой», детская смертность
    молча терялась бы, и прогноз занижал бы отказы новых скважин.
    """
    m = _model(w1=0.05, eta1=1.0, beta1=0.9, eta2=500.0, beta2=1.3)   # заметная масса дня-0
    live = _live_dates([("MC_1", 400)])
    asof = pd.Timestamp("2026-07-01")
    ent = _entrants([("MC_NEW", "2026-08")], asof)
    _, pw = M.forecast(m, live, asof=asof, months=18, entrants=ent)
    first = float(pw[pw["скв."] == "MC_NEW"].iloc[0]["2026-08"])
    assert first > 0.04, "месяц ввода обязан нести массу дня-0, получено %.4f" % first


def test_entrants_get_their_share_of_events():
    """Сетка обязана отдать ВНС столько событий, сколько им приписывает прогноз.

    Прежнее правило «событие самому рискованному по возрасту» морило новые скважины:
    старый насос всегда выигрывал очередь (6.4%/мес при 730 сут против 4.7% при 0), и
    49 ВНС получали 9 событий против 23.2 ожидаемых. Долг из per_well это чинит.

    Здесь ожидание каждой ВНС > 1, поэтому проверяется именно правило. На живом парке
    Mc ожидания меньше 1 (0.47 у ВНС), и там сетка упирается в целочисленность — см.
    `allocate_events`; это ограничение арифметики, а не правила.
    """
    m = _model(eta1=300.0, eta2=300.0, beta1=1.3, beta2=1.3)
    asof = pd.Timestamp("2026-07-01")
    live = _live_dates([("MC_%d" % i, 500 + 20 * i) for i in range(10)])
    ent = _entrants([("NEW_%d" % i, "2026-08") for i in range(10)], asof)
    monthly, pw = M.forecast(m, live, asof=asof, months=18, entrants=ent)
    got = M.allocate_events(monthly, pw)
    new = {c for c in got if c.startswith("NEW_")}
    exp_new = float(pw[pw["новая"]]["всего за горизонт"].sum())
    got_new = sum(len(got[c]) for c in new)
    assert abs(got_new - exp_new) <= 1.5, (
        "ВНС выдано %d против ожидаемых %.1f" % (got_new, exp_new))


def test_allocation_matches_per_well_expectation():
    """Выданное каждому насосу сходится к его СОБСТВЕННОМУ ожиданию из forecast.

    Это то свойство, которого не было у раскладки со своим счётом возраста: она
    трекала возраст точкой, forecast — распределением, и две динамики расходились.
    """
    m = _model(eta1=250.0, eta2=250.0, beta1=1.25, beta2=1.25)
    asof = pd.Timestamp("2026-07-01")
    live = _live_dates([("MC_%d" % i, 50 + 90 * i) for i in range(12)])
    monthly, pw = M.forecast(m, live, asof=asof, months=18)
    got = M.allocate_events(monthly, pw)
    exp = pw.set_index("скв.")["всего за горизонт"]
    for code, periods in got.items():
        assert abs(len(periods) - float(exp[code])) < 1.0, (
            "%s: выдано %d против ожидания %.2f" % (code, len(periods), exp[code]))


def test_deterministic_method_is_gone():
    """Детерминированный подход удалён — остался только статистический."""
    assert not hasattr(M, "deterministic_event_days")
    m = _model()
    with pytest.raises(ValueError):
        M.repair_payload(_live_dates([("MC_1", 50)]), m,
                         asof=pd.Timestamp("2026-07-01"), horizon_months=6,
                         method="deterministic")


def test_resource_is_the_blend_and_never_collapses_to_zero():
    """Отображаемый ОР = w·RMST+(1-w)·MRL: у старых насосов не 0 (как чистый RMST)
    и не полный MRL — «дай доработать», а не «руби сегодня» / «жди год»."""
    m = _model(eta1=400.0, eta2=400.0, beta1=1.4, beta2=1.4)
    m = M.SurvivalModel(**{**m.as_dict(), "tau": 475.0})
    old = 700.0
    assert m.rmst(old) == 0.0                      # чистый RMST схлопывается за tau
    assert m.resource(old) > 0.0                   # смесь — нет
    assert m.resource(old) < m.mrl(old)            # но и не весь MRL
    w = M.RESOURCE_BLEND_W
    assert m.resource(old) == pytest.approx((1 - w) * m.mrl(old), rel=1e-6)


def test_history_counts_pre_2024_runs_that_failed_in_window():
    """ФАКТ — наблюдение, когорта 2024+ к нему НЕ применяется: пробег, смонтированный
    в 2023 и отказавший в 2024, отказал по-настоящему. Фильтр съедал 3 из 47 отказов."""
    pop = pd.DataFrame([{
        "code": "MC_9", "install": pd.Timestamp("2023-06-01"),
        "end": pd.Timestamp("2024-02-10"), "tte": 254.0, "event": 1,
        "field": "Mc", "h2s_class": "nonsour", "contractor_group": "brt",
    }])
    df = M.history_match(_model(), pop, first="2024-02", last="2024-02")
    assert int(df.iloc[0]["факт"]) == 1, "отказ 2024-02 обязан попасть в факт"
    assert int(df.iloc[0]["в работе"]) == 1
    # а с cohort_only=True — исчезает (так было раньше, отсюда расхождение с релизами)
    df2 = M.history_match(_model(), pop, first="2024-02", last="2024-02", cohort_only=True)
    assert int(df2.iloc[0]["факт"]) == 0


def test_wells_without_events_get_nno_beyond_the_horizon():
    """Скважинам без события ННО заполняется как age+MRL: отказ ложится ЗА горизонт,
    что согласуется с отсутствием нуля в сетке. Смесь w=0.9 сюда не годится —
    она дала бы ННО ВНУТРИ горизонта и противоречила сетке."""
    # долгоживущая модель => E на скважину заметно < 1 => событий меньше, чем скважин
    m = _model(eta1=1500.0, eta2=1500.0, beta1=1.4, beta2=1.4)
    m = M.SurvivalModel(**{**m.as_dict(), "tau": 475.0})
    live = _live_dates([("MC_%d" % i, 20 + 30 * i) for i in range(12)])
    asof = pd.Timestamp("2026-07-01")
    p = M.repair_payload(live, m, asof=asof, horizon_months=18)
    horizon_days = (pd.Timestamp("2027-12-31") - asof).days
    empty = [r for r in p["rows"] if not r["event_dates"]]
    assert empty, "нужна хотя бы одна скважина без события"
    for r in empty:
        assert r["probabilistic_nno"] is not None, "ННО должен быть заполнен"
        assert r["probabilistic_nno"] > r["actual_nno"] + horizon_days,             "ННО обязан лежать ЗА горизонтом, иначе противоречит пустой строке в сетке"
