"""TTF viewer — интерактивный биннинг фактической панели Свода по любой рабочей характеристике.

    streamlit run streamlit_apps/ttf_viewer.py

Ось X выбирается: частота, дебит жидкости, номинальная производительность, коэффициент
подачи, газожидкостный фактор, обводнённость, забойное давление, Pзаб/Pнас. Всё остальное —
сборка сетки, статистики, тренд, перестановочный тест, панель с цензурированием — от выбора
не зависит и работает одинаково, потому что ни одна из этих машинок не знает, на какую ось
смотрит.

Популяция тоже выбирается: любое месторождение, набравшее порог отказов, плюс «Весь фонд» —
все месторождения вместе, плюс половины Vt «кислый / некислый» — те же стратумы, на которых
подогнана модель Vt, по well-level классу сернистости. На объединённой панели рисуется состав
фонда по бинам, потому что месторождения различаются по сроку службы в разы и форма кривой
там может оказаться просто сменой состава.

Галка «в долях от опорного бина» переводит график в отношения: каждая статистика делится на
своё значение в бине, куда попало опорное значение по оси X (по умолчанию медиана
характеристики, поле рядом с галкой), Y(опора) = 1. Так сравнимы панели с разным уровнем —
сутки против приведённых суток, факт против KM, кислый Vt против некислого.

Три вещи, ради которых он существует:

* **границы бинов — это данные, а не настройка.**  Их можно двигать, сливать и делить, и
  форма кривой должна пережить перекройку — иначе это не форма, а артефакт сетки.  Исключение
  — номинальная производительность: ось дискретная, сетка по умолчанию даёт **один бин на
  типоразмер**, правило «доращивать до n» там не применяется, а маркер стоит на самом
  значении. Соседние размеры — разные насосы, и ступенька между ними и есть форма оси;
* **среднее и медиана расходятся**, и расхождение здесь разложено на три причины —
  скошенность (геом. среднее), шум (бутстрэп + перестановочный тест) и отбор (вкладка с
  цензурированием);
* сборка по умолчанию идёт по правилу оператора: старт с якорного бина, рост наружу,
  расширение пока в бин не попадёт целевое число отказов — но не шире предела ширины.

Снятие подогнанного слоя доступно **по каждому участку недр**: модель «Пофондовая v5»
несёт свой θ_Qном для Ya, Vt (кислый/некислый), Az, Ic, Au, Mc, а участок без собственной
строки уходит на Fleet — пул по всему фонду. Это те же два блока параметров, которыми прошиты
калькуляторы, поэтому вьювер и книги не могут разойтись в том, что такое слой Qном. И это
единственная модель, применимая к «Весь фонд»: все слои Qном пинятся к ОДНОМУ опорному насосу
Qном 250, так что объединённая панель приводится к общей точке, а не к разным нулям.

Вся расчётная часть — в ``analysis.workflows.production_risk.freq_bins``; здесь только UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.workflows.production_risk import freq_bins as FB   # noqa: E402

st.set_page_config(page_title="TTF viewer — наработка по характеристикам", layout="wide")

#: Проверено валидатором палитры: светлота, цветность и различимость при CVD.
#: Синий и фиолетовый лежат в «полосе 6–8» по протанопии, поэтому у каждой серии ещё и свой
#: пунктир и свой маркер — цвет здесь никогда не единственный признак.
STYLE = {
    "mean":     dict(color="#E8762C", dash="solid", symbol="circle"),
    "median":   dict(color="#1F6F45", dash="solid", symbol="diamond"),
    "geomean":  dict(color="#1B6DB5", dash="dash", symbol="square"),
    "trimmed":  dict(color="#A0439B", dash="dot", symbol="triangle-up"),
    "km_median": dict(color="#1F6F45", dash="solid", symbol="diamond"),
    "rmst":     dict(color="#1B6DB5", dash="solid", symbol="square"),
}
BAR_COLOR, BAR_THIN = "#9DB4CE", "#DDE4EC"
INK, INK2, INK3 = "#1A1A1A", "#4A4A4A", "#8A8A8A"

#: Палитра для состава фонда по бинам: четыре проверённых цвета на четыре крупнейших
#: месторождения, серый — на «прочие». Больше четырёх категорий цветом не различить.
MIX_COLORS = ["#E8762C", "#1F6F45", "#1B6DB5", "#A0439B", "#9DB4CE"]


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="чтение Свода …")
def fact(field: str, variant: str, source: str) -> pd.DataFrame:
    g = FB.load_fact(field, variant=variant, source=source)
    return g.assign(_n_field=g.attrs["n_field"])


@st.cache_data(show_spinner="сборка популяции с цензурированием (долго) …")
def censored(field: str, source: str) -> pd.DataFrame:
    g = FB.load_censored(field, source=source)
    return g.assign(_n_field=g.attrs["n_field"], _n_open=g.attrs["n_open"])


@st.cache_data(show_spinner="бутстрэп …")
def stats_for(x: tuple, ttf: tuple, edges: tuple, trim: float, n_boot: int,
              ci: float) -> pd.DataFrame:
    return FB.bin_stats(np.array(x), np.array(ttf), list(edges),
                        trim=trim, n_boot=n_boot, ci=ci)


@st.cache_data(show_spinner="Каплан–Мейер по бинам …")
def km_for(x: tuple, t: tuple, ev: tuple, edges: tuple, n_boot: int,
           ci: float) -> pd.DataFrame:
    return FB.km_bin_stats(np.array(x), np.array(t), np.array(ev), list(edges),
                           n_boot=n_boot, ci=ci)


@st.cache_data(show_spinner="перестановочный тест …")
def flatness(x: tuple, ttf: tuple, edges: tuple, stat: str, n_perm: int,
             min_n: int, trim: float):
    return FB.permutation_flatness(np.array(x), np.array(ttf), list(edges),
                                   stat=stat, n_perm=n_perm, min_n=min_n, trim=trim)


def trends(tab: pd.DataFrame, cols, form: str, min_n: int, prop,
           count_col: str = "n") -> dict:
    """Один тренд на статистику; None там, где надёжных бинов меньше, чем параметров.

    ``scale``/``ref`` берутся у характеристики: парабола в СЫРЫХ м³/сут на оси 5…2000 —
    это не кривая по данным, а прямая с ошибкой округления.
    """
    out = {}
    for c in cols:
        f = FB.fit_trend(tab, stat=c, form=form, min_n=min_n, count_col=count_col,
                         scale=prop.scale, ref=prop.anchor[0])
        if f is not None:
            out[c] = f
    return out


# ---------------------------------------------------------------------------
# Состояние: список границ + стек отмены
# ---------------------------------------------------------------------------
def set_edges(new, prop, *, remember: bool = True) -> None:
    new = FB.sanitize_edges(new, min_width=prop.list_min_width)
    if remember and st.session_state.get("edges") is not None:
        st.session_state.undo = (st.session_state.get("undo", []) + [st.session_state.edges])[-40:]
    st.session_state.edges = new


def build_catalog(x: np.ndarray, prop) -> list[float]:
    """Сетка дискретной оси: один бин на типоразмер, правило «доращивать до n» не работает."""
    return FB.catalog_edges(x, bounds=prop.bounds, scale=prop.scale,
                            min_width=prop.list_min_width, tol=prop.min_width)


def build_rule(x: np.ndarray, prop, target_n: int, max_width: float) -> list[float]:
    return FB.adaptive_edges(x, anchor=prop.anchor, min_n=int(target_n), bounds=prop.bounds,
                             step=prop.step, max_width=float(max_width), scale=prop.scale,
                             min_width=prop.list_min_width, sig=prop.sig)


def build_default(x: np.ndarray, prop, target_n: int, max_width: float) -> list[float]:
    """Сборка по умолчанию — каталог на дискретной оси, правило оператора на остальных."""
    return (build_catalog(x, prop) if prop.discrete
            else build_rule(x, prop, target_n, max_width))


def with_centres(tab: pd.DataFrame, x: np.ndarray, edges, prop) -> pd.DataFrame:
    """На дискретной оси точка стоит на самом значении, а не в середине скобки."""
    return FB.observed_centres(tab, x, edges) if prop.discrete else tab


def ensure_edges(x: np.ndarray, key: str, prop, target_n: int,
                 max_width: float) -> list[float]:
    """Пересборка при первом заходе, при смене панели И при смене параметров сборки.

    ``key`` включает характеристику, целевое n и предел ширины, поэтому правка любого из них
    немедленно перестраивает сетку — это параметры ПОСТРОЕНИЯ, и оставлять после них старые
    границы значило бы показывать сетку, которой эти настройки не соответствуют.  Ручные
    правки при этом теряются, но не безвозвратно: прежние границы ложатся в стек отмены.
    """
    if st.session_state.get("edges_key") != key or st.session_state.get("edges") is None:
        prev = st.session_state.get("edges")
        st.session_state.edges_key = key
        st.session_state.undo = ([] if prev is None
                                 else (st.session_state.get("undo", []) + [prev])[-40:])
        st.session_state.edges = build_default(x, prop, target_n, max_width)
    return st.session_state.edges


# ---------------------------------------------------------------------------
# Графика
# ---------------------------------------------------------------------------
def _add_stat(fig, tab: pd.DataFrame, col: str, label: str, *, target_n: int, prop,
              row: int = 1, show_ci: bool = True, ratio: bool = False) -> None:
    """Точки в серединах бинов, соединённые линией, и лента интервала вокруг них.

    Именно линия, а не ступенька по ширине бина: ступенька при неравных бинах читается
    как столбчатая диаграмма и спорит со столбцами счётчика снизу. Ширина бина никуда не
    девается — она видна в нижней панели и во всплывающей подсказке.

    Маркеры тонких бинов рисуются полыми: точка на кривой есть, но она не выдаёт себя за
    такую же надёжную, как остальные.
    """
    d = tab[(tab["n"] > 0) & tab[col].notna()].sort_values("f_mid")
    if d.empty:
        return
    s = STYLE[col]
    lo_c, hi_c = f"{col}_lo", f"{col}_hi"
    if show_ci and lo_c in d.columns and d[lo_c].notna().any():
        band = d[d[lo_c].notna() & d[hi_c].notna()]
        fig.add_trace(go.Scatter(
            x=list(band["f_mid"]) + list(band["f_mid"])[::-1],
            y=list(band[hi_c]) + list(band[lo_c])[::-1],
            fill="toself", fillcolor=s["color"], opacity=0.09, line=dict(width=0),
            hoverinfo="skip", showlegend=False), row=row, col=1)
    thin = (d["n"] < target_n).to_numpy()
    fig.add_trace(go.Scatter(
        x=d["f_mid"], y=d[col], mode="lines+markers", name=label,
        line=dict(color=s["color"], width=2.4, dash=s["dash"]),
        marker=dict(color=np.where(thin, "white", s["color"]), size=9, symbol=s["symbol"],
                    line=dict(color=s["color"], width=1.8)),
        customdata=np.column_stack([d["f_lo"], d["f_hi"], d["n"]]),
        hovertemplate=(f"<b>{label}</b>: "
                       + ("×%{y:.3f}" if ratio else "%{y:.0f} сут") + "<br>"
                       f"бин %{{customdata[0]:{prop.fmt[1:]}}}–%{{customdata[1]:{prop.fmt[1:]}}}"
                       f" {prop.unit} · n = %{{customdata[2]:.0f}}<extra></extra>")),
        row=row, col=1)


def _add_trend(fig, fit, label: str, col: str, prop, row: int = 1,
               divisor: float = 1.0) -> None:
    """Тренд рисуется ТОЛЬКО на своём носителе — там, где были надёжные бины.

    Полином за пределами подгонки уходит куда угодно (кубика — особенно быстро), и линия,
    дотянутая до края оси, читалась бы как предсказание, которого в данных нет.

    ``divisor`` — опорное значение в режиме отношений. Подгонка при этом остаётся ТОЙ ЖЕ:
    она идёт по логарифму статистики, деление на константу — сдвиг свободного члена, форма
    и R² не меняются. Поэтому делится результат, а не пересчитывается тренд.
    """
    grid = (np.geomspace(*fit.support, 241) if prop.scale == "log"
            else np.linspace(*fit.support, 241))
    ratio = divisor != 1.0
    fig.add_trace(go.Scatter(
        x=grid, y=fit.predict(grid) / divisor, mode="lines", name=f"тренд · {label}",
        line=dict(color=STYLE[col]["color"], width=1.8, dash="longdash"), opacity=0.85,
        hovertemplate=(f"тренд {label}: " + ("×%{y:.3f}" if ratio else "%{y:.0f} сут")
                       + f"<br>%{{x:.4g}} {prop.unit}"
                       f"<extra>R²={fit.weighted_r2:.2f}</extra>")), row=row, col=1)


def _count_panel(fig, d: pd.DataFrame, count_col: str, target_n: int, prop,
                 row: int = 2) -> None:
    """Счётчик отказов — прямоугольниками, а не ``go.Bar``.

    У столбца ширина задаётся одним числом в единицах оси, и на логарифмической оси это
    не работает: бин 5–11 и бин 1000–2000 занимают там одинаковую долю экрана, но в
    единицах данных различаются в двести раз. Полигон строится по СВОИМ границам, поэтому
    на любой шкале столбец точно накрывает свой бин — ровно те же координаты, что у точки
    сверху.
    """
    thin = (d[count_col] < target_n).to_numpy()
    for mask, color in ((~thin, BAR_COLOR), (thin, BAR_THIN)):
        sub = d[mask]
        if sub.empty:
            continue
        xs: list = []
        ys: list = []
        for lo, hi, n in zip(sub["f_lo"], sub["f_hi"], sub[count_col]):
            a, b = ((lo * 1.012, hi / 1.012) if prop.scale == "log"
                    else (lo + (hi - lo) * 0.02, hi - (hi - lo) * 0.02))
            xs += [a, a, b, b, None]
            ys += [0, n, n, 0, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=color, mode="lines",
                                 line=dict(color="white", width=1), hoverinfo="skip",
                                 showlegend=False), row=row, col=1)
    # подпись стоит там же, где точка сверху (``f_mid``), а не в геометрической середине
    # скобки: на дискретной оси точка сидит на типоразмере, и разъезд читался бы как
    # «столбец не от этой точки»
    fig.add_trace(go.Scatter(
        x=d["f_mid"], y=d[count_col], mode="text", text=d[count_col].astype(int),
        textposition="top center", textfont=dict(size=10, color=INK2), showlegend=False,
        customdata=np.column_stack([d["f_lo"], d["f_hi"]]),
        hovertemplate=(f"бин %{{customdata[0]:{prop.fmt[1:]}}}–"
                       f"%{{customdata[1]:{prop.fmt[1:]}}} {prop.unit}<br>"
                       "%{y:.0f}<extra></extra>")), row=row, col=1)


def bins_figure(tab: pd.DataFrame, cols: list[str], labels: dict, *, target_n: int, prop,
                ylab: str, count_col: str = "n", count_lab: str = "отказов в бине",
                show_ci: bool = True, title: str = "", fits: dict | None = None,
                ratio_ref: dict | None = None) -> go.Figure:
    """``ratio_ref`` — {статистика: опорное значение в сутках}: панель уже нормирована.

    Таблица приходит поделённой (``freq_bins.normalize_to_bin``), а этот словарь нужен
    ЗДЕСЬ только для тренда — он подгоняется на исходных сутках, чтобы «оптимум» рядом с
    графиком оставался в сутках, и делится на ту же константу при отрисовке.
    """
    ratio = ratio_ref is not None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    for c in cols:
        _add_stat(fig, tab, c, labels[c], target_n=target_n, prop=prop, show_ci=show_ci,
                  ratio=ratio)
    for c in cols:
        # статистику, которой опорный бин не дал (пустой бин, неопределённая медиана KM),
        # нормировать не на что — точки уже пустые, и тренд к ним рисовать нечестно
        if fits and c in fits and (not ratio or c in ratio_ref):
            _add_trend(fig, fits[c], labels[c], c, prop,
                       divisor=float(ratio_ref[c]) if ratio else 1.0)

    d = tab[tab["n"] > 0]
    _count_panel(fig, d, count_col, target_n, prop)

    for r in (1, 2):
        fig.add_vline(x=prop.anchor[0], line=dict(color=INK3, width=1, dash="dot"),
                      row=r, col=1)
    if ratio:
        fig.add_hline(y=1.0, line=dict(color=INK3, width=1, dash="dot"), row=1, col=1)
    fig.update_yaxes(title_text=ylab, row=1, col=1, rangemode="tozero",
                     gridcolor="#ECECEC", zeroline=False)
    # запас сверху, иначе подпись самого высокого столбца упирается в край панели
    fig.update_yaxes(title_text=count_lab, row=2, col=1, gridcolor="#ECECEC",
                     zeroline=False, range=[0, float(d[count_col].max()) * 1.20])
    xkw = dict(gridcolor="#F4F4F4")
    if prop.scale == "log":
        xkw["type"] = "log"
    elif prop.key == "freq":
        xkw["dtick"] = 2
    fig.update_xaxes(title_text=prop.axis_title(), row=2, col=1, **xkw)
    fig.update_xaxes(row=1, col=1, **xkw)
    fig.update_layout(
        height=640, template="plotly_white", title=title, bargap=0.05,
        hovermode="closest", margin=dict(l=60, r=20, t=60 if title else 30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=12)))
    return fig


def fmt(v: float, prop) -> str:
    return (prop.fmt % v) if abs(v) >= 1 or prop.scale == "linear" else f"{v:.3g}"


def ratio_controls(x: np.ndarray, prop, panel: str, field: str):
    """Галка «в долях от опорного бина» и поле опорного значения рядом с ней.

    Опорное значение — точка на оси X, а не номер бина: бины перекраиваются, а «×1 при
    Ql = 250» переживает перекройку и совпадает с якорями слоёв. По умолчанию — медиана
    характеристики по этой же панели, поэтому ключ поля включает ось, месторождение и
    панель: сменилась выборка — сменилось и значение по умолчанию.
    """
    c = st.columns([1.15, 1.0, 1.85])
    on = c[0].checkbox(
        "В долях от опорного бина", key=f"ratio_{panel}",
        help="Каждая статистика делится на СВОЁ значение в бине, куда попало опорное "
             "значение: Y(опора) = 1, остальные бины — доли от неё. Так сравнимы панели "
             "с разным уровнем — сутки против приведённых суток, факт против KM, одно "
             "месторождение против другого.")
    ref = None
    if on:
        step = float(prop.ui_step)
        ref = float(c[1].number_input(
            f"Опора по X, {prop.unit}", min_value=float(prop.bounds[0]),
            max_value=float(prop.bounds[1]), value=float(np.median(x)), step=step,
            format="%.2f" if step < 1 else "%.0f",
            key=f"ratio_x_{panel}_{prop.key}_{field}",
            help="По умолчанию — медиана характеристики в этой панели."))
    return on, ref


def apply_ratio(tab: pd.DataFrame, cols: list[str], on: bool, ref_x, prop, labels: dict,
                *, count_col: str = "n", count_lab: str = "n"):
    """→ (таблица для графика, {статистика: опора в сутках} | None, подпись оси Y | None).

    При любой осечке — опора вне сетки, пустой опорный бин — панель возвращается в сутках
    с предупреждением: молча показать ненормированную кривую с галкой во включённом
    состоянии значило бы соврать про то, что нарисовано.
    """
    if not on:
        return tab, None, None
    try:
        nt = FB.normalize_to_bin(tab, cols, ref_x, bounds=prop.bounds)
    except ValueError as exc:                        # noqa: BLE001 — показать оператору
        st.warning(f"Нормировка не применена: {exc}.")
        return tab, None, None
    a = nt.attrs
    if a["unnormalised"]:
        st.warning("В опорном бине нет значения для: "
                   + ", ".join(labels[c] for c in a["unnormalised"])
                   + " — эти кривые скрыты. Возьмите другую опору.")
    if not a["ref_values"]:
        return tab, None, None
    n_ref = int(tab.iloc[a["ref_bin"]][count_col])
    st.caption(
        f"Опорный бин **{fmt(a['ref_lo'], prop)}–{fmt(a['ref_hi'], prop)} {prop.unit}** "
        f"({count_lab} = {n_ref}): "
        + " · ".join(f"{labels[c]} {v:.0f} сут" for c, v in a["ref_values"].items())
        + " ⇒ ×1.00. Делитель — одно шумное число: его собственная ошибка входит во все "
        "точки одинаково и лентой не показана, поэтому кривая целиком качается, если "
        "опорный бин тонкий. Уровень читайте на панели в сутках.")
    return nt, a["ref_values"], "× к опорному бину"


# ---------------------------------------------------------------------------
# Боковая панель
# ---------------------------------------------------------------------------
st.sidebar.header("Панель")
_PROP_KEY = {f"{p.label}, {p.unit}": k for k, p in FB.PROPERTIES.items()}
prop = FB.PROPERTIES[_PROP_KEY[st.sidebar.selectbox(
    "Характеристика (ось X)", list(_PROP_KEY), key="prop",
    help="Вся машинка ниже одинакова для любой оси: меняются только окно допустимых "
         "значений, якорь и шаг сетки.")]]
st.sidebar.caption(prop.note)

VARIANTS = {"только настоящие отказы": "failures", "все закрытые пуски": "all_closed"}
variant = VARIANTS[st.sidebar.radio(
    "Что считаем отказом", list(VARIANTS), index=0,
    help="ГТМ/ППР и пуски без сигнала в первом варианте отброшены — это оценка "
         "cause-specific. Ни в одном варианте нет работающих насосов: ННО для них "
         "не существует.")]

min_field_n = st.sidebar.number_input(
    "Порог отказов для месторождения", 20, 1000, FB.MIN_FIELD_FAILURES, 10,
    key="min_field_n",
    help="Отдельную панель получают месторождения, набравшие столько отказов; остальные "
         "всё равно входят в «Весь фонд» — порог решает, кому хватает данных на свою "
         "кривую, а не кто состоит в фонде. Мирнинский считается ПОСЛЕ обязательного "
         "отбора «монтажи с 2024», поэтому его число здесь — то, которое можно приводить.")
counts = FB.field_failure_counts(variant)          # только месторождения: фонд суммируется
pcounts = FB.population_counts(variant)            # + половины Vt, их складывать НЕЛЬЗЯ
POPS = FB.selectable_populations(variant, int(min_field_n))
_flab = {FB.FLEET: f"{FB.FLEET} ({int(counts.sum())})"} | {
    p: f"{FB.population_label(p)} ({int(pcounts.get(p, 0))})" for p in POPS[1:]}
field = {v: k for k, v in _flab.items()}[st.sidebar.selectbox(
    "Месторождение", list(_flab.values()), index=min(1, len(_flab) - 1), key="field",
    help="В скобках — сколько строк даёт популяция в выбранном варианте панели. "
         "«Vt · кислый / некислый» — те же стратумы, на которых подогнана модель Vt: "
         "класс сернистости well-level (свойство флюида скважины, разнесённое на все её "
         "пуски). На остальных месторождениях «Кислый/Некислый» не заполняется, поэтому "
         "делить там нечего.")]
_below = pcounts[pcounts < int(min_field_n)]
if len(_below):
    st.sidebar.caption("Ниже порога, своей панели нет (но в фонд входят): "
                       + ", ".join(f"{FB.population_label(f)} ({int(n)})"
                                   for f, n in _below.items()))
# На дискретной оси это число уже НЕ параметр сборки (бин = типоразмер, доращивать нечего),
# только порог «тонкого» и порог тренда — и 40 отказов на один типоразмер не набирает почти
# никто, так что там свой ключ виджета и свой умолчательный порог.
target_n = st.sidebar.number_input(
    "Порог «тонкого» бина, n" if prop.discrete else "Целевое n отказов в бине",
    5, 300, FB.CATALOG_BIN_N if prop.discrete else FB.TARGET_BIN_N, 5,
    key=f"target_n_{'cat' if prop.discrete else 'rule'}",
    help=("Ось дискретная: сетку это число больше не строит — бин равен типоразмеру. "
          "Оно только помечает бины полым маркером и решает, какие бины берёт тренд. "
          "40 отказов на ОДИН типоразмер не набирает почти никто, поэтому здесь порог ниже."
          if prop.discrete else
          "Правило сборки, порог «тонкого» бина и порог надёжности для тренда — один и тот "
          "же знаменатель. Изменение сразу пересобирает сетку."))
_cap_default = prop.width_ui(prop.max_width)
max_w_shown = st.sidebar.number_input(
    prop.width_label,
    min_value=(1.05 if prop.scale == "log" else float(prop.step)),
    max_value=(20.0 if prop.scale == "log" else float(np.ptp(prop.bounds))),
    value=float(_cap_default),
    step=(0.1 if prop.scale == "log" else float(prop.step)),
    key=f"max_w_{prop.key}",
    help="Ограничение автосборки, и оно ГЛАВНЕЕ целевого n: в разреженных хвостах бин "
         "упирается в этот предел и остаётся тонким. На логарифмических осях предел — "
         "ОТНОШЕНИЕ верхней границы бина к нижней. Изменение сразу пересобирает сетку. "
         "Ручная правка границ ничем не ограничена.")
max_w = prop.width_internal(max_w_shown)

st.sidebar.header("Статистики")
_STAT_KEY = {v: k for k, v in FB.STATS.items()}
picked = [_STAT_KEY[v] for v in st.sidebar.multiselect(
    "Показать", list(_STAT_KEY), default=[FB.STATS[k] for k in ("mean", "median")])]
trim = st.sidebar.slider("Усечение (с каждого хвоста)", 0.0, 0.35, 0.10, 0.05,
                         format="%.2f")
n_boot = st.sidebar.select_slider("Бутстрэп-повторов", [0, 100, 200, 400, 800], value=400)
ci = st.sidebar.slider("Доверительный интервал, %", 50, 99, 95, 1)
show_ci = st.sidebar.checkbox("Показывать интервалы", value=True)

@st.cache_data(show_spinner="подгонка эмпирического слоя …")
def empirical_fit(field: str, variant: str, source: str, col: str, ref: float,
                  on_censored: bool):
    """AFT-наклон жизни по ковариате, подогнанный НА ВЫБРАННЫХ ДАННЫХ.

    На объединённом фонде — со своим уровнем у каждого месторождения: общий наклон по
    пулу был бы отчасти не зависимостью, а чередованием месторождений.
    """
    src = (FB.load_censored(field, source=source) if on_censored
           else FB.load_fact(field, variant=variant, source=source))
    # то же окно допустимых значений, что и у оси: наклон не должен тянуться на брак ввода
    d = FB.prepare(src, FB.PROPERTIES[col]).rename(columns={"x": col + "_x"})
    d[col] = d[col + "_x"]
    g = d["field"] if field == FB.FLEET else None
    return FB.fit_empirical_layer(d[col], d["t" if on_censored else "ttf"], d.get("event"),
                                  column=col, ref=ref, groups=g)


st.sidebar.header("Слои модели")
EMPIRICAL = "эмпирический (по этим данным)"
_MODELS = FB.models_for_field(field)
if field == FB.FLEET:
    # Пофондовая v5 переживает объединённую панель, а старые модели — нет: у v5 слой Qном
    # подобран отдельно по каждому участку, но ВСЕ они пинятся к одному опорному насосу
    # Qном 250, так что разные месторождения приводятся к общей точке. Ya v2 / Vt v4 пинятся
    # каждая к своей — их снятие на «Весь фонд» смешало бы разные нули.
    _MODELS = [k for k in _MODELS if FB.LAYER_MODEL_INFO[k]["field"] == "*"]
NO_MODEL = "— не снимать —"
# Эмпирический слой доступен ВСЕГДА: он подгоняется на самих выбранных данных, поэтому не
# требует, чтобы у месторождения была своя модель.
_mlab = {FB.LAYER_MODEL_INFO[k]["label"]: k for k in _MODELS} | {EMPIRICAL: "empirical"}
model_key = ({NO_MODEL: None} | _mlab)[st.sidebar.selectbox(
    "Модель", [NO_MODEL, *_mlab], key=f"model_{field}",
    help="Снятие слоя — это СДВИГ ШКАЛЫ ВРЕМЕНИ (AFT-офсет) t·θ^(1/β), а не вычет "
         "остатка: пуск на тяжёлом режиме несёт θ>1, и его наработка масштабируется "
         "ВВЕРХ к тому, чем она была бы на опорной точке. Результат — по-прежнему срок "
         "в сутках на той же оси, поэтому кривые «до» и «после» сравнимы напрямую.")]
if not _MODELS:
    st.sidebar.caption(
        "Подогнанные слои пинятся каждый к своей опорной точке по месторождению — снимать "
        "их с объединённой панели значило бы приводить разные месторождения к разным нулям. "
        "Эмпирический слой этим не связан." if field == FB.FLEET else
        f"Для «{FB.population_label(field)}» готовой модели слоёв нет; эмпирический "
        f"считается по самим данным.")

picked_layers, emp_fit = [], None
if model_key == "empirical":
    _elab = {FB.LAYER_AXIS[l][0]: l for l in ("ql", "qnom", "kpod")}
    emp_col = _elab[st.sidebar.selectbox(
        "Ковариата", list(_elab), index=0, key="emp_col",
        help="Ql стоит первым не случайно: из всех осей он даёт самую сильную и самую "
             "устойчивую зависимость.")]
    emp_on_cens = st.sidebar.radio(
        "Подгонять по", ("фактической панели", "популяции с цензурированием"), index=0,
        key="emp_on",
        help="Фактическая панель — только поднятые насосы, поэтому наклон наследует отбор "
             "«уже отказал». Популяция с цензурированием честнее, но её сборка идёт минуту. "
             "Подгонка ОДНА и применяется к обеим панелям — иначе они перестали бы быть "
             "сравнимыми.") == "популяции с цензурированием"
    try:
        with st.spinner("подгонка эмпирического слоя …"):
            emp_fit = empirical_fit(field, variant, prop.source, emp_col,
                                    float(FB.PROPERTIES[emp_col].anchor[0]), emp_on_cens)
        st.sidebar.caption(str(emp_fit))
        if st.sidebar.checkbox(f"Снять {FB.LAYER_AXIS[emp_col][0]} (эмпирический)",
                               value=True, key="emp_use"):
            picked_layers = [emp_col]
    except Exception as exc:                          # noqa: BLE001 — показать оператору
        st.sidebar.warning(f"Не подогнать: {exc}")
        model_key = None
elif model_key:
    info = FB.LAYER_MODEL_INFO[model_key]
    st.sidebar.caption(info["note"])
    _llab = {FB.LAYER_AXIS[l][0]: l for l in info["layers"]}
    picked_layers = [_llab[v] for v in st.sidebar.multiselect(
        "Снять слои", list(_llab), default=[],
        help="Снимаются со шкалы времени ДО биннинга. Число пусков в бинах не меняется — "
             "меняются только наработки, поэтому сетка остаётся той же.")]


def chosen_layer_model():
    """Подогнанная модель или обёртка вокруг эмпирического наклона — путь снятия один."""
    return (FB.empirical_layer_model(emp_fit) if model_key == "empirical"
            else FB.layer_model(model_key))


# снятие слоя, совпадающего с осью X, выпрямляет ровно ту зависимость, на которую смотрим
self_cancel = [l for l in picked_layers if FB.LAYER_AXIS[l][1] == prop.key]

st.sidebar.header("Тренд")
show_trend = st.sidebar.checkbox("Показывать тренд", value=True)
_FORM_KEY = {v: k for k, v in FB.TREND_FORMS.items()}
trend_form = _FORM_KEY[st.sidebar.selectbox(
    "Форма", list(_FORM_KEY), index=list(_FORM_KEY).index(FB.TREND_FORMS["poly2"]),
    help="Подгонка идёт по ЛОГАРИФМУ статистики и только по бинам, набравшим целевое n; "
         "тонкие бины исключаются, а не берутся с малым весом. Вес бина = n / sigma_log² — "
         "обратная дисперсия, иначе большой, но разбросанный бин перетягивает кривую.")]

# ---------------------------------------------------------------------------
# Данные и границы
# ---------------------------------------------------------------------------
st.title(f"Бины по характеристике «{prop.label}» — {FB.population_label(field)}, "
         f"фактические отказы Свода")

# Тяжёлые источники собираются по кнопке: молчаливая пауза на несколько минут при
# переключении характеристики читалась бы как зависание.
HEAVY = {"warehouse": "суточному окну t0 из склада",
         "pressure": "телеметрии Pзаб (первые 30 рабочих суток) и марту Pнас"}
if prop.source in HEAVY and not st.session_state.get(f"src_ok_{prop.source}"):
    st.info(f"«{prop.label}» в Своде нет — считается по {HEAVY[prop.source]}. "
            f"Сборка идёт несколько минут и потом кэшируется на всю сессию.")
    if st.button("Собрать ковариаты", type="primary"):
        st.session_state[f"src_ok_{prop.source}"] = True
        st.rerun()
    st.stop()

df = FB.prepare(fact(field, variant, prop.source), prop)
if df.empty:
    st.error(f"На «{FB.population_label(field)}» нет строк с ННО и пригодным значением "
             f"«{prop.label}».")
    st.stop()

adj_meta = {}
ylab = "ННО, сут"
if picked_layers:
    with st.spinner("подгонка модели (первый раз — пара минут, дальше из кэша) …"):
        lm = chosen_layer_model()
    df["ttf"], adj_meta = FB.remove_layers(df, "ttf", lm, picked_layers)
    ylab = "ННО приведённое, сут"
x, ttf = df["x"].to_numpy(float), df["ttf"].to_numpy(float)
edges = ensure_edges(x, f"{prop.key}|{field}|{variant}|{int(target_n)}|{max_w_shown:g}",
                     prop, int(target_n), max_w)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Отказов в панели", f"{len(df):,}".replace(",", " "),
          help="Это НЕ все отказы месторождения: панель — отфильтрованный лист Свода. "
               "Полный счёт по шагам отбора — в «Откуда это число» под метриками.")
c2.metric("Отброшено", f"{df.attrs['n_missing'] + df.attrs['n_out_of_bounds']:,}"
                       .replace(",", " "),
          help=f"Только потери НА ЭТОЙ ОСИ. Нет значения: {df.attrs['n_missing']}. Вне окна "
               f"{fmt(prop.bounds[0], prop)}–{fmt(prop.bounds[1], prop)} {prop.unit}: "
               f"{df.attrs['n_out_of_bounds']} (это не данные, а брак ввода). Отбор до оси "
               f"(вариант панели, полнота строки Свода, когорта Мирнинского) сюда не входит "
               f"— он в «Откуда это число».")
c3.metric("Бинов", len(edges) - 1)
w = np.diff(edges)
if prop.scale == "log":
    r = np.array(edges[1:]) / np.array(edges[:-1])
    c4.metric("Отношение границ, ×", f"{r.min():.2f}–{r.max():.2f}")
else:
    c4.metric(f"Ширина, {prop.unit}", f"{fmt(w.min(), prop)}–{fmt(w.max(), prop)}")

with st.expander(f"Откуда это число: {len(df)} из "
                 f"{FB.population_funnel(field, variant)['kept'].iloc[0]} строк Свода"):
    fun = FB.funnel_table(field, prop, df, variant=variant)
    st.markdown(
        "Панель — это лист Свода после нескольких отборов, и они крупные: каждая строка "
        "ниже показывает, сколько **пусков выбранной популяции** пережило один названный "
        "фильтр. Число, которое обычно помнят про месторождение, — это одна из ВЕРХНИХ "
        "строк, а не нижняя.")
    st.dataframe(
        fun.rename(columns={"stage": "шаг отбора", "kept": "осталось",
                            "dropped": "отсеяно"})[["шаг отбора", "осталось", "отсеяно"]],
        use_container_width=True, hide_index=True)
    big = fun.iloc[1:].sort_values("dropped", ascending=False).iloc[0]
    st.caption(
        f"Самый крупный отсев — «{big['stage']}»: −{int(big['dropped'])}. "
        + ("Вариант «только настоящие отказы» отбрасывает ГТМ/ППР и подъёмы без признака "
           "отказа — это цензурирования, а не отказы, и ННО на них меряет приезд бригады. "
           "Переключатель «Что считаем отказом» в боковой панели возвращает их. "
           if variant == "failures" else "")
        + "Телеметрия здесь ни при чём: оси Свода (частота, дебит, номинал, Kпод) считаются "
          "по самому листу и телеметрию не требуют. Требуют её только ГЖ, обводнённость и "
          "Pзаб — и там недостачу видно строкой «есть значение».")

if picked_layers:
    names = ", ".join(FB.LAYER_AXIS[l][0] for l in picked_layers)
    src_lab = (EMPIRICAL if model_key == "empirical"
               else FB.LAYER_MODEL_INFO[model_key]["label"])
    st.info(f"Со шкалы времени снято: **{names}** ({src_lab}, "
            f"β={adj_meta['beta_median']:.4f}). Медианное θ = {adj_meta['theta_median']:.3f}; "
            f"{adj_meta['n_unadjusted']} пусков без нужной ковариаты прошли без поправки "
            f"(θ=1) — популяция не меняется при переключении слоёв, поэтому сдвиг кривой "
            f"означает поправку, а не другую выборку.")
if self_cancel:
    st.warning(f"Снимается слой **{', '.join(FB.LAYER_AXIS[l][0] for l in self_cancel)}**, "
               f"а ось X — «{prop.label}». Это одна и та же переменная: модель выпрямит "
               f"ровно ту зависимость, на которую вы смотрите. Как диагностика годится "
               f"(«достаточно ли снято?»), как измерение эффекта — нет.")

tab_bins, tab_shape, tab_cens, tab_table = st.tabs(
    ["Биннинг", "Форма распределения", "С цензурированием", "Таблица и выгрузка"])

# ---------------------------------------------------------------------------
# Вкладка 1 — биннинг
# ---------------------------------------------------------------------------
with tab_bins:
    st.subheader("Границы")
    cap_txt = (f"×{max_w_shown:g}" if prop.scale == "log"
               else f"{max_w_shown:g} {prop.unit}")
    rule_lab = (f"Пересобрать: n ≥ {int(target_n)}, ≤ {cap_txt}, "
                f"от {fmt(prop.anchor[0], prop)} наружу")
    # на дискретной оси правило n тоже доступно — но вторым, а не по умолчанию
    b = st.columns([1.5, 1.2, 1.1, 1.1, 1.0, 0.8] if prop.discrete
                   else [1.4, 1.1, 1.1, 1.0, 0.8])
    k = 1 if prop.discrete else 0
    if prop.discrete:
        if b[0].button("Пересобрать: один бин на типоразмер", use_container_width=True,
                       type="primary",
                       help="Ось дискретная: границы режутся посередине между соседними "
                            "значениями, ни один бин не доращивается до целевого n. "
                            "Тонкие бины здесь — не брак сетки, а сколько раз этот "
                            "типоразмер вообще ставили."):
            set_edges(build_catalog(x, prop), prop)
            st.rerun()
        if b[1].button(f"Правило n ≥ {int(target_n)}", use_container_width=True,
                       help=rule_lab + ". Сливает соседние типоразмеры — годится, чтобы "
                                       "посмотреть на ось как на непрерывную."):
            set_edges(build_rule(x, prop, int(target_n), max_w), prop)
            st.rerun()
    elif b[0].button(rule_lab, use_container_width=True, type="primary"):
        set_edges(build_rule(x, prop, int(target_n), max_w), prop)
        st.rerun()
    even = prop.width_ui(prop.step * 2)
    if b[k + 1].button(f"Равномерно ({'×' if prop.scale == 'log' else ''}{even:g}"
                       f"{'' if prop.scale == 'log' else ' ' + prop.unit})",
                       use_container_width=True):
        set_edges(FB.uniform_edges(*prop.bounds, even, scale=prop.scale,
                                   min_width=prop.list_min_width, sig=prop.sig), prop)
        st.rerun()
    if b[k + 2].button("Равное число (10 бинов)", use_container_width=True):
        set_edges(FB.equal_count_edges(x, 10, bounds=prop.bounds, scale=prop.scale,
                                       round_to=prop.ui_step, min_width=prop.list_min_width,
                                       sig=prop.sig), prop)
        st.rerun()
    if b[k + 3].button("Слить тонкие", use_container_width=True,
                       help=f"Складывать самый тонкий бин в меньшего соседа, пока все не "
                            f"наберут {int(target_n)}."):
        set_edges(FB.merge_thin_bins(edges, x, int(target_n), min_width=prop.list_min_width),
                  prop)
        st.rerun()
    if b[k + 4].button("Отменить", use_container_width=True,
                       disabled=not st.session_state.get("undo")) and st.session_state.get("undo"):
        st.session_state.edges = st.session_state.undo.pop()
        st.rerun()

    left, right = st.columns([1.05, 1.0])

    with left:
        st.markdown("**Сдвинуть границу**")
        n_e = len(edges)
        # Ярлыки, а не индексы с format_func: значение виджета тогда — сама строка,
        # и состояние переживает перерисовку без обратного отображения.
        e_labels = [f"#{k}: {fmt(edges[k], prop)} {prop.unit}" for k in range(n_e)]
        i = e_labels.index(st.select_slider(
            "Какую", options=e_labels, value=e_labels[min(1, n_e - 1)], key="edge_pick"))
        # зазор берётся у характеристики: на логарифмической оси он ОТНОСИТЕЛЬНЫЙ,
        # иначе у низкого конца шкалы границу было бы некуда двигать
        lo = edges[i - 1] + prop.min_gap(edges[i - 1]) if i > 0 else prop.bounds[0]
        hi = edges[i + 1] - prop.min_gap(edges[i + 1]) if i < n_e - 1 else prop.bounds[1]
        if hi - lo >= prop.ui_step:
            pos = st.slider(f"Куда, {prop.unit}", float(lo), float(hi), float(edges[i]),
                            float(prop.ui_step), key=f"edge_pos_{i}_{edges[i]:g}")
            if abs(pos - edges[i]) > 1e-9 and st.button("Применить сдвиг",
                                                        use_container_width=True):
                set_edges(FB.move_edge(edges, i, pos,
                                       min_width=prop.min_gap(edges[i])), prop)
                st.rerun()
        else:
            st.caption("Соседние границы вплотную — сдвигать некуда.")

        st.markdown("**Слить или разделить бин**")
        counts = FB.bin_counts(x, edges)
        names = [f"#{k}: {fmt(edges[k], prop)}–{fmt(edges[k + 1], prop)} "
                 f"(n={int(counts[k])})" for k in range(len(edges) - 1)]
        j = names.index(st.selectbox("Бин", names, key="bin_pick"))
        # Условие повторено в обработчике, а не только в ``disabled``: состояние виджета
        # переживает перестройку сетки, и выбранный бин может исчезнуть до клика.
        can_split = (edges[j + 1] - edges[j]) >= 2 * prop.min_gap(edges[j])
        m1, m2, m3 = st.columns(3)
        if m1.button("← слить с пред.", use_container_width=True, disabled=j == 0) and j > 0:
            set_edges(FB.merge_bin(edges, j - 1), prop)
            st.rerun()
        if (m2.button("слить со след. →", use_container_width=True,
                      disabled=j >= len(names) - 1) and j < len(names) - 1):
            set_edges(FB.merge_bin(edges, j), prop)
            st.rerun()
        if m3.button("разделить ÷2", use_container_width=True,
                     disabled=not can_split) and can_split:
            set_edges(FB.split_bin(edges, j, min_width=prop.min_gap(edges[j]),
                                   scale=prop.scale), prop)
            st.rerun()

    with right:
        st.markdown("**Список границ** — правьте значение, добавляйте строку (разделить) "
                    "или удаляйте (слить)")
        col = f"граница, {prop.unit}"
        ed = st.data_editor(
            pd.DataFrame({col: [float(v) for v in edges]}),
            num_rows="dynamic", hide_index=True, use_container_width=True, height=300,
            column_config={col: st.column_config.NumberColumn(
                format="%.4g", step=float(prop.ui_step), min_value=0.0,
                max_value=float(prop.bounds[1]) * 10)},
            key=f"edge_table_{prop.key}_{len(edges)}_{edges[0]:g}_{edges[-1]:g}")
        new = [float(v) for v in ed[col].dropna().tolist()]
        if len(new) >= 2 and FB.sanitize_edges(new, min_width=prop.list_min_width) != list(
                map(float, edges)):
            if st.button("Применить список", type="primary", use_container_width=True):
                set_edges(new, prop)
                st.rerun()

    tab = with_centres(
        stats_for(tuple(x), tuple(ttf), tuple(edges), trim, int(n_boot), float(ci)),
        x, edges, prop)
    outside = tab.attrs.get("n_outside", 0)
    if outside:
        st.warning(f"{outside} отказов вне диапазона границ "
                   f"({fmt(edges[0], prop)}–{fmt(edges[-1], prop)} {prop.unit}) — "
                   f"в расчёт не входят.")

    cols_shown = picked or ["median"]
    fits = trends(tab, cols_shown, trend_form, int(target_n), prop) if show_trend else {}
    r_on, r_x = ratio_controls(x, prop, "fact", field)
    plot_tab, ratio_ref, r_ylab = apply_ratio(tab, cols_shown, r_on, r_x, prop, FB.STATS)
    st.plotly_chart(
        bins_figure(plot_tab, cols_shown, FB.STATS, target_n=int(target_n), prop=prop,
                    ylab=r_ylab or ylab, show_ci=show_ci, fits=fits, ratio_ref=ratio_ref),
        use_container_width=True)

    if show_trend:
        if fits:
            tc = st.columns(len(fits))
            for c, (k, f) in zip(tc, fits.items()):
                # «оптимум» только если он внутри диапазона: на монотонной кривой
                # (больше дебит — короче жизнь) максимум просто упирается в край
                head = "макс. на краю" if f.opt_at_edge else "оптимум"
                c.metric(f"{head} · {FB.STATS[k]}", f"{fmt(f.opt_hz, prop)} {prop.unit}",
                         f"{f.opt_value:.0f} сут · R²={f.weighted_r2:.2f} · "
                         f"{f.n_bins} бинов / {f.n_runs} отказов", delta_color="off")
            if any(f.opt_at_edge for f in fits.values()):
                st.caption("«Макс. на краю» значит, что на подогнанном диапазоне кривая "
                           "монотонна и внутреннего оптимума нет — максимум просто упёрся "
                           "в границу носителя.")
            f0 = next(iter(fits.values()))
            st.caption(
                f"{FB.TREND_FORMS[trend_form]}, подгонка по логарифму статистики с весом "
                f"n/sigma_log², только бины с n ≥ {int(target_n)}. Линия нарисована лишь на "
                f"диапазоне {fmt(f0.support[0], prop)}–{fmt(f0.support[1], prop)} "
                f"{prop.unit} — за его пределами надёжных бинов нет, и полином там "
                f"ничего не знает.")
        else:
            # конкретное число, а не «понизьте»: на 84 отказах порог 40 оставляет два бина,
            # и оператору важно знать, до чего именно опускать
            suggest = max(10, int(round(len(df) / 8 / 5) * 5))
            st.info(f"Надёжных бинов (n ≥ {int(target_n)}) меньше, чем параметров у формы "
                    f"«{FB.TREND_FORMS[trend_form]}» — тренд не строится. На {len(df)} "
                    f"отказах порог {int(target_n)} слишком высок: попробуйте ≈{suggest} "
                    f"(даст порядка 8 бинов) или возьмите форму попроще.")

    if field == FB.FLEET:
        mix = FB.composition(x, df["field"], edges)
        st.markdown("**Состав фонда по бинам.** На объединённой панели среднее в бине — это "
                    "отчасти среднее по составу месторождений, а они различаются по сроку "
                    "службы в разы. Где состав ровный, форма кривой — это характеристика; "
                    "где состав едет, сначала подозревайте состав.")
        fm = go.Figure()
        for k, name in enumerate(mix["label"].unique()):
            sub = mix[mix["label"] == name]
            fm.add_trace(go.Bar(
                x=(np.sqrt(sub["f_lo"] * sub["f_hi"]) if prop.scale == "log"
                   else sub["f_mid"]),
                y=sub["share"], name=str(name),
                marker=dict(color=MIX_COLORS[k % len(MIX_COLORS)],
                            line=dict(color="white", width=0.5)),
                customdata=np.column_stack([sub["f_lo"], sub["f_hi"], sub["n"], sub["n_bin"]]),
                hovertemplate=(f"<b>{name}</b>: %{{y:.0%}}<br>"
                               f"бин %{{customdata[0]:{prop.fmt[1:]}}}–"
                               f"%{{customdata[1]:{prop.fmt[1:]}}} {prop.unit} · "
                               "%{customdata[2]:.0f} из %{customdata[3]:.0f}<extra></extra>")))
        fm.update_layout(barmode="stack", height=280, template="plotly_white",
                         margin=dict(l=60, r=20, t=10, b=45), bargap=0.06,
                         xaxis_title=prop.axis_title(), yaxis_title="доля бина",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fm.update_yaxes(tickformat=".0%", gridcolor="#ECECEC", range=[0, 1])
        fm.update_xaxes(gridcolor="#F4F4F4",
                        **({"type": "log"} if prop.scale == "log" else {}))
        st.plotly_chart(fm, use_container_width=True)
        st.caption("Столбцы стоят в серединах бинов и читаются по ВЫСОТЕ — ширина здесь "
                   "автоматическая, а точные границы бинов видны на панели счётчика выше.")

    st.markdown("**Отличается ли кривая от плоской?** Перестановочный тест: значения "
                "характеристики перемешиваются относительно наработок, сетка держится, "
                "сравнивается размах кривой. Бины тоньше целевого n в тест не входят.")
    if st.button("Прогнать тест по выбранным статистикам"):
        res = [flatness(tuple(x), tuple(ttf), tuple(edges), s, 600, int(target_n), trim)
               for s in cols_shown]
        cols = st.columns(len(res))
        for col_, r in zip(cols, res):
            col_.metric(FB.STATS[r.stat], f"p = {r.p_value:.3f}",
                        f"размах {r.spread:.0f} д · {r.n_bins} из {len(edges) - 1} бинов",
                        delta_color="off")
        if res and res[0].n_bins < len(edges) - 1:
            k = len(edges) - 1 - res[0].n_bins
            word = "бин" if k % 10 == 1 and k % 100 != 11 else (
                "бина" if k % 10 in (2, 3, 4) and k % 100 not in (12, 13, 14) else "бинов")
            st.caption(
                f"⚠ {k} {word} тоньше {int(target_n)} отказов и в тест не вошли. При "
                f"ограничении ширины это обычно хвосты — а именно они несут крайние "
                f"значения, так что p здесь строже, чем на сетке, где хвосты слиты "
                f"в широкие бины.")

# ---------------------------------------------------------------------------
# Вкладка 2 — форма распределения
# ---------------------------------------------------------------------------
with tab_shape:
    tab = with_centres(
        stats_for(tuple(x), tuple(ttf), tuple(edges), trim, int(n_boot), float(ci)),
        x, edges, prop)
    d = tab[tab["n"] > 0]
    st.subheader("Почему среднее и медиана расходятся")
    st.markdown(
        "Разрыв держится на трёх разных вещах, и они разделимы:\n\n"
        "* **скошенность** — жизнь распределена мультипликативно, поэтому среднее всегда "
        "выше медианы; правильный центр здесь — **геометрическое среднее**, а порождающий "
        "разрыв масштаб — `sigma_log`;\n"
        "* **смесь** — отрицательная скошенность *в логарифмах* означает не длинный правый "
        "хвост, а **комок ранних смертей** слева. Такую смесь ни одно число не описывает "
        "честно: смотреть надо на долю ранних отказов отдельно;\n"
        "* **шум** — у медианы тонкого бина интервал в сотни суток.")

    DIAGS = {
        "sigma_log — разброс в логарифмах (масштаб разрыва)": "sigma_log",
        "skew_log — скошенность в логарифмах (<0 ⇒ комок ранних смертей)": "skew_log",
        "среднее / медиана": "mean_over_median",
        "доля отказов до 90 суток": "share_under_90d",
        "p90 / медиана": "p90_over_p50"}
    diag = DIAGS[st.selectbox("Диагностика", list(DIAGS))]
    fig = go.Figure()
    thin = (d["n"] < int(target_n)).to_numpy()
    fig.add_trace(go.Scatter(
        x=d["f_mid"], y=d[diag], mode="lines+markers", showlegend=False,
        line=dict(color="#1B6DB5", width=2.4),
        marker=dict(color=np.where(thin, "white", "#1B6DB5"), size=9,
                    line=dict(color="#1B6DB5", width=1.8)),
        customdata=np.column_stack([d["f_lo"], d["f_hi"], d["n"]]),
        hovertemplate=("%{y:.2f}<br>бин %{customdata[0]:.4g}–%{customdata[1]:.4g} · "
                       "n = %{customdata[2]:.0f}<extra></extra>")))
    if diag in ("mean_over_median", "p90_over_p50"):
        fig.add_hline(y=1.0, line=dict(color=INK3, width=1, dash="dot"))
    if diag == "skew_log":
        fig.add_hline(y=0.0, line=dict(color=INK3, width=1, dash="dot"))
    fig.add_vline(x=prop.anchor[0], line=dict(color=INK3, width=1, dash="dot"))
    fig.update_layout(height=380, template="plotly_white",
                      xaxis_title=prop.axis_title(), yaxis_title=diag,
                      margin=dict(l=60, r=20, t=20, b=50))
    fig.update_xaxes(gridcolor="#F4F4F4", **({"type": "log"} if prop.scale == "log" else
                                             ({"dtick": 2} if prop.key == "freq" else {})))
    fig.update_yaxes(gridcolor="#ECECEC", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Распределение внутри бина")
    idx = FB.bin_index(x, edges)
    names = [f"#{k}: {fmt(edges[k], prop)}–{fmt(edges[k + 1], prop)} "
             f"(n={int((idx == k).sum())})" for k in range(len(edges) - 1)]
    sel = [names.index(v) for v in st.multiselect(
        "Бины", names, default=[names[0], names[-1]] if len(names) > 1 else names,
        help="Наложенные ECDF: смесь видна как ступенька у самого начала оси, "
             "а не как сдвиг всей кривой.")]
    if sel:
        f2 = go.Figure()
        ramp = ["#1B6DB5", "#E8762C", "#1F6F45", "#A0439B", "#8A8A8A"]
        for k, kk in enumerate(sel):
            y = np.sort(ttf[idx == kk])
            if not y.size:
                continue
            f2.add_trace(go.Scatter(
                x=y, y=np.arange(1, y.size + 1) / y.size, mode="lines",
                line=dict(color=ramp[k % len(ramp)], width=2, shape="hv"),
                name=f"{fmt(edges[kk], prop)}–{fmt(edges[kk + 1], prop)}",
                hovertemplate="%{x:.0f} сут · %{y:.0%}<extra></extra>"))
        f2.add_vline(x=90.0, line=dict(color=INK3, width=1, dash="dot"),
                     annotation_text="90 сут", annotation_font_size=10)
        f2.update_layout(height=400, template="plotly_white", xaxis_title="ННО, сут",
                         yaxis_title="доля отказов ≤ t", margin=dict(l=60, r=20, t=20, b=50),
                         legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
        f2.update_yaxes(tickformat=".0%", gridcolor="#ECECEC")
        f2.update_xaxes(gridcolor="#F4F4F4")
        st.plotly_chart(f2, use_container_width=True)

    st.dataframe(
        d[["f_lo", "f_hi", "n", "mean", "median", "geomean", "trimmed", "sigma_log",
           "skew_log", "mean_over_median", "share_under_90d"]]
        .style.format({c: "{:.2f}" for c in ("mean", "median", "geomean", "trimmed",
                                             "sigma_log", "skew_log", "mean_over_median",
                                             "share_under_90d")}),
        use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Вкладка 3 — с цензурированием
# ---------------------------------------------------------------------------
with tab_cens:
    st.subheader("Те же бины, но с работающими насосами")
    st.markdown(
        "ННО существует только у поднятого насоса, поэтому фактическая панель обусловлена "
        "на «уже отказал». Здесь та же колонка характеристики, но популяция целиком: "
        "работающие насосы дают экспозицию без события, а центр читается с кривой "
        "Каплана–Мейера — **медиана KM** и **RMST(0, 730)**. Это те оценки, которые "
        "проект обязан приводить; сырое среднее ННО ими не является.")
    if field in FB.SOUR_SPLIT:
        st.caption(
            "На половинах Vt класс сернистости здесь — **well-level relabel v3.2**, "
            "включённый принудительно: пофлажный класс пишется только из базы отказов, и "
            "работающий насос его не получает. Без relabel «кислая» половина приходит с "
            "нулём открытых пусков — то есть панель с цензурированием без цензурирования. "
            "С ним 30 работающих насосов кислых скважин возвращаются в кислую половину, "
            "и число отказов при этом не меняется.")
    if st.button("Собрать популяцию с цензурированием", type="primary"):
        st.session_state.want_cens = True
    if st.session_state.get("want_cens"):
        cen = FB.prepare(censored(field, prop.source), prop)
        if picked_layers and not cen.empty:
            # тот же офсет и на цензурированных строках: сдвиг шкалы времени двигает и
            # событие, и цензурирование, поэтому KM остаётся состоятельным
            cen["t"], _ = FB.remove_layers(cen, "t", chosen_layer_model(), picked_layers)
        if cen.empty:
            st.error("Нет строк с пригодным значением характеристики.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Пусков", f"{len(cen):,}".replace(",", " "))
            k2.metric("Отказов", f"{int((cen['event'] == 1).sum()):,}".replace(",", " "))
            k3.metric("Цензурировано", f"{int((cen['event'] != 1).sum()):,}".replace(",", " "))
            k4.metric("Отброшено", f"{cen.attrs['n_missing'] + cen.attrs['n_out_of_bounds']:,}"
                                   .replace(",", " "),
                      help="В основном пуски из Big, которых нет в Своде, — у них нет "
                           "колонок Свода.")
            kt = with_centres(
                km_for(tuple(cen["x"].to_numpy(float)), tuple(cen["t"].to_numpy(float)),
                       tuple(cen["event"].to_numpy(float)), tuple(edges), int(n_boot),
                       float(ci)),
                cen["x"].to_numpy(float), edges, prop)
            km_key = {v: k for k, v in FB.KM_STATS.items()}
            which = [km_key[v] for v in st.multiselect(
                "Показать", list(km_key), default=list(km_key))]
            km_cols = which or ["rmst"]
            # надёжность здесь — число СОБЫТИЙ в бине, а не число пусков: цензурированный
            # пуск даёт экспозицию, но кривую в бине двигают отказы
            km_fits = (trends(kt, km_cols, trend_form, int(target_n), prop,
                              count_col="events") if show_trend else {})
            kr_on, kr_x = ratio_controls(cen["x"].to_numpy(float), prop, "km", field)
            km_tab, km_ratio, km_ylab = apply_ratio(kt, km_cols, kr_on, kr_x, prop,
                                                    FB.KM_STATS, count_col="events",
                                                    count_lab="отказов")
            st.plotly_chart(
                bins_figure(km_tab, km_cols, FB.KM_STATS, target_n=int(target_n), prop=prop,
                            ylab=km_ylab or ("сутки приведённые" if picked_layers
                                             else "сутки"),
                            count_col="events",
                            count_lab="отказов (из n пусков)", show_ci=show_ci,
                            fits=km_fits, ratio_ref=km_ratio),
                use_container_width=True)
            if km_fits:
                tc = st.columns(len(km_fits))
                for c, (k, f) in zip(tc, km_fits.items()):
                    head = "макс. на краю" if f.opt_at_edge else "оптимум"
                    c.metric(f"{head} · {FB.KM_STATS[k]}",
                             f"{fmt(f.opt_hz, prop)} {prop.unit}",
                             f"{f.opt_value:.0f} сут · R²={f.weighted_r2:.2f} · "
                             f"{f.n_bins} бинов", delta_color="off")

            st.markdown("**Сколько добавляет отбор.** Наивные среднее и медиана считаются "
                        "по тем же строкам, но только по отказавшим — разница с KM/RMST "
                        "и есть цена условия «уже отказал».")
            cmp = kt[kt["n"] > 0].copy()
            cmp["rmst_minus_naive_mean"] = cmp["rmst"] - cmp["naive_mean"]
            st.dataframe(
                cmp[["f_lo", "f_hi", "n", "events", "censored", "km_median", "rmst",
                     "naive_median", "naive_mean", "rmst_minus_naive_mean",
                     "surv_at_horizon"]]
                .style.format({c: "{:.0f}" for c in ("km_median", "rmst", "naive_median",
                                                     "naive_mean", "rmst_minus_naive_mean")}
                              | {"surv_at_horizon": "{:.2f}"}),
                use_container_width=True, hide_index=True)
            st.caption("`surv_at_horizon` — доля выживших на 730 сут: чем она выше, тем "
                       "больше площади под RMST держится на плоской экстраполяции хвоста.")
    else:
        st.info("Сборка идёт через `esp_population.build` (Свод + Big) и занимает около "
                "минуты; результат кэшируется.")

# ---------------------------------------------------------------------------
# Вкладка 4 — таблица и выгрузка
# ---------------------------------------------------------------------------
with tab_table:
    tab = with_centres(
        stats_for(tuple(x), tuple(ttf), tuple(edges), trim, int(n_boot), float(ci)),
        x, edges, prop)
    st.dataframe(tab, use_container_width=True, hide_index=True)
    st.download_button("Скачать таблицу бинов (csv)",
                       tab.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"bins_{prop.key}_{field}_{variant}.csv", mime="text/csv")

    st.markdown("**Границы** — строка ниже полностью задаёт сетку: её можно вставить "
                "обратно сюда или передать в `freq_bins.parse_edges` в скрипте.")
    st.code("|".join(f"{v:g}" for v in edges), language="text")
    paste = st.text_input("Вставить границы", "")
    if paste and st.button("Загрузить"):
        try:
            set_edges(FB.parse_edges(paste), prop)
            st.rerun()
        except Exception as exc:                     # noqa: BLE001 — показать оператору
            st.error(f"Не разобрать: {exc}")

    label = st.text_input("Метка для сохранения", f"{prop.key}_{field}")
    if st.button("Сохранить в results/"):
        kt = None
        if st.session_state.get("want_cens"):
            cen = FB.prepare(censored(field, prop.source), prop)
            if picked_layers:
                cen["t"], _ = FB.remove_layers(cen, "t", chosen_layer_model(),
                                               picked_layers)
            kt = with_centres(
                km_for(tuple(cen["x"].to_numpy(float)), tuple(cen["t"].to_numpy(float)),
                       tuple(cen["event"].to_numpy(float)), tuple(edges), int(n_boot),
                       float(ci)),
                cen["x"].to_numpy(float), edges, prop)
        out = FB.save(edges, fact=tab, km=kt, label=label or f"{prop.key}_{field}")
        st.success(f"Записано в {out}")
