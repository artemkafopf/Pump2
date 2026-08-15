"""Interactive explorer for the Vt frequency and Kpod layers.

Shows, for each layer: the ORIGINAL Ya empirical curves (all-pulls and failures-only, as
recovered from the 2026-07-24 study tables), a bootstrap band re-derived from the warehouse,
the fitted Ya polyline, the currently shipped closed form — and a controllable form you can
drive with two sliders.

    streamlit run streamlit_apps/vt_layer_explorer_app.py

Units note: the surviving study table is in **life (TTF) multipliers** pinned at the reference;
this app converts them to hazard multipliers with ``θ = (1 / mult) ** β``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.paths import RESULTS_ROOT                                   # noqa: E402
from analysis.workflows.production_risk import unified_v4 as U            # noqa: E402
from analysis.workflows.production_risk import vt_composed_model as CM    # noqa: E402
from analysis.workflows.production_risk import vt_v4 as V                 # noqa: E402
from analysis.workflows.production_risk import ya_freq_empirical as E     # noqa: E402

st.set_page_config(page_title="Vt layer explorer", layout="wide")

ORIG_DIR = Path(RESULTS_ROOT) / "production_risk_ya_freq_kpod_shape" / "2026-07-24" / "tables"

#: Baselines for the RMST multiplier on the frequency and Kpod tabs.  Left on the v3.2/v4 Vt
#: values deliberately: those two tabs are the shipped hand-tuned deliverable and are not part
#: of the unified refit.  The Qном tab carries unified v4's own per-stratum baselines instead.
BASE = {"nonsour": (1.2133, 686.2), "sour": (1.2766, 202.1)}

U_STRATA = ("Ya", "Vt_nonsour", "Vt_sour")
U_COLOR = {"Ya": "#3B7EA1", "Vt_nonsour": "#C4622D", "Vt_sour": "#7A4E9E"}
#: Свод strata carry the field's own label; map them onto the unified strata.
U_OF_SVOD = {("Ya", "nonsour"): "Ya", ("Ya", "sour"): "Ya",
             ("Vt", "nonsour"): "Vt_nonsour", ("Vt", "sour"): "Vt_sour"}


@st.cache_data(show_spinner="загрузка unified v4 …")
def unified_layers(refit: bool = False) -> dict:
    """θ_Qном knots, contractor levels and baselines from unified v4.

    Prefers the newest **written** run so the bootstrap bands come for free; falls back to a
    live fit (without bands) when nothing has been written yet.
    """
    if not refit:
        base = Path(RESULTS_ROOT) / U.SLUG
        runs = sorted(p for p in base.glob("*/tables") if (p / "layers.csv").exists())
        if runs:
            lay = pd.read_csv(runs[-1] / "layers.csv")
            bl = pd.read_csv(runs[-1] / "baselines.csv").set_index("stratum")
            out = {}
            for s in U_STRATA:
                d = {"beta0": float(bl.loc[s, "beta0"]), "eta0": float(bl.loc[s, "eta0"]),
                     "rmst_ref": float(bl.loc[s, "rmst730_ref"]),
                     "n": int(bl.loc[s, "n"]), "events": int(bl.loc[s, "events"]),
                     "level": {c: float(bl.loc[s, f"hr_{c}"]) for c in ("brt", "slb", "oth")}}
                for name in ("qnom", "kpod", "freq"):
                    q = lay[(lay.stratum == s) & (lay.layer == name)].sort_values("x")
                    d[name] = {"x": q.x.to_numpy(float), "theta": q.theta.to_numpy(float),
                               "lo": q.theta_lo.to_numpy(float), "hi": q.theta_hi.to_numpy(float)}
                out[s] = d
            return {"src": str(runs[-1].parent.name), **out}

    r = U.run(n_boot=0, write=False)
    return {"src": "живой пересчёт (без бутстрэпа)", **{
        s: {"beta0": f.beta0, "eta0": f.eta0, "rmst_ref": U.rmst(f.eta0, f.beta0),
            "n": f.n, "events": f.events, "level": dict(f.contractor),
            **{name: {"x": np.asarray(U.KNOTS[name], float),
                      "theta": np.asarray(f.theta[name], float),
                      "lo": np.full(len(U.KNOTS[name]), np.nan),
                      "hi": np.full(len(U.KNOTS[name]), np.nan)}
               for name in ("qnom", "kpod", "freq")}}
        for s, f in r.fits.items()}}


def u_knot_theta(lay: dict, stratum: str, name: str) -> np.ndarray:
    """Knot θ for one arm, with the operator's nameplate rotation applied if one is set.

    The rotation lives in ``session_state`` rather than in the loader so it reaches every tab —
    a view on the slope taken on the Qном tab has to show up in the Свод validation too,
    otherwise the validation is checking a different model from the one on screen.
    """
    th = np.asarray(lay[stratum][name]["theta"], float)
    if name != "qnom":
        return th
    fold = (st.session_state.get("qnom_fold") or {}).get(stratum)
    if fold is None:
        return th
    return U.rotate_qnom(th, float(fold), lay[stratum]["qnom"]["x"])


def u_theta(lay: dict, stratum: str, name: str, x) -> np.ndarray:
    """Polyline θ(x) — log-linear between knots, clamped flat outside (as fitted).

    Always returns at least 1-d: ``np.interp`` on a scalar yields a 0-d array, which then
    raises on ``[0]`` at the call sites that feed it one row at a time.
    """
    d = lay[stratum][name]
    xx = np.atleast_1d(np.asarray(x, float))
    return np.exp(np.interp(np.clip(xx, d["x"][0], d["x"][-1]), d["x"],
                            np.log(u_knot_theta(lay, stratum, name))))


def u_rmst(lay: dict, stratum: str, theta) -> np.ndarray:
    """RMST(0, 730) in DAYS under this stratum's own baseline.  θ multiplies; RMST does not."""
    d = lay[stratum]
    b0, e0 = d["beta0"], d["eta0"]
    return np.array([U.rmst(e0 * float(t) ** (-1.0 / b0), b0) for t in np.atleast_1d(theta)])


# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_original_freq() -> pd.DataFrame | None:
    f = ORIG_DIR / "ya_ttf_freq_multiplier.csv"
    return pd.read_csv(f) if f.exists() else None


@st.cache_data(show_spinner="deriving empirical curves + bootstrap …")
def load_empirical(kind: str):
    c = E.load_cached() if kind == "freq" else E.load_cached_kpod()
    if c is None:
        return None
    return {"all_pulls": c.all_pulls, "failures_only": c.failures_only,
            "band": c.band, "beta": c.beta,
            "polyline": c.polyline if c.polyline is not None else None}


@st.cache_data(show_spinner=False)
def vt_exposure(col: str) -> np.ndarray:
    try:
        vt = V.prepare_frame()
        return pd.to_numeric(vt[col], errors="coerce").dropna().to_numpy(float)
    except Exception:
        return np.array([])


def rmst_mult(theta, stratum: str) -> np.ndarray:
    b0, e0 = BASE[stratum]
    r0 = V.rmst(e0, b0)
    return np.array([V.rmst(e0 * float(t) ** (-1.0 / b0), b0) / r0
                     for t in np.atleast_1d(theta)])


def theta_kpod_two_anchor(k, left: float, right: float,
                          plateau: tuple = (0.8, 0.9),
                          a_lo: float = 0.2, a_hi: float = 1.2, b2: float = 0.30,
                          clamp=(0.15, 1.8),
                          shape_lo: float = 2.0, shape_hi: float = 2.0) -> np.ndarray:
    """Bathtub with a flat plateau and a per-arm CURVATURE exponent (see theta_freq_arms).

    v = |ln(k / plateau edge)|;  θ = 1 + c·v^p/(1 + b₂v^p), c set so the anchor is hit.
    shape 1 = linear (corner), 2 = parabolic (default), 3-4 = flat-bottomed then steep.
    """
    kk = np.clip(np.asarray(k, float), clamp[0], clamp[1])
    p_lo, p_hi = float(min(plateau)), float(max(plateau))
    pl, ph = max(float(shape_lo), 1.0), max(float(shape_hi), 1.0)
    v_lo, v_hi = abs(np.log(a_lo / p_lo)), abs(np.log(a_hi / p_hi))
    c_l = (left - 1.0) * (1.0 + b2 * v_lo ** pl) / v_lo ** pl if v_lo > 1e-9 else 0.0
    c_r = (right - 1.0) * (1.0 + b2 * v_hi ** ph) / v_hi ** ph if v_hi > 1e-9 else 0.0
    out = np.ones_like(kk)
    lo_m, hi_m = kk < p_lo, kk > p_hi
    vl = np.where(lo_m, np.log(p_lo / np.maximum(kk, 1e-9)), 0.0)
    vr = np.where(hi_m, np.log(np.maximum(kk, 1e-9) / p_hi), 0.0)
    out = np.where(lo_m, 1.0 + c_l * vl ** pl / (1.0 + b2 * vl ** pl), out)
    out = np.where(hi_m, 1.0 + c_r * vr ** ph / (1.0 + b2 * vr ** ph), out)
    return out


def draw(x, curves, pts, band, expo, xlabel, stratum, ylim=None, xline=None):
    fig, ax = plt.subplots(1, 2, figsize=(15.0, 5.6))
    for a, mode in zip(ax, ("theta", "rmst")):
        if band is not None and st.session_state.get("show_band", True):
            for nm, col in (("failures_only", "#e67e22"),):   # all-pulls hidden
                b = band[band.variant == nm].sort_values("f")
                if not len(b):
                    continue
                lo, hi = (b.lo, b.hi) if mode == "theta" else (rmst_mult(b.hi, stratum),
                                                               rmst_mult(b.lo, stratum))
                a.fill_between(b.f, lo, hi, color=col, alpha=.11, lw=0, zorder=1,
                               label=f"бутстрэп 95% — {nm}")
        for nm, (xs, ys, col, style, lw) in curves.items():
            yy = ys if mode == "theta" else rmst_mult(ys, stratum)
            a.plot(xs, yy, color=col, lw=lw, ls=style, zorder=4, label=nm)
        for nm, (xs, ys, col, sz) in pts.items():
            yy = ys if mode == "theta" else rmst_mult(ys, stratum)
            a.scatter(xs, yy, s=sz, facecolor=col, edgecolor="white", lw=1.2,
                      alpha=.85, zorder=6, label=nm)
        if len(expo):
            h = a.twinx()
            h.hist(expo, bins=40, color="#666", alpha=.14, zorder=0)
            h.set_yticks([]); h.set_ylim(0, None)
        a.axhline(1.0, color="#888", lw=.9)
        for xv in (xline or []):
            a.axvline(xv, ls=":", color="#555", lw=1.0)
        a.grid(alpha=.2); a.set_axisbelow(True)
        a.set_xlabel(xlabel)
        a.set_ylabel("θ (множитель риска)" if mode == "theta"
                     else f"множитель RMST(0,730) — {stratum}")
        a.set_title("θ" if mode == "theta" else "множитель наработки", fontsize=11)
        if ylim and mode == "theta":
            a.set_ylim(*ylim)
        a.legend(fontsize=7.5, loc="upper left" if mode == "theta" else "lower left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
st.title("Vt — слои частоты и Kpod")
stratum = st.sidebar.selectbox("страта (для множителя RMST)", list(BASE), index=0)
st.session_state["show_band"] = st.sidebar.checkbox("показывать бутстрэп-зону", True)
st.session_state["show_uv4"] = st.sidebar.checkbox(
    "показывать полилинии unified v4", True,
    help="Подобранная свободная полилиния (форма не задана априори). Справочная кривая — "
         "управляемые формы частоты и Kpod ниже она НЕ меняет.")
tab_f, tab_k, tab_q, tab_v = st.tabs(
    ["частота", "Kpod", "Qном + подрядчик (unified v4)", "проверка на Своде"])

# ------------------------------------------------------------------ frequency
with tab_f:
    emp = load_empirical("freq")
    orig = load_original_freq()
    st.caption(
        f"**Полиномиальная форма в log θ на {E.FREQ_CLAMP[0]:.0f}–{E.FREQ_CLAMP[1]:.0f} Гц**, "
        "заданная тремя значениями θ@40 / θ@50 / θ@60.  Три опоры определяют параболу "
        "ОДНОЗНАЧНО (n = 3) — свободных форм-параметров не остаётся.  n = 4 — это ЧЕТВЁРТАЯ "
        f"ОПОРА θ@{E.FREQ_ANCHOR4:.0f}: первые три держатся точно, а кубический член "
        "подбирается под неё.  (Сам коэффициент d на ползунок не выносится: он крайне "
        "нелинеен — d = 1.5·10⁻³ уводит θ(70) к 2·10⁴ — и двигает кривую не только в хвостах, "
        "но и между опорами.)"
    )
    c1, c2, c3 = st.columns(3)
    f40 = c1.slider("θ при 40 Гц (недокрут)", 0.50, 3.00, 1.45, 0.01)
    f50 = c2.slider("θ при 50 Гц (номинал)", 0.50, 2.00, 1.00, 0.01,
                    help="≠ 1 сдвигает УРОВЕНЬ всего слоя, а не только глубину ямы — "
                         "базовая линия η₀ должна это поглотить.")
    f60 = c3.slider("θ при 60 Гц (перекрут)", 0.50, 3.00, 1.20, 0.01)
    cq1, cq2 = st.columns([1, 2])
    npar = cq1.radio("число параметров", ["n = 3 (парабола)", "n = 4 (4-я опора)"], index=0)
    _t70_quad = float(E.theta_freq_poly(E.FREQ_ANCHOR4, f40, f50, f60)[0])
    if npar.startswith("n = 4"):
        t70 = cq2.slider(f"θ при {E.FREQ_ANCHOR4:.0f} Гц (4-я опора)", 0.50, 6.00,
                         float(np.clip(round(_t70_quad, 2), 0.5, 6.0)), 0.05,
                         help="Значение параболы здесь = ровно n = 3 (d = 0).")
        fcub = E.cubic_from_fourth_anchor(f40, f50, f60, t70)
    else:
        fcub = 0.0
    lo_hz, hi_hz = E.FREQ_CLAMP        # fixed by decision — the clamp is load-bearing here

    f = np.linspace(lo_hz - 2.0, hi_hz + 2.0, 700)
    _qkw = dict(cubic=fcub, clamp=(lo_hz, hi_hz))
    edges = E.freq_poly_edges(f40, f50, f60, fcub, (lo_hz, hi_hz))
    curves = {f"полином: θ40={f40:.2f}, θ50={f50:.2f}, θ60={f60:.2f}"
              + (f", θ70={t70:.2f}" if fcub else ""):
              (f, E.theta_freq_poly(f, f40, f50, f60, **_qkw), "#c0392b", "-", 3.0),
              "прежние плечи (для сравнения)":
              (f, E.theta_freq_arms(f, f40, f60, clamp=(lo_hz, hi_hz)),
               "#2e86c1", (0, (2, 2)), 1.4),
              "Ya V3 — как сейчас (полюс 31.7 Гц)":
              (f, np.array([float(CM.theta_freq(x)) for x in f]), "#111111", (0, (4, 2)), 1.6)}
    if emp and emp.get("polyline") is not None:
        p = emp["polyline"]
        curves["Ya полилиния (v2 tent)"] = (p.f, p.theta, "#7d3c98", (0, (1, 2)), 1.8)
    if st.session_state.get("show_uv4", True):
        # Reference only — the unified fit is on the DEVIATION from each run's own nominal,
        # so it is drawn against 50 + Δf.  It does not drive the controllable form above.
        _u = unified_layers()
        for _s, _lab in (("Ya", "Ya"), ("Vt_nonsour", "Vt")):
            _d = _u[_s]["freq"]
            curves[f"unified v4 — свободная полилиния, {_lab} (ось = 50 + Δf)"] = (
                50.0 + _d["x"], _d["theta"], U_COLOR[_s], (0, (5, 2)), 2.0)

    pts, band = {}, (emp["band"] if emp else None)
    # "оригинал" points removed on request — the re-derivation reproduces them to
    # mean |Δmult| 0.067 and is the only variant carrying a bootstrap band.
    if emp:
        for k, lab, c in (("failures_only", "эмпирика failures-only", "#e67e22"),):
            d = emp[k]
            pts[lab] = (d.f.to_numpy(float), d.theta.to_numpy(float), c,
                        np.clip(d.n.to_numpy(float), 20, 200))
    st.pyplot(draw(f, curves, pts, band, vt_exposure("freq_run"), "частота, Гц",
                   stratum, (0.5, 3.0), [50, 60]))
    g = np.array([35, 40, 42, 45, 50, 55, 60, 65, 70.])
    th_g = E.theta_freq_poly(g, f40, f50, f60, **_qkw)
    st.dataframe(pd.DataFrame({
        "f": g,
        "θ": np.round(th_g, 3),
        "RMST мн.": np.round(rmst_mult(th_g, stratum), 3),
    }).set_index("f").T, use_container_width=True)

    k = E.freq_poly_coeffs(f40, f50, f60, fcub)
    st.caption(
        f"`ln θ = {k['a']:+.6f} {k['b']:+.6f}·u {k['c']:+.8f}·u²"
        + (f" {k['d']:+.8f}·u³" if fcub else "") + "`,  `u = f − 50`,  обрезка "
        f"{lo_hz:.0f}–{hi_hz:.0f} Гц.  "
        f"**{edges['kind']}**, оптимум {edges['opt_hz']:.1f} Гц (θ={edges['theta_opt']:.3f}); "
        f"на краях θ({lo_hz:.0f})={edges['theta_lo_edge']:.3f}, "
        f"θ({hi_hz:.0f})={edges['theta_hi_edge']:.3f}. "
        "Обрезка здесь несущая: полином неограничен, за опорами он ускоряется.")

    with st.expander("подбор θ@40 / θ@50 / θ@60 под существующие кривые (МНК в log θ)"):
        tg = {}
        if emp:
            d_ = emp["failures_only"]
            tg["эмпирика Ya failures-only"] = (d_.f.to_numpy(float), d_.theta.to_numpy(float),
                                               d_.n.to_numpy(float))
            if emp.get("polyline") is not None:
                tg["Ya полилиния (v2 tent)"] = (emp["polyline"].f, emp["polyline"].theta, None)
        tg["шиппед θ_freq (Ya V3)"] = (f, np.array([float(CM.theta_freq(x)) for x in f]), None)
        if st.session_state.get("show_uv4", True):
            _u = unified_layers()
            for _s in ("Ya", "Vt_nonsour"):
                _d = _u[_s]["freq"]
                tg[f"unified v4 {_s}"] = (50.0 + _d["x"], _d["theta"], None)
        rows = []
        for nm, (xf, xt, xw) in tg.items():
            for deg, lab in ((2, "n = 3"), (3, "n = 4")):
                r = E.fit_freq_poly(xf, xt, xw, degree=deg)
                rows.append({"цель": nm, "форма": lab, "θ@40": round(r["t40"], 3),
                             "θ@50": round(r["t50"], 3), "θ@60": round(r["t60"], 3),
                             "d": round(r["cubic"], 6), "RMSE(log)": round(r["rmse_log"], 4)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Четвёртый параметр даёт 0.4–4 % RMSE на всех реальных целях — n = 3 "
                   "остаётся рабочей формой.")

# ----------------------------------------------------------------------- Kpod
with tab_k:
    empk = load_empirical("kpod")
    c1, c2, c3 = st.columns(3)
    left = c1.slider("θ при Kpod = 0.2 (недогруз)", 0.8, 2.5, 1.15, 0.01)
    right = c2.slider("θ при Kpod = 1.2 (перегруз)", 0.8, 3.0, 1.50, 0.01)
    plateau = c3.slider("плато оптимума (θ = 1)", 0.30, 1.40, (0.80, 0.90), 0.01)
    ks1, ks2 = st.columns(2)
    ksh_lo = ks1.slider("форма левого плеча Kpod", 1.0, 4.0, 2.0, 0.1)
    ksh_hi = ks2.slider("форма правого плеча Kpod", 1.0, 4.0, 2.0, 0.1)

    k = np.linspace(0.15, 1.8, 600)
    curves = {f"управляемая: {left:.2f} @0.2, {right:.2f} @1.2":
              (k, theta_kpod_two_anchor(k, left, right, plateau, shape_lo=ksh_lo, shape_hi=ksh_hi), "#c0392b", "-", 3.0),
              "ванна — как сейчас": (k, np.asarray(CM.theta_kpod(k), float),
                                     "#111111", (0, (4, 2)), 1.6)}
    if st.session_state.get("show_uv4", True):
        # Reference only.  This is the curve that overturned the "Kpod ≡ 1" verdict: fitted
        # free it is MONOTONE RISING, a shape the bathtub above structurally cannot produce.
        _u = unified_layers()
        for _s, _lab in (("Ya", "Ya"), ("Vt_nonsour", "Vt")):
            _d = _u[_s]["kpod"]
            curves[f"unified v4 — свободная полилиния, {_lab}"] = (
                _d["x"], _d["theta"], U_COLOR[_s], (0, (5, 2)), 2.0)
    pts, band = {}, (empk["band"] if empk else None)
    if empk:
        for kk, lab, c in (("failures_only", "эмпирика failures-only", "#e67e22"),):
            d = empk[kk]
            pts[lab] = (d.f.to_numpy(float), d.theta.to_numpy(float), c,
                        np.clip(d.n.to_numpy(float), 20, 200))
    # y-floor lowered from 0.8 to 0.5 so the unified v4 polyline is visible: fitted free it
    # drops to 0.58 at Kpod 0.2, i.e. the protective half a θ≥1 bathtub can never show.
    st.pyplot(draw(k, curves, pts, band, vt_exposure("kpod_run"), "Kpod = Ql/Qном",
                   stratum, (0.5, 2.2), list(plateau)))
    g = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
    st.dataframe(pd.DataFrame({
        "Kpod": g,
        "θ": np.round(theta_kpod_two_anchor(g, left, right, plateau, shape_lo=ksh_lo, shape_hi=ksh_hi), 3),
        "RMST мн.": np.round(rmst_mult(theta_kpod_two_anchor(g, left, right, plateau, shape_lo=ksh_lo, shape_hi=ksh_hi), stratum), 3),
    }).set_index("Kpod").T, use_container_width=True)

# --------------------------------------------------------- Qном + contractor (unified v4)
with tab_q:
    ulay = unified_layers()
    st.caption(
        "**Опорный слой unified v4.**  Свободная полилиния: кусочно-линейная в log θ, "
        "закреплена в Qном = 250, вне узлов — плато.  Форма НЕ задана априори — она вышла "
        "монотонной сама, поэтому монотонное ограничение здесь оправдано (в отличие от Kpod "
        f"и частоты).  Уровни подрядчика — из окна перекрытия по Ql {U.CONTRACTOR_WINDOW}, "
        f"где все три подрядчика реально пересекаются.  Источник: `{ulay['src']}`.")

    cq = st.columns(3)
    contractor = cq[0].selectbox("подрядчик", ("brt", "slb", "oth"), index=0)
    show = cq[1].multiselect("страты", list(U_STRATA), default=list(U_STRATA))
    absolute = cq[2].checkbox("RMST в сутках (иначе — множитель)", True)

    st.markdown(
        f"**Поворот кривой — средний наклон, по одному параметру на страту.**  "
        f"Полилиния линейна в log θ по log Qном, поэтому «сделать круче» — это буквально "
        f"добавить постоянный наклон, то есть повернуть кривую вокруг опоры Qном = "
        f"{U.QNOM_REF:g}.  Наклон читается как **во сколько раз растёт θ на декаду Qном**.  "
        "Опора остаётся ровно 1, поэтому базовая линия и опорный RMST не пересчитываются "
        "втихую; локальная форма кривой сохраняется целиком — меняется только тренд.  "
        "Ниже опоры кривая при этом идёт в ПРОТИВОПОЛОЖНУЮ сторону: это и есть поворот.")
    ct = st.columns(len(U_STRATA))
    folds = {}
    for i, s in enumerate(U_STRATA):
        f0 = U.qnom_fold_per_decade(ulay[s]["qnom"]["theta"], ulay[s]["qnom"]["x"])
        folds[s] = ct[i].slider(f"θ на декаду Qном — {s}", 0.60, 4.00,
                                float(round(f0, 2)), 0.01,
                                help=f"подобранный наклон {f0:.3f}× на декаду")
    st.session_state["qnom_fold"] = folds
    if any(abs(folds[s] - U.qnom_fold_per_decade(ulay[s]["qnom"]["theta"],
                                                 ulay[s]["qnom"]["x"])) > 5e-3
           for s in U_STRATA):
        st.info("Наклон задан вручную — бутстрэп-полоса ниже относится к ПОДОБРАННОЙ кривой, "
                "а не к текущей.")

    qg = np.geomspace(U.QNOM_KNOTS[0], U.QNOM_KNOTS[-1], 400)
    figq, axq = plt.subplots(1, 2, figsize=(15.0, 5.6))
    for s in show:
        d, c = ulay[s], U_COLOR[s]
        hr = d["level"].get(contractor, 1.0)
        th = u_theta(ulay, s, "qnom", qg) * hr
        kth = u_knot_theta(ulay, s, "qnom")
        axq[0].plot(qg, th, color=c, lw=2.4, label=f"{s} × {contractor} (HR {hr:.2f})")
        axq[0].plot(d["qnom"]["x"], kth * hr, "o", color=c, ms=6)
        if not np.allclose(kth, d["qnom"]["theta"]):     # show what was fitted, faintly
            axq[0].plot(d["qnom"]["x"], d["qnom"]["theta"] * hr, ":", color=c, lw=1.2,
                        alpha=.65, label=f"{s} — подобранная")
        if st.session_state.get("show_band", True) and np.isfinite(d["qnom"]["lo"]).any():
            axq[0].fill_between(d["qnom"]["x"], d["qnom"]["lo"] * hr, d["qnom"]["hi"] * hr,
                                color=c, alpha=.13, lw=0)
        # θ multiplies, RMST does not — convert once, at the end, per stratum baseline.
        r = u_rmst(ulay, s, th)
        axq[1].plot(qg, r if absolute else r / d["rmst_ref"], color=c, lw=2.4,
                    label=f"{s}  (опора {d['rmst_ref']:.0f} сут)")
    for a in axq:
        a.set_xscale("log")
        a.set_xticks(list(U.QNOM_KNOTS))
        a.set_xticklabels([f"{k_:g}" for k_ in U.QNOM_KNOTS], fontsize=8)
        a.minorticks_off()
        a.grid(alpha=.25); a.set_axisbelow(True)
        a.set_xlabel("Qном, м³/сут")
        a.legend(fontsize=8)
    axq[0].axhline(1.0, color="#888", lw=.9)
    axq[0].axvline(U.QNOM_REF, ls=":", color="#555", lw=1.0)
    axq[0].set_ylabel("θ — множитель интенсивности отказов")
    axq[0].set_title(f"θ_Qном × подрядчик ({contractor})", fontsize=11)
    axq[1].set_ylabel("RMST(0,730), сут" if absolute else "множитель RMST(0,730)")
    axq[1].set_title("наработка при данном Qном" if absolute else "множитель наработки",
                     fontsize=11)
    if not absolute:
        axq[1].axhline(1.0, color="#888", lw=.9)
    figq.tight_layout()
    st.pyplot(figq)

    st.markdown("**Базовые линии и уровни подрядчика** — уровни оценены один раз на окне "
                "перекрытия и далее зафиксированы, чтобы рейт не утекал в подрядчика.")
    st.dataframe(pd.DataFrame(
        [{"страта": s, "n": ulay[s]["n"], "отказов": ulay[s]["events"],
          "β₀": round(ulay[s]["beta0"], 3), "η₀": round(ulay[s]["eta0"], 1),
          "RMST₇₃₀ опора, сут": round(ulay[s]["rmst_ref"], 1),
          **{f"HR {c}": round(ulay[s]["level"][c], 3) for c in ("brt", "slb", "oth")}}
         for s in U_STRATA]), use_container_width=True)

    st.markdown("**θ_Qном в узлах** (95 % — кластерный бутстрэп по скважинам; "
                "звёздочка — узел, сдвинутый настройкой хвоста)")
    st.dataframe(pd.DataFrame(
        {"Qном": [f"{k_:g}" for k_ in U.QNOM_KNOTS],
         **{s: [(f"{t:.2f}*" if abs(t - f) > 5e-3 else f"{t:.2f}")
                + (f" [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else "")
                for t, f, lo, hi in zip(u_knot_theta(ulay, s, "qnom"),
                                        ulay[s]["qnom"]["theta"],
                                        ulay[s]["qnom"]["lo"], ulay[s]["qnom"]["hi"])]
            for s in U_STRATA}}
    ).set_index("Qном").T, use_container_width=True)

st.sidebar.caption(
    "Оригинальные кривые — из `production_risk_ya_freq_kpod_shape/2026-07-24` "
    "(в множителях наработки, переведены в хазард как θ=(1/mult)^β). "
    "Бутстрэп и переоценка — `ya_freq_empirical.py`.  "
    "Слой Qном + подрядчик и пунктирные справочные полилинии — `unified_v4.py`."
)


# ------------------------------------------------- Свод validation (v1)
@st.cache_data(show_spinner="loading Свод panel …")
def load_svod(variant: str = "failures") -> pd.DataFrame | None:
    """``variant`` is a LOADER argument, not a row filter — the panel it returns is already
    restricted, so filtering on ``event`` afterwards is a no-op (every row has event == 1)."""
    try:
        from analysis.workflows.production_risk.svod_nno_decomposition import load_panel
        p = load_panel(variant=variant)
        p["stratum"] = np.where(p["h2s_class"].eq("sour"), "sour", "nonsour")
        return p
    except Exception:
        return None


@st.cache_data(show_spinner="fitting Ya layers …")
def ya_layers():
    """Ya's OWN rate (Ql) and contractor layers — Vt's θ_Qnom does not apply to Ya wells."""
    from analysis.workflows.production_risk import ya_k1k2_hybrid as YA
    r = YA.run(rate="Ql", write=False)
    m = r.model
    grid = np.geomspace(10.0, 2000.0, 240)          # tabulated: a lambda cannot be cached
    return {"beta": float(m.baseline.beta1), "eta1": float(m.baseline.eta1),
            "hr": {c: float(m.contractor_hr(c)) for c in ("brt", "slb", "oth")},
            "ql_grid": grid,
            "ql_theta": np.asarray(m.theta_at("Ql", grid), float)}


@st.cache_data(show_spinner=False)
def vt_v4_layers():
    """Baselines, contractor levels and θ_Qnom knots from the shipped v4 fit."""
    r = V.run(write=False)
    return {s: {"beta0": r.model.fits[s].beta0, "eta0": r.model.fits[s].eta0,
                "level": dict(r.model.fits[s].level),
                "qnom_theta": r.model.fits[s].theta_qnom.tolist()} for s in V.STRATA}


with tab_v:
    st.caption(
        "Слои **частоты и Kpod — управляемые формы с соседних вкладок** (не тронуты).  "
        "Рейт и подрядчик — переключатель ниже: **unified v4** даёт θ_Qном и уровни ОБОИМ "
        "полям на одной структуре; **прежние** брали θ_Ql из Ya v2 для Ya и θ_Qном из Vt v4 "
        "для Vt.  Данные — Свод (ННО пробега)."
    )
    _probe = load_svod("failures")
    if _probe is None:
        st.error("Свод-панель недоступна")
    else:
        cc = st.columns(5)
        which = cc[0].selectbox("данные", ["Ya", "Vt", "Vt_nonsour", "Vt_sour"], index=0)
        xaxis = cc[4].selectbox("ось X", ["частота", "Ql"], index=0)
        rows_kind = cc[1].selectbox("строки (знаменатель Свода)",
                                    ["только отказы", "все съёмы"], index=0)
        nbin = cc[2].slider("частотных корзин", 4, 14, 8)
        ycap = cc[3].slider("предел оси ННО, сут", 300, 1500, 900, 50)
        uc1, uc2 = st.columns([1, 3])
        src = uc1.radio("рейт + подрядчик", ["unified v4", "прежние (Ya v2 / Vt v4)"], index=0)
        use = uc2.multiselect(
            "накладывать слои",
            ["частота", "Kpod", "рейт", "подрядчик"], default=["частота", "рейт", "подрядчик"])

        panel = load_svod("failures" if rows_kind == "только отказы" else "all_closed")
        fld = "Ya" if which == "Ya" else "Vt"
        g = panel[panel.field.eq(fld)].copy()
        if which == "Vt_nonsour":
            g = g[g.stratum.eq("nonsour")]
        elif which == "Vt_sour":
            g = g[g.stratum.eq("sour")]
        # Ql taken as Qном × Кпод, as requested.  NB: in this panel ``kpod`` is defined as
        # ql/qnom, so the product reproduces the ``ql`` column exactly (max |Δ| = 0.0) — the
        # derivation is explicit provenance, not a different quantity.
        g["ql_calc"] = g["qnom"] * g["kpod"]
        xcol, xlab, xrange, xref = (("freq", "частота, Гц", (30.0, 70.0), 50.0)
                                    if xaxis == "частота"
                                    else ("ql_calc", "Ql = Qном × Кпод, м³/сут",
                                          (0.0, 1200.0), 250.0))
        # Drop rows missing ANY model input, not just the x-axis one: on the Ql axis the
        # freq-less rows used to survive, turn θ_freq into NaN, poison the global scale and
        # blank out BOTH curves — leaving only the heatmap.
        g = g.dropna(subset=[xcol, "nno", "freq", "kpod", "ql", "qnom"])
        g = g[(g[xcol].between(*xrange)) & (g.nno > 0)]

        if len(g) < 30:
            st.warning(f"мало строк: {len(g)}")
        else:
            lay = vt_v4_layers()
            b0 = float(np.mean([lay[s]["beta0"] for s in lay]))
            # θ is split in two: the BASE (contractor + rate) sets the scale, the OVERLAY
            # (freq + Kpod) is then applied on top of the already-calibrated base.  Folding
            # the overlay into the normalisation would let the scale silently absorb a wrong
            # frequency layer, which is exactly what we are trying to test.
            th_base = np.ones(len(g))
            th_over = np.ones(len(g))
            if "частота" in use:
                th_over = th_over * E.theta_freq_poly(g.freq.to_numpy(float),
                                                      f40, f50, f60, **_qkw)
            if "Kpod" in use:
                th_over = th_over * theta_kpod_two_anchor(g.kpod.to_numpy(float), left, right,
                                                          plateau, shape_lo=ksh_lo, shape_hi=ksh_hi)
            th = th_base
            # Which stratum of the chosen model each Свод row belongs to.  Ya has a single
            # H₂S class, so both Свод strata map onto the one Ya stratum.
            u_str = np.array([U_OF_SVOD[(fld, s_)] for s_ in g.stratum])
            if src == "unified v4":
                ulay = unified_layers()
                # ONE structure for both fields: nameplate rate, not realised Ql.
                if "рейт" in use:
                    qn = g.qnom.to_numpy(float)
                    tq = np.ones(len(g))
                    for s_ in np.unique(u_str):          # per-stratum, not per-row
                        m = u_str == s_
                        tq[m] = u_theta(ulay, s_, "qnom", qn[m])
                    th = th * tq
                if "подрядчик" in use:
                    th = th * np.array([ulay[s_]["level"].get(c, 1.0)
                                        for c, s_ in zip(g.contractor, u_str)])
            elif fld == "Ya":
                ya = ya_layers()
                b0 = ya["beta"]
                if "рейт" in use:      # Ya's own Ql layer
                    th = th * np.exp(np.interp(np.log(np.clip(g.ql.to_numpy(float), 10, 2000)),
                                               np.log(ya["ql_grid"]), np.log(ya["ql_theta"])))
                if "подрядчик" in use:
                    th = th * np.array([ya["hr"].get(c, 1.0) for c in g.contractor])
            else:
                if "рейт" in use:      # Vt v4 nameplate layer
                    qn = g.qnom.to_numpy(float)
                    th = th * np.array([float(np.exp(np.interp(
                        np.clip(q, V.QNOM_GUARD, V.QNOM_KNOTS[-1]),
                        np.asarray(V.QNOM_KNOTS, float),
                        np.log(lay[st_]["qnom_theta"])))) for q, st_ in zip(qn, g.stratum)])
                if "подрядчик" in use:
                    th = th * np.array([lay[st_]["level"].get(c, 1.0)
                                        for c, st_ in zip(g.contractor, g.stratum)])
            # Per-run life from the field's OWN baseline, per stratum:
            #     η_eff = η₀(stratum) · θ^(−1/β₀(stratum));  mean life = η_eff · Γ(1 + 1/β₀)
            # so the sour/nonsour level difference comes from the MODEL, not from the scaling.
            # A single global factor then absorbs the definitional gap between ННО and the
            # model's mean life — it cannot manufacture a between-stratum difference.
            # RMST(0,730), NOT the Weibull mean.  With β < 1 (Ya: 0.787) the distribution
            # mean is dominated by an unobservable right tail — 807 d against an observed
            # mean of ~390 — and the scale factor would silently absorb the ~2× gap.
            # RMST is the project's standing reporting quantity for exactly this reason.
            if src == "unified v4":
                ulay = unified_layers()
                bb = np.array([ulay[s_]["beta0"] for s_ in u_str])
                ee = np.array([ulay[s_]["eta0"] for s_ in u_str])
                b0 = float(np.mean(bb))
            elif fld == "Ya":
                bb = np.full(len(g), b0)
                ee = np.full(len(g), float(ya_layers()["eta1"]))
            else:
                bb = np.array([lay[s_]["beta0"] for s_ in g.stratum])
                ee = np.array([lay[s_]["eta0"] for s_ in g.stratum])
            _rm = np.vectorize(lambda e_, b_: V.rmst(e_, b_))
            life_base = _rm(ee * th ** (-1.0 / bb), bb)            # contractor + rate only
            scale = float(g.nno.mean() / np.nanmean(life_base))    # calibrated on the BASE
            life_full = _rm(ee * (th * th_over) ** (-1.0 / bb), bb)  # overlay on top
            g["model_ttf"] = life_full * scale
            g["model_base"] = life_base * scale

            edges = np.quantile(g[xcol], np.linspace(0, 1, nbin + 1))
            edges = np.unique(np.round(edges, 2))
            g["bin"] = pd.cut(g[xcol], edges, include_lowest=True)
            agg = g.groupby("bin", observed=True).agg(
                f=(xcol, "mean"), obs=("nno", "mean"), mod=("model_ttf", "mean"),
                n=("nno", "size")).dropna()

            # Column-normalised density, as in the parallel Свод analysis: each frequency
            # column is divided by its own count, so a column reads as the conditional
            # distribution of ННО at that frequency, not "how many wells sit here".
            xe = np.linspace(xrange[0], xrange[1], 27)
            ye = np.linspace(0, ycap, 26)
            counts, _, _ = np.histogram2d(g[xcol].to_numpy(float),
                                          np.clip(g.nno.to_numpy(float), 0, ycap),
                                          bins=[xe, ye])
            col_n = counts.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.where(col_n[:, None] > 0, counts / col_n[:, None], np.nan)
            z[col_n < 8, :] = np.nan          # thin columns stay blank, never coloured
            z[counts == 0] = np.nan
            fig, ax = plt.subplots(figsize=(13.5, 6.0))
            pc = ax.pcolormesh(xe, ye, np.ma.masked_invalid(z.T * 100.0), cmap="Blues",
                               shading="flat", zorder=1, linewidth=0, vmin=0.0)
            fig.colorbar(pc, ax=ax,
                         label="доля пробегов колонки, % (сумма по колонке = 100)")
            fx = agg["f"].to_numpy(float)
            fo = agg["obs"].to_numpy(float)
            fm = agg["mod"].to_numpy(float)
            ax.plot(fx, fo, "o-", color="#c0392b", lw=2.6, ms=8, zorder=6,
                    label="ННО факт (среднее по корзине)")
            ax.plot(fx, fm, "s--", color="#117a65", lw=2.4, ms=7, zorder=6,
                    label="ННО модель (слои: " + ", ".join(use) + ")")
            for xx, yy, nn in zip(fx, fo, agg["n"].to_numpy(int)):
                ax.annotate(f"n={nn}", (xx, yy), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7.5, color="#c0392b")
            ax.axvline(xref, ls=":", color="#555", lw=1.0)
            ax.set_xlim(*xrange); ax.set_ylim(0, ycap)
            ax.set_xlabel(xlab); ax.set_ylabel("ННО, сут")
            ax.set_title(f"{which} — ННО vs {xaxis} ({rows_kind}, n={len(g)})", fontsize=11)
            ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=.2)
            st.pyplot(fig)
            tbl = pd.DataFrame({"f": agg["f"].round(1), "n": agg["n"],
                                "факт": agg["obs"].round(0), "модель": agg["mod"].round(0),
                                "Δ": (agg["mod"] - agg["obs"]).round(0)})
            st.dataframe(tbl.reset_index(drop=True), use_container_width=True)
            st.caption(
                f"**TTF₀ = среднее ННО этой же выборки ({g.nno.mean():.0f} сут)** — модель "
                f"перенормирована на него, поэтому проверяется ТОЛЬКО ФОРМА по частоте, а не "
                f"уровень: страта с неверной абсолютной наработкой здесь всё равно выглядела бы "
                f"хорошо; ось X = {xlab} (масштаб ×{scale:.2f} откалиброван ТОЛЬКО на "
                f"подрядчике+рейте, частота и Кпод наложены сверху; уровни СТРАТ берутся "
                f"из базовых линий модели, а не из масштаба). β={b0:.3f}. "
                f"Доля строк выше предела оси: "
                f"{(g.nno > ycap).mean()*100:.1f}%. "
                f"Слои: рейт = " + ("θ_Qном (unified v4)" if src == "unified v4"
                                    else ("θ_Ql (Ya v2)" if fld == "Ya" else "θ_Qном (Vt v4)")) + ".")
