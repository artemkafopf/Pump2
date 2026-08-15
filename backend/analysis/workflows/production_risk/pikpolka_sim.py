"""The ПикПолка / NPV_УВЧ calculator, in Python — so its inputs can be swept.

The two Excel deliverables answer one well at a time: type in a rate, a pump, a contractor and
a frequency, read ННО and NPV. Everything a *study* needs — how the answer moves across the
input space, where the sign of Δ NPV flips, which assumption the conclusion actually rests on —
is out of reach there, because every point costs a recalculation by hand.

This is a faithful port of the sheet's engine, not an approximation of it:

* the hazard layers are :mod:`ModuleFailureV4` line for line (contractor × θ_Qnom × θ_freq ×
  θ_Kpod, η_eff = η₀·θ^(−1/β₀), life = RMST(0,730), η₀ back-solved from ``RMST_ref``);
* the day grid reproduces the sheet's columns — decline, water cut off the ХВ recovery curve,
  the nominal re-pick at each restart, Kpod = Ql/nominal, the consume-and-renew failure rule,
  the uptime window, costs, discounting;
* the reference tables (ХВ curve, net-backs, rent/КО by pump × contractor, the nominal ladder,
  downtime days) are **read out of the workbook**, never re-typed here.

Validated against the workbook's own cached values — see :func:`validate_against_workbook`.

Usage::

    cfg = load_config(r"...\\Копия Калькулятор_NPV_УВЧ_V02.xlsm")
    base = simulate(cfg)                       # one case
    grid = sweep(cfg, freq_alt=[50, 55, 60, 65], ql=[400, 800, 1200])

⚠ The deployed θ_Kpod is the operator's **bathtub**, which disagrees with the fitted layer at
low Kpod (the fit is monotone rising).  That is a deliberate deployment choice; a sweep that
touches Kpod is exploring the assumption, not the data.  Pass ``ctrl=fitted_kpod_ctrl()`` to
see the other side.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# --- layer constants: unified v4 as deployed in ModuleFailureV4 --------------
RMST_HORIZON = 730.0
QNOM_KNOTS = (60.0, 100.0, 160.0, 250.0, 400.0, 640.0, 1000.0, 1600.0)

#: ``TuneBlock`` rows: key -> (β, RMST_ref days, brt, slb, oth).
TUNE: dict[str, tuple] = {
    "Vt_nonsour": (1.2511259, 407.284, 1.0, 1.19994, 1.90976),
    "Vt_sour": (1.3447649, 183.8034, 1.0, 1.19994, 1.90976),
}

#: ``QnomBlock`` rows: key -> θ at :data:`QNOM_KNOTS`.  Sour's top knot carries the deployment
#: clamp (fitted 1.2034 → 1.3668): free-fitted the arm turns down above 1000 on 20 runs.
QNOM: dict[str, tuple] = {
    "Vt_nonsour": (0.817254, 0.754386, 0.91978, 1.0, 1.30472, 1.635608, 1.63824, 1.826061),
    "Vt_sour": (0.555767, 0.790572, 1.051675, 1.0, 1.11554, 1.334801, 1.366776, 1.366776),
}

#: The block as it stood **before** the sour deployment clamp — kept so the port can reproduce
#: a workbook that has not been rewired yet, and so the cost of the clamp is measurable rather
#: than asserted.
QNOM_PRECLAMP: dict[str, tuple] = {
    **QNOM,
    "Vt_sour": (0.555767, 0.790572, 1.051675, 1.0, 1.11554, 1.334801, 1.366776, 1.203401),
}

#: ``CtrlBlock`` — operator-set, NOT fitted.  θ_freq(50) = 0.9, not 1.
CTRL_DEFAULT = {"t40": 1.8, "t50": 0.9, "t60": 1.3, "t70": 3.75,
                "k02": 2.0, "k12": 1.25, "kpl_lo": 0.7, "kpl_hi": 0.95,
                "ksh_lo": 2.0, "ksh_hi": 2.0}

FREQ_LO, FREQ_HI = 35.0, 70.0
FREQ_A1, FREQ_A2, FREQ_A3, FREQ_A4 = 40.0, 50.0, 60.0, 70.0
KPOD_B2, KPOD_LO, KPOD_HI = 0.3, 0.15, 1.8
KPOD_A_LO, KPOD_A_HI = 0.2, 1.2


#: Scenario 0 — both operator layers switched off, θ_freq ≡ 1 and θ_Kpod ≡ 1.
#:
#: The statistically defensible floor, not a curiosity: fitted free, frequency is a null in
#: every field measured (CV Vt −0.60, Az −0.28, Ic −0.79, Ya +1.13, Za +1.93), so "no frequency
#: effect" is inside the evidence, while the deployed curve is a prior.  Running it says how
#: much of an answer rests on the two layers nobody measured.  With both Kpod anchors at 1 the
#: arm coefficient is exactly 0, and the plateau spans the whole clamp, so the arm cannot fire.
NULL_CTRL = {"t40": 1.0, "t50": 1.0, "t60": 1.0, "t70": 1.0,
             "k02": 1.0, "k12": 1.0, "kpl_lo": KPOD_LO, "kpl_hi": KPOD_HI,
             "ksh_lo": 1.0, "ksh_hi": 1.0}


#: Scenario 1 — «Базовый»: the censoring-aware reading.  Frequency is a smooth rendering of the
#: Ya transfer curve with θ(50) = 1 (a level, not a severity — the shipped 0.9 re-levelled the
#: whole layer); Kpod is the **fitted** arm, where a lightly loaded pump is protected.
CTRL_BASE = {"t40": 1.30, "t50": 1.00, "t60": 1.20, "t70": 1.80,
             "k02": 0.58, "k12": 1.22, "kpl_lo": 0.80, "kpl_hi": 0.80,
             "ksh_lo": 1.0, "ksh_hi": 1.0}

#: Scenario 2 — «Стресс»: the operator prior kept as the pessimistic arm, Kpod as the bathtub.
CTRL_STRESS = {"t40": 1.80, "t50": 1.00, "t60": 1.30, "t70": 3.75,
               "k02": 2.00, "k12": 1.25, "kpl_lo": 0.70, "kpl_hi": 0.95,
               "ksh_lo": 2.0, "ksh_hi": 2.0}

#: Selector value on the sheet → the CtrlBlock it resolves to.  Single definition: the deploy
#: scripts write these into the workbooks, and `freq_bins` reads them for the viewer, so the
#: three cannot drift apart.
CTRL_SCENARIOS = {0: NULL_CTRL, 1: CTRL_BASE, 2: CTRL_STRESS}

#: Which of the ten parameters are multipliers (geometric blend) — the rest are shape
#: parameters and blend linearly.  Order matches the sheet.
CTRL_ORDER = ("t40", "t50", "t60", "t70", "k02", "k12",
              "kpl_lo", "kpl_hi", "ksh_lo", "ksh_hi")
CTRL_MULTIPLIERS = CTRL_ORDER[:6]


def blend_ctrl(w: float = 0.5, base: dict | None = None, stress: dict | None = None) -> dict:
    """Scenario 3 — «Пользовательский»: ``B^(1−w)·C^w`` on the multipliers, linear on the rest.

    Geometric on the θ anchors because they compose multiplicatively and the layer is
    piecewise-linear in **log** θ — a linear average bends the curve into a shape neither
    scenario supports.  ⚠ The blend is on the INPUTS: ННО and NPV are non-linear in them, so
    the result need not sit between the two endpoints.
    """
    b = dict(base or CTRL_BASE)
    c = dict(stress or CTRL_STRESS)
    return {k: (b[k] ** (1 - w) * c[k] ** w if k in CTRL_MULTIPLIERS
                else b[k] + w * (c[k] - b[k])) for k in CTRL_ORDER}


def fitted_kpod_ctrl(ctrl: dict | None = None) -> dict:
    """CtrlBlock whose Kpod arm is disabled, for comparison against the fitted layer.

    The fitted Vt Kpod is monotone **rising** (0.58 at 0.2 → 1.27 at 1.6); the shipped bathtub
    penalises both ends.  There is no bathtub parameterisation that reproduces a monotone arm,
    so the honest comparison is bathtub vs flat, with the fitted arm applied separately.
    """
    c = dict(CTRL_DEFAULT if ctrl is None else ctrl)
    c.update(k02=1.0, k12=1.0, kpl_lo=KPOD_LO, kpl_hi=KPOD_HI)
    return c


# ---------------------------------------------------------------------------
# Layers — ModuleFailureV4, line for line
# ---------------------------------------------------------------------------
def theta_qnom(key: str, qnom, block: dict | None = None) -> float:
    """Piecewise-linear in log θ between the knots, flat below 60 and above 1600."""
    th = np.asarray((block or QNOM)[key], float)
    q = float(np.clip(qnom, QNOM_KNOTS[0], QNOM_KNOTS[-1]))
    return float(np.exp(np.interp(np.log(q), np.log(QNOM_KNOTS), np.log(th))))


#: v5.2 layer knots.  ``WCUT``/``KPOD_FIT`` stay EMPTY until a fitted block is supplied, so
#: importing this module changes nothing: an absent block means θ ≡ 1 for that layer.
WCUT_KNOTS = (0.0, 5.0, 25.0, 50.0, 80.0, 95.0)
KPOD_KNOTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6)
#: ⚠ Nine knots, not seven.  The frequency block was extended past 60 Hz (+15, +20) when the
#: layer was given a right arm; this tuple was left at seven and every caller that fed it a
#: shipped block died in ``np.interp`` with "fp and xp are not of the same length".
FREQ_KNOTS_FIT = (-22.0, -15.0, -9.0, -4.0, 0.0, 4.0, 9.0, 15.0, 20.0)
WCUT: dict[str, tuple] = {}
KPOD_FIT: dict[str, tuple] = {}
FREQ_FIT: dict[str, tuple] = {}


def theta_tab(x: float, knots, values, *, log_x: bool = False) -> float:
    """Piecewise-linear in **log θ** between knots, clamped flat outside them.

    The one interpolation rule every deployed layer uses — never splined, because a smooth
    interpolant invents curvature between knots that the fit never estimated.  ``log_x``
    interpolates against log x (right for rate-like axes such as Qnom), otherwise against x
    (right for the additive axes: water cut, Kpod, frequency deviation).
    """
    k = np.asarray(knots, float)
    th = np.asarray(values, float)
    v = float(np.clip(x, k[0], k[-1]))
    if log_x:
        return float(np.exp(np.interp(np.log(v), np.log(k), np.log(th))))
    return float(np.exp(np.interp(v, k, np.log(th))))


def theta_freq(f: float, ctrl: dict | None = None) -> float:
    """Polynomial in log θ through θ(40), θ(50), θ(60) plus an optional 70 Hz anchor.

    Taken in log θ because a plain polynomial through a valley crosses zero before 35 Hz, i.e.
    a negative hazard multiplier.  The 35–70 clamp is load-bearing, not cosmetic.
    """
    c = CTRL_DEFAULT if ctrl is None else ctrl
    if f <= 0 or min(c["t40"], c["t50"], c["t60"]) <= 0:
        return 1.0
    h = 0.5 * (FREQ_A3 - FREQ_A1)
    la, lb, lc = np.log(c["t40"]), np.log(c["t50"]), np.log(c["t60"])
    a = lb
    cc = (la + lc - 2 * lb) / (2 * h * h)
    b = (lc - la) / (2 * h)
    d = 0.0
    if c.get("t70", 0.0) > 0:
        u4 = FREQ_A4 - FREQ_A2
        den = u4 ** 3 - h * h * u4
        if abs(den) > 1e-9:
            d = (np.log(c["t70"]) - (a + b * u4 + cc * u4 * u4)) / den
    b -= d * h * h
    u = float(np.clip(f, FREQ_LO, FREQ_HI)) - FREQ_A2
    return float(max(np.exp(a + b * u + cc * u * u + d * u ** 3), 1e-6))


def theta_kpod(k: float, ctrl: dict | None = None) -> float:
    """Bathtub with a flat plateau (θ = 1 across kpl_lo..kpl_hi) and a curvature per arm."""
    c = CTRL_DEFAULT if ctrl is None else ctrl
    if k <= 0:
        return 1.0
    plo, phi = sorted((c["kpl_lo"], c["kpl_hi"]))
    if plo <= 0 or phi <= 0:
        return 1.0
    pl, ph = max(c["ksh_lo"], 1.0), max(c["ksh_hi"], 1.0)
    kk = float(np.clip(k, KPOD_LO, KPOD_HI))
    if plo <= kk <= phi:
        return 1.0
    v_lo, v_hi = abs(np.log(KPOD_A_LO / plo)), abs(np.log(KPOD_A_HI / phi))
    if v_lo < 1e-9 or v_hi < 1e-9:
        return 1.0
    c_l = (c["k02"] - 1.0) * (1 + KPOD_B2 * v_lo ** pl) / (v_lo ** pl)
    c_r = (c["k12"] - 1.0) * (1 + KPOD_B2 * v_hi ** ph) / (v_hi ** ph)
    if kk < plo:
        v = np.log(plo / kk)
        return float(max(1 + c_l * v ** pl / (1 + KPOD_B2 * v ** pl), 1e-6))
    v = np.log(kk / phi)
    return float(max(1 + c_r * v ** ph / (1 + KPOD_B2 * v ** ph), 1e-6))


# ---------------------------------------------------------------------------------------
# v5.2 shipped layer forms — the Python twin of ``vba/ModuleFailureV5.bas``
# ---------------------------------------------------------------------------------------
# The V4 closed forms above are the OPERATOR's shapes: the frequency cubic and the Kпод
# bathtub were defined by the anchors a person typed.  v5.2 fits both layers and ships the
# fitted curve as a rational in log θ, so the shape lives in code and the sheet keeps only
# levels.  Both files must agree to the last digit — ``scripts/deploy/check_module_v5.py``
# parses the .bas and asserts it against these constants rather than trusting either.
#
# Frequency (deviation axis):  u = clip(dev, −15, 20)/10,  log θ = P(u)/(1 + u·Q(u))
#   the numerator is NOT multiplied by u, so θ(0) ≈ 0.999 rather than exactly 1 → floored at 1
# Kпод and water cut:          u = (clip(x) − ref)/scale,  log θ = u·P(u)/(1 + u·Q(u))
#   pinned at the reference by construction (u = 0 ⇒ log θ = 0)
FREQ_RAT = {
    1: ((-0.0008212625912, 0.003900963071, 0.2193743376, -0.09858808433, 0.3041728279),
        (-0.4746388993, 1.848669429, 0.1245599728)),
    2: ((0.005526769523, -0.1571024758, 0.4723855629, 0.005712809603),
        (0.08027070802, 0.140555742, -0.06156767127)),
}
KPOD_RAT = {
    1: ((-0.044287585, 0.20526101), (-0.033041251, 0.97026351)),
    2: ((-0.024292629, 0.29992627), (0.23177276,)),
}
WCUT_RAT = ((0.14802941, -0.10012055), (0.24529177, -0.054688025))

FREQ_DEV_LO, FREQ_DEV_HI = -15.0, 20.0
FREQ_ANCHORS = (-10.0, 10.0)          # 40 and 60 Hz at a 50 Hz nominal
KPOD_REF_V52, KPOD_CLAMP_HI = 0.8, 1.8
KPOD_ANCHORS = (0.2, 1.6)
#: Нижняя обрезка Kпод — СВОЯ на каждый набор, и это не симметрия ради симметрии.
#:
#: «Базовый» продолжается до 0: его знаменатель на [0, 1.8] не опускается ниже 1, кривая
#: монотонна и от 1.087 на 0.4 доходит всего до 1.165 на нуле.  Продолжение важно не
#: величиной, а тем, что 28 % месячных интервалов фонда лежат ниже 0.4 — на плоском участке
#: якорь «θ_Kпод при 0.2» был мёртвым и задать глубину недогруза решением было нельзя.
#:
#: «Стресс» обрезан на 0.4 и должен остаться обрезанным: его знаменатель — многочлен ПЕРВОЙ
#: степени (1 + 0.2318·u), к Kпод = 0 он сжимается до 0.69 и раздувает числитель, давая
#: θ = 2.27.  Полюс лежит на Kпод = −1.79 — вне области, но именно он и загибает плечо.
#: Это число производит функциональная форма, а не данные: нижний равночисленный бин
#: начинается с Kпод 0.45, всё ниже — экстраполяция.
KPOD_CLAMP_LO = {1: 0.0, 2: 0.0}

#: Какой рациональной формой описывается ПЛЕЧО кривой Kпод.
#:
#: Нижнее плечо у ОБОИХ сценариев — стрессовой формы, и это следствие принятого решения
#: «θ(0) не меньше 1.5».  Базовая форма слишком полога (1.166 в нуле), и довести её до 1.5
#: экспонентой можно только с γ = 2.65 — а он задирает и середину плеча: θ(0.4) 1.087 → 1.246,
#: θ(0.6) 1.034 → 1.093, то есть «Базовый» обгоняет «Стресс» ровно там, где данные ЕСТЬ.
#: Приёмка это поймала (ННО «Базового» 247 против 270 у «Стресса»).
#:
#: Стрессовая форма круче у самого нуля и почти совпадает с базовой у опорной точки, поэтому
#: на ней то же решение достигается ослаблением (γ = 0.495), а не усилением: θ(0.4) выходит
#: 1.092 против 1.087 у подгонки — участок с данными практически не тронут, вся добавленная
#: строгость сидит ниже Kпод 0.3.  И порядок сценариев соблюдается ПО ПОСТРОЕНИЮ: одна форма,
#: у «Стресса» γ = 1, у «Базового» γ < 1, значит стресс не может оказаться мягче.
#:
#: Верхние плечи остаются своими: базовое 1.121 при Kпод 1.6, стрессовое 1.466.
KPOD_ARM = {1: {"lo": 2, "hi": 1}, 2: {"lo": 2, "hi": 2}}
WCUT_REF_V52, WCUT_ANCHOR = 25.0, 95.0

#: CtrlBlock as it ships in «МодельОтказов» 3.3 — nine values, in sheet order.  The anchors
#: are the fitted curve's OWN values, so γ = 1 and the deployed curve IS the fit.
#: ⚠ k02 = 1.2326 — это РЕШЕНИЕ, а не подгонка: принято, что полностью разгруженный насос
#: стоит не меньше 1.5 по интенсивности.  На стрессовой форме нижнего плеча (см. KPOD_ARM)
#: это соответствует γ = 0.495 и даёт θ: 0 → 1.500, 0.2 → 1.233, 0.4 → 1.092, 0.6 → 1.022.
#: Участок с данными почти не тронут (подгонка на 0.4 даёт 1.087), вся строгость ниже 0.3.
CTRL5_BASE = {"set": 1, "on_freq": 1, "t40": 1.2129, "t60": 1.1869,
              "on_kpod": 1, "k02": 1.2326, "k16": 1.1209,
              "on_wcut": 1, "w95": 0.855}
#: ⚠ k02 = 1.5251 — собственное значение продолженной кривой в точке 0.2 (при обрезке
#: на 0.4 там стояло 1.1934).  У «Стресса» γ = 1: он и есть верхняя граница, ослаблять
#: его нечем и незачем.
CTRL5_STRESS = {"set": 2, "on_freq": 1, "t40": 1.7523, "t60": 1.3253,
                "on_kpod": 1, "k02": 1.5251, "k16": 1.4661,
                "on_wcut": 1, "w95": 0.855}
CTRL5_NULL = {"set": 1, "on_freq": 0, "t40": 1.0, "t60": 1.0,
              "on_kpod": 0, "k02": 1.0, "k16": 1.0,
              "on_wcut": 0, "w95": 1.0}
CTRL5_SCENARIOS = {0: CTRL5_NULL, 1: CTRL5_BASE, 2: CTRL5_STRESS}
CTRL5_ORDER = ("set", "on_freq", "t40", "t60", "on_kpod", "k02", "k16", "on_wcut", "w95")


def _rat(u: float, a, b, *, mul_u: bool) -> float:
    """P(u)/(1+u·Q(u)), optionally with the numerator multiplied by u.  Denominator floored."""
    num = float(np.polyval(np.asarray(a, float)[::-1], u))
    den = 1.0 + float(np.polyval(np.asarray(b, float)[::-1], u)) * u if len(b) else 1.0
    return (u * num if mul_u else num) / max(den, 0.2)


def _arm_gamma(ln_fit: float, target: float) -> float:
    """γ mapping the fitted anchor onto the operator's target, in log θ.

    No leverage when the fit is flat at the anchor: the exponent multiplies zero, so any
    target is unreachable and 1 is the honest answer rather than a division blow-up.
    """
    if abs(ln_fit) < 1e-6 or target <= 0:
        return 1.0
    return float(np.log(target) / ln_fit)


def ln_theta_freq_v52(dev: float, iset: int = 1) -> float:
    a, b = FREQ_RAT[2 if iset >= 2 else 1]
    return _rat(float(np.clip(dev, FREQ_DEV_LO, FREQ_DEV_HI)) / 10.0, a, b, mul_u=False)


def ln_theta_kpod_v52(k: float, iset: int = 1) -> float:
    s = 2 if iset >= 2 else 1
    kk = float(np.clip(k, KPOD_CLAMP_LO[s], KPOD_CLAMP_HI))
    a, b = KPOD_RAT[KPOD_ARM[s]["lo" if kk < KPOD_REF_V52 else "hi"]]
    return _rat((kk - KPOD_REF_V52) / 0.6, a, b, mul_u=True)


def ln_theta_wcut_v52(w: float) -> float:
    a, b = WCUT_RAT
    return _rat((float(np.clip(w, 0.0, 100.0)) - WCUT_REF_V52) / 30.0, a, b, mul_u=True)


def theta_freq_v52(freq: float, ctrl: dict | None = None, *, f_nom: float = 50.0) -> float:
    """Frequency layer with a per-arm operator exponent; floored at 1 (see the note above)."""
    c = CTRL5_BASE if ctrl is None else ctrl
    if freq <= 0:
        return 1.0
    dev = float(np.clip(float(freq) - f_nom, FREQ_DEV_LO, FREQ_DEV_HI))
    lt = ln_theta_freq_v52(dev, c["set"])
    if dev < 0:
        g = _arm_gamma(ln_theta_freq_v52(FREQ_ANCHORS[0], c["set"]), c["t40"])
    elif dev > 0:
        g = _arm_gamma(ln_theta_freq_v52(FREQ_ANCHORS[1], c["set"]), c["t60"])
    else:
        g = 1.0
    return max(float(np.exp(g * lt)), 1.0)


def theta_kpod_v52(kpod: float, ctrl: dict | None = None) -> float:
    c = CTRL5_BASE if ctrl is None else ctrl
    if kpod < 0:
        return 1.0
    s = 2 if c["set"] >= 2 else 1
    kk = float(np.clip(kpod, KPOD_CLAMP_LO[s], KPOD_CLAMP_HI))
    lt = ln_theta_kpod_v52(kk, c["set"])
    if kk < KPOD_REF_V52:
        g = _arm_gamma(ln_theta_kpod_v52(KPOD_ANCHORS[0], c["set"]), c["k02"])
    elif kk > KPOD_REF_V52:
        g = _arm_gamma(ln_theta_kpod_v52(KPOD_ANCHORS[1], c["set"]), c["k16"])
    else:
        g = 1.0
    return max(float(np.exp(g * lt)), 1e-6)


def theta_wcut_v52(wcut: float, ctrl: dict | None = None) -> float:
    """Accepts a fraction or per cent; ≤ 1 reads as a fraction, the same rule the fit uses."""
    c = CTRL5_BASE if ctrl is None else ctrl
    if wcut is None or wcut < 0:
        return 1.0
    w = float(wcut) * 100.0 if float(wcut) <= 1.0 else float(wcut)
    g = _arm_gamma(ln_theta_wcut_v52(WCUT_ANCHOR), c["w95"])
    return max(float(np.exp(g * ln_theta_wcut_v52(w))), 1e-6)


def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def contractor_mult(name: str, brt: float, slb: float, oth: float) -> float:
    """Same matcher as the VBA: Cyrillic and Latin spellings of the three groups."""
    nm = _norm_name(name)
    if not nm:
        return brt
    if any(t in nm for t in ("бор", "borets", "borec", "brt")):
        return brt
    if any(t in nm for t in ("шлю", "слай", "слб", "slb", "schlumberger")):
        return slb
    return oth


def rmst(eta: float, beta: float, horizon: float = RMST_HORIZON) -> float:
    """η·Γ(1+1/β)·P(1/β,(H/η)^β) — the closed form the VBA evaluates with its own gamma."""
    from scipy.special import gamma, gammainc
    if eta <= 0 or beta <= 0:
        return 0.0
    return float(eta * gamma(1 + 1 / beta) * gammainc(1 / beta, (horizon / eta) ** beta))


def eta_from_rmst(target: float, beta: float, horizon: float = RMST_HORIZON) -> float:
    """Invert RMST(η) — monotone in η, so bisection is exact (the VBA does the same)."""
    lo, hi = 1.0, 5e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rmst(mid, beta, horizon) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class Layers:
    """The deployed hazard model for one stratum key."""
    key: str = "Vt_nonsour"
    tune: dict = dc_field(default_factory=lambda: dict(TUNE))
    qnom_block: dict = dc_field(default_factory=lambda: dict(QNOM))
    ctrl: dict = dc_field(default_factory=lambda: dict(CTRL_DEFAULT))
    #: Nameplate frequency the fitted freq block's deviation axis is measured against.
    f_nom: float = 50.0
    #: v5.2 fitted layers as TABULATED blocks — research use, and the way the layers were
    #: first compared.  ``None``/absent key ⇒ fall through to ``ctrl5`` below.
    wcut_block: dict | None = None
    kpod_block: dict | None = None
    freq_block: dict | None = None
    #: The v5.2 shipped form: rational curves in code, levels in the CtrlBlock.  Pass
    #: :data:`CTRL5_BASE` / :data:`CTRL5_STRESS` to model ПикПолка **v9** and NPV **V04**.
    #:
    #: Default is None — the V4 operator closed forms — on purpose: this module's job is to
    #: be the verified twin of the LIVE workbook, and until v9 replaces v8 the live workbook
    #: is still V4.  Flipping the default before then would silently break the parity test
    #: that makes the twin worth anything.  Flip it when v9 goes live.
    ctrl5: dict | None = None

    def __post_init__(self):
        if self.key not in self.tune:
            raise KeyError(f"no TuneBlock row for {self.key!r}; have {sorted(self.tune)}")
        self._beta, self._rmst_ref = self.tune[self.key][0], self.tune[self.key][1]
        self._eta0 = eta_from_rmst(self._rmst_ref, self._beta)

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def eta0(self) -> float:
        return self._eta0

    def theta_kpod_layer(self, kpod: float) -> float:
        """Tabulated block if one is supplied, else the shipped v5.2 rational, else V4.

        ⚠ The v5.2 curve and the V4 bathtub disagree in SHAPE, not just level: the bathtub
        penalises both ends (θ=0.58 at Kпод 0.2) while every leakage-proof fit gives a
        shallow rise.  Switching between them is a modelling decision, never a tidy-up.
        """
        blk = (self.kpod_block or KPOD_FIT).get(self.key)
        if blk:
            return theta_tab(kpod, KPOD_KNOTS, blk)
        if self.ctrl5 is None:
            return theta_kpod(kpod, self.ctrl)
        if self.ctrl5["on_kpod"] < 0.5:
            return 1.0
        return theta_kpod_v52(kpod, self.ctrl5)

    def theta_freq_layer(self, freq: float) -> float:
        """Same precedence as Kпод.  Indexed on the DEVIATION from nominal, not absolute Hz —
        a 60 Hz run on a 60 Hz nameplate is at deviation 0, and pricing it as +10 is wrong.
        """
        blk = (self.freq_block or FREQ_FIT).get(self.key)
        if blk:
            return theta_tab(float(freq) - self.f_nom, FREQ_KNOTS_FIT, blk)
        if self.ctrl5 is None:
            return theta_freq(freq, self.ctrl)
        if self.ctrl5["on_freq"] < 0.5:
            return 1.0
        return theta_freq_v52(freq, self.ctrl5, f_nom=self.f_nom)

    def theta_wcut_layer(self, wcut: float | None) -> float:
        """θ_wcut.  Ships OFF, so this returns 1.0 unless the switch is turned on."""
        if wcut is None or not np.isfinite(wcut):
            return 1.0
        blk = (self.wcut_block or WCUT).get(self.key)
        if blk:
            return theta_tab(float(wcut), WCUT_KNOTS, blk)
        if self.ctrl5 is None or self.ctrl5["on_wcut"] < 0.5:
            return 1.0
        return theta_wcut_v52(wcut, self.ctrl5)

    def theta(self, *, contractor: str, qnom: float, freq: float, kpod: float,
              wcut: float | None = None) -> float:
        _, _, brt, slb, oth = self.tune[self.key]
        return max(contractor_mult(contractor, brt, slb, oth)
                   * theta_qnom(self.key, qnom, self.qnom_block)
                   * self.theta_freq_layer(freq)
                   * self.theta_kpod_layer(kpod)
                   * self.theta_wcut_layer(wcut), 1e-6)

    def hazard(self, age_days: float, **kw) -> float:
        """Weibull hazard at ``age_days`` since the last install/restart, times the θ layers.

        This is what :func:`simulate_branch` integrates in ``hazard_mode="integrate"``.  It is
        the quantity the consume-and-renew rule only approximates: that rule accumulates
        ``1/RMST`` per day, which is a life, not a hazard, and is age-blind.
        """
        b, e0 = self._beta, self._eta0
        a = max(float(age_days), 1e-6)
        return (b / e0) * (a / e0) ** (b - 1.0) * self.theta(**kw)

    def life_days(self, **kw) -> float:
        """Operating life = RMST(0,730) at this operating point."""
        th = self.theta(**kw)
        return rmst(self._eta0 * th ** (-1.0 / self._beta), self._beta)

    def life_at_ref(self) -> float:
        """Life at the reference operating point — brt, Qnom 250, 50 Hz, Kpod in the plateau.

        NOT the bare baseline: θ_freq(50) = 0.9, so "all θ = 1" is a different number.
        """
        k = 0.5 * (self.ctrl["kpl_lo"] + self.ctrl["kpl_hi"])
        return self.life_days(contractor="brt", qnom=250.0, freq=50.0, kpod=k)


# ---------------------------------------------------------------------------
# Sheet configuration, read from the workbook
# ---------------------------------------------------------------------------
@dataclass
class Branch:
    """One scenario column-block of the sheet.

    The two workbooks differ here and it matters: in ``NPV_УВЧ`` both branches decline, while
    in ``ПикПолка`` the «Полка» block is deliberately **flat** (``N16 = E$4*T$4``, no decline
    term) — it is a plateau case, not a second decline. Its uptime window differs too («смена,
    дней ПИК» 19 vs «Полка» 12), although both are handed the same value as the schedule's
    downtime.
    """
    name: str
    ql0: float
    freq: float
    nominal0: float
    contractor: str
    decline: bool = True
    downtime_sheet: int | None = None      # uptime/cost window; None → cfg.downtime_days
    downtime_model: int | None = None      # what the schedule is given


@dataclass
class SheetConfig:
    """Everything the day grid needs.  Reference tables come from the workbook."""
    field: str = "Vt"
    sour: int = 1
    days: int = 4000
    oil_density: float = 0.86          # Плотн нефти
    wcut0: float = 0.5                 # Обв. д.е.
    decline_6m: float = 1.0            # Падение первые 6 мес
    decline_after: float = 1.0         # Падение после 6 мес
    bend: float = 0.6                  # параметр прогиба 6 мес
    phase1_days: int = 180             # длина первой фазы падения
    phase2_days: int = 915             # знаменатель линейной второй фазы
    discount: float = 0.15             # Ставка Диск
    niz: float = 0.0                   # НИЗ (запасы)
    tiz: float = 0.0                   # ТИЗ
    downtime_days: int = 12            # uptime/cost window on the sheet (column J)
    #: Downtime the **schedule** is given — ``ПикПолка`` hands the UDF the base «смена, дней»
    #: (P4) while column J uses the ускоренная O4, so the nominal-reset edge and the revenue
    #: outage can legitimately differ.  Defaults to :attr:`downtime_days`.
    downtime_model: int | None = None
    netback: float = 0.0               # руб/тн
    repair_cost_per_day: float = 0.0   # КРС+ЭПУ+потеря добычи, руб/сут
    ladder: tuple = ()                 # nominal ladder
    wcut_curve: tuple = ()             # (отбор от НИЗ, обводнённость)
    rent: dict = dc_field(default_factory=dict)   # "«номинал»«подрядчик»" -> руб/сут
    ko: dict = dc_field(default_factory=dict)     # "«номинал»«подрядчик»" -> руб/сут
    branches: tuple = ()
    layers_key: str | None = None      # override the stratum key

    @property
    def key(self) -> str:
        if self.layers_key:
            return self.layers_key
        return "Vt_sour" if (self.field == "Vt" and self.sour) else (
            f"{self.field}_sour" if self.sour and f"{self.field}_sour" in TUNE
            else (f"{self.field}_nonsour" if f"{self.field}_nonsour" in TUNE else self.field))


def _cell(ws, addr):
    v = ws[addr].value
    return v


def load_config(path: str | Path, *, layout: str = "npv") -> SheetConfig:
    """Read a calculator workbook into a :class:`SheetConfig` (cached values, no Excel needed).

    ``layout`` selects the cell map: ``"npv"`` for ``Копия Калькулятор_NPV_УВЧ_V02.xlsm``
    (ПрогнозРемонтов row 4, grid from row 11) and ``"pikpolka"`` for
    ``2026_07_Калькулятор_ПикПолка_v7.xlsm`` (grid from row 15).
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
    p, hv, kr, ec = wb["ПрогнозРемонтов"], wb["ХВ"], wb["КРС_ЭПУ"], wb["Экономика"]

    ladder = tuple(float(kr.cell(r, 2).value) for r in range(41, 58)
                   if isinstance(kr.cell(r, 2).value, (int, float)))
    curve = tuple((float(hv.cell(r, 16).value), float(hv.cell(r, 17).value))
                  for r in range(21, 92)
                  if isinstance(hv.cell(r, 16).value, (int, float))
                  and isinstance(hv.cell(r, 17).value, (int, float)))
    # Column A is the concatenated «номинал & подрядчик» key the sheet XLOOKUPs on
    # (`E13&F13`), and Excel's XLOOKUP is case-insensitive — the sheet says
    # «Новые_Технологии» while the table says «Новые_технологии».  Fold case here or every
    # cost lookup silently returns "нд".
    rent, ko = {}, {}
    for r in range(35, 119):
        key = kr.cell(r, 1).value
        if key:
            rent[str(key).casefold()] = float(kr.cell(r, 6).value or 0.0)
            ko[str(key).casefold()] = float(kr.cell(r, 8).value or 0.0)

    if layout == "npv":
        field = str(_cell(p, "H4"))
        cfg = SheetConfig(
            field=field, sour=int(_cell(p, "I4") or 0), days=4000,
            oil_density=float(_cell(p, "B4")), wcut0=float(_cell(p, "C4")),
            decline_6m=float(_cell(p, "D4")), decline_after=float(_cell(p, "E4")),
            bend=float(_cell(p, "AG4")), discount=float(_cell(p, "AE4")),
            niz=float(_cell(p, "AC4") or 0.0), tiz=float(_cell(p, "AF4") or 0.0),
            downtime_days=int(_cell(p, "U4") or 12),
            ladder=ladder, wcut_curve=curve, rent=rent, ko=ko)
        ql, ql_alt = float(_cell(p, "L4")), float(_cell(p, "M4"))
        f0, f1 = float(_cell(p, "J4")), float(_cell(p, "K4"))
        nom0 = float(_cell(p, "F4"))
        cg = str(_cell(p, "G4"))
        cfg = replace(cfg, branches=(Branch("ТекЧастота", ql, f0, nom0, cg),
                                     Branch("Разгон", ql_alt, f1, nom0, cg)))
    elif layout == "pikpolka":
        field = str(_cell(p, "N4"))
        cfg = SheetConfig(
            field=field, sour=int(_cell(p, "Q4") or 0), days=1094,
            oil_density=float(_cell(p, "C4")), wcut0=float(_cell(p, "D4")),
            decline_6m=float(_cell(p, "J4")), decline_after=float(_cell(p, "K4")),
            bend=float(_cell(p, "L4")), discount=float(_cell(p, "I4")),
            niz=float(_cell(p, "G4") or 0.0), tiz=float(_cell(p, "H4") or 0.0),
            downtime_days=int(_cell(p, "O4") or 12),      # «смена, дней ПИК» — column J
            downtime_model=int(_cell(p, "P4") or 12),     # what both spills hand the UDF
            ladder=ladder, wcut_curve=curve, rent=rent, ko=ko)
        ql, ql_alt = float(_cell(p, "E4")), float(_cell(p, "U4"))
        nom0, nom1 = float(_cell(p, "M4")), float(_cell(p, "W4"))
        cg = str(_cell(p, "S4"))
        f0 = float(_cell(p, "G15") or 50.0)
        dt_pik, dt_polka = int(_cell(p, "O4") or 12), int(_cell(p, "P4") or 12)
        cfg = replace(cfg, branches=(
            Branch("Пик", ql, f0, nom0, cg, decline=True,
                   downtime_sheet=dt_pik, downtime_model=dt_polka),
            Branch("Полка", ql_alt, f0, nom1, cg, decline=False,
                   downtime_sheet=dt_polka, downtime_model=dt_polka)))
    else:
        raise ValueError(f"unknown layout {layout!r}")

    for r in range(10, 21):
        if str(ec.cell(r, 3).value) == cfg.field:
            cfg = replace(cfg, netback=float(ec.cell(r, 5).value or 0.0))
    for r in range(10, 18):
        if str(kr.cell(r, 7).value) == cfg.field:
            cfg = replace(cfg, downtime_days=int(kr.cell(r, 8).value or cfg.downtime_days),
                          repair_cost_per_day=float(kr.cell(r, 9).value or 0.0))
    if not cfg.repair_cost_per_day:                     # the sheet reads one fixed cell
        v = kr.cell(15, 9).value
        cfg = replace(cfg, repair_cost_per_day=float(v or 0.0))
    wb.close()
    return cfg


# ---------------------------------------------------------------------------
# The day grid
# ---------------------------------------------------------------------------
def decline_ql(cfg: SheetConfig, ql0: float, day: int) -> float:
    """The sheet's **two-phase** decline, both workbooks::

        u = day − 1
        u ≤ 179 :  ql0 · (1 + (падение_6мес − 1) · (u/180)^прогиб)      curved
        u ≥ 180 :  ql₁₈₀ · (1 + (падение_после − 1) · (u−180)/915)      linear

    ⚠ Easy to miss: the formula **changes row** at day 181 (``C195 = C$194*…`` in ПикПолка,
    ``C191 = C$190*…`` in NPV).  Extrapolating phase 1 to the end of the grid instead drives
    Ql negative within a year on a steep case — which is exactly the sort of silent divergence
    a port has to be checked against the sheet to catch.
    """
    u = max(day - 1, 0)
    p1, p2 = cfg.phase1_days, cfg.phase2_days
    if u <= p1 - 1:
        return ql0 * (1 + (cfg.decline_6m - 1) * ((u / p1) ** cfg.bend))
    anchor = ql0 * (1 + (cfg.decline_6m - 1) * (((p1 - 1) / p1) ** cfg.bend))
    return anchor * (1 + (cfg.decline_after - 1) * ((u - p1) / p2))


def lookup_nominal(ql: float, ladder) -> float:
    """Smallest catalogue nominal ≥ Ql, else the **largest** one.

    The workbook's XLOOKUP default (99999/999999) is the bug this port refuses to reproduce:
    it drives Kpod to zero and costs ~40 % of the modelled life for a scenario that simply
    runs above the top of the ladder.
    """
    cands = [x for x in ladder if x >= ql]
    return float(min(cands)) if cands else float(max(ladder))


def _wcut(curve, x: float) -> float:
    """XLOOKUP(…, if_not_found=0, match_mode=1): the **smallest** «отбор от НИЗ» ≥ x.

    Next-larger, not first-encountered — and the distinction is load-bearing here because the
    ХВ curve is not monotone at its tail (…0.99901, 1.00085, 1.00227, 1.0).  At a recovery
    share of 0.99903 next-larger picks the final 1.0 → водность 1.0 → oil 0, and because oil
    stops the cumulative freezes: the well is dead from that day on.  First-encountered would
    return 0.99 instead, leak 1 % of the rate forever, and eventually run the share past the
    end of the curve where the lookup falls back to 0 % water cut and the well comes back to
    life at FULL rate — 1729 phantom days on the NPV sheet's 4000-day grid.
    """
    best = None
    for p, q in curve:
        if p >= x and (best is None or p < best[0]):
            best = (p, q)
    return best[1] if best else 0.0


def simulate_branch(cfg: SheetConfig, br: Branch, layers: Layers,
                    *, hazard_mode: str = "consume") -> pd.DataFrame:
    """One scenario over the day grid — the sheet's columns, in order.

    ``hazard_mode``
        ``"consume"`` (default, unchanged) reproduces the sheet: accumulate ``1/RMST`` per day
        and fire when it reaches 1.  That quantity is a *life*, not a hazard, and it is
        age-blind — a pump on day 900 consumes at the same rate as on day 9.

        ``"integrate"`` accumulates the real Weibull hazard ``H += h(age)`` and fires when
        ``H ≥ 1`` (i.e. S ≤ e⁻¹ = 0.368), with ``age`` reset at each restart.  This is the
        form that makes TTF respond to the **trajectory**: because θ_Kпод and θ_wcut are
        evaluated at each day's Kпод(t) and обв(t), changing the decline or the ХВ curve now
        changes the failure date instead of leaving it byte-identical.

    ⚠ The two rules fire at different thresholds and will not agree numerically.  The default
    stays ``"consume"`` so every previously validated number is untouched.
    """
    n = int(cfg.days)
    dd_sheet = max(int(br.downtime_sheet or cfg.downtime_days), 1)
    dd = max(int(br.downtime_model or cfg.downtime_model or cfg.downtime_days), 1)
    ql = np.empty(n)
    oil = np.empty(n)
    nominal = np.empty(n)
    nominal_sheet = np.empty(n)
    kpod = np.empty(n)
    life = np.empty(n)
    flag = np.ones(n, dtype=int)
    up = np.ones(n, dtype=int)

    wcut_d = np.empty(n)
    age_d = np.empty(n)
    haz = np.zeros(n)
    cumhaz = np.zeros(n)
    consumed, nom = 0.0, float(br.nominal0)
    nom_sheet = float(br.nominal0)
    cum_oil = 0.0
    age, H = 0.0, 0.0
    for i in range(n):
        day = i + 1
        ql[i] = decline_ql(cfg, br.ql0, day) if br.decline else br.ql0
        if i == 0:
            # the sheet writes the first row directly from the starting water cut
            oil[i] = ql[i] * (1 - cfg.wcut0) * cfg.oil_density
        else:
            share = (cum_oil + (cfg.niz - cfg.tiz)) / cfg.niz if cfg.niz else 0.0
            oil[i] = ql[i] * cfg.oil_density * (1 - _wcut(cfg.wcut_curve, share))
        cum_oil += oil[i]

        # restart edge, VBA form: J(i−1) = 1 and J(i−2) = 0, one-based rows.
        # The UDF rebuilds J from its OWN DownDays argument, which in ПикПолка is not the one
        # column J on the sheet uses — so the model's nominal and the displayed «Номинал»
        # (which drives the cost lookup) genuinely step on different days.  Both are tracked.
        if i >= 1 and _up_at(flag, i - 1, dd) == 1 and _up_at(flag, i - 2, dd) == 0:
            nom = lookup_nominal(ql[i], cfg.ladder)
        if i >= 1 and _up_at(flag, i - 1, dd_sheet) == 1 and _up_at(flag, i - 2, dd_sheet) == 0:
            nom_sheet = lookup_nominal(ql[i], cfg.ladder)
        nominal[i] = nom
        nominal_sheet[i] = nom_sheet
        kpod[i] = ql[i] / nom if nom > 0 else np.nan
        # water cut implied by the ХВ curve on this day: oil = ql·ρ·(1−обв)
        den = ql[i] * cfg.oil_density
        wcut_d[i] = 1.0 - oil[i] / den if den > 0 else np.nan
        age += 1.0
        age_d[i] = age
        kw = dict(contractor=br.contractor, qnom=nom, freq=br.freq, kpod=kpod[i],
                  wcut=100.0 * wcut_d[i] if np.isfinite(wcut_d[i]) else None)
        life[i] = layers.life_days(**kw)
        if hazard_mode == "integrate":
            haz[i] = layers.hazard(age, **kw)
            H += haz[i]
            cumhaz[i] = H
            if H >= 1.0:
                flag[i] = 0
                H, age = 0.0, 0.0
        else:
            consumed += 1.0 / max(life[i], 1e-6)
            cumhaz[i] = consumed
            if consumed >= 1.0:
                flag[i] = 0
                consumed -= 1.0
                age = 0.0
        up[i] = _up_at(flag, i, dd_sheet)

    keys = [f"{nominal_sheet[i]:g}{br.contractor}".casefold() for i in range(n)]
    rent = np.array([cfg.rent.get(k, 0.0) for k in keys])
    ko = np.array([cfg.ko.get(k, 0.0) for k in keys])
    cost = -np.where(up == 0, cfg.repair_cost_per_day, 0.0) - rent - ko
    revenue = oil * cfg.netback * up
    day = np.arange(1, n + 1)
    disc = revenue / (1 + cfg.discount) ** (day / 365.0)
    return pd.DataFrame({"day": day, "ql": ql, "oil": oil, "nominal": nominal,
                         "nominal_sheet": nominal_sheet, "kpod": kpod,
                         # what the sheet's «Отношение к номиналу» shows — the model does not
                         # use it, and on ПикПолка the two differ for the rows between the
                         # two restart edges
                         "kpod_sheet": ql / np.where(nominal_sheet > 0, nominal_sheet, np.nan),
                         "life": life, "fail": flag, "up": up,
                         "wcut": wcut_d, "age": age_d, "hazard": haz, "cumhaz": cumhaz,
                         "cost": cost, "revenue": revenue, "npv_day": disc})


def _up_at(flag: np.ndarray, i: int, dd: int) -> int:
    """Uptime at day index ``i``: 0 if any failure in the trailing ``dd`` days."""
    if i < 0:
        return 1
    lo = max(i - dd + 1, 0)
    return 0 if np.any(flag[lo:i + 1] == 0) else 1


def simulate(cfg: SheetConfig, *, layers: Layers | None = None,
             tune: dict | None = None, qnom_block: dict | None = None,
             ctrl: dict | None = None, hazard_mode: str = "consume") -> dict:
    """Run every branch and return the summary the sheets report, plus the grids."""
    lay = layers or Layers(key=cfg.key,
                           tune=dict(tune or TUNE),
                           qnom_block=dict(qnom_block or QNOM),
                           ctrl=dict(ctrl or CTRL_DEFAULT))
    grids, rows = {}, []
    for br in cfg.branches:
        g = simulate_branch(cfg, br, lay, hazard_mode=hazard_mode)
        grids[br.name] = g
        first = g.loc[g["fail"] == 0, "day"]
        rows.append({"branch": br.name, "ql0": br.ql0, "freq": br.freq,
                     "nominal0": br.nominal0, "contractor": br.contractor,
                     "nno_days": int(first.iloc[0]) if len(first) else 9999,
                     "failures": int((g["fail"] == 0).sum()),
                     "npv": float(g["npv_day"].sum()),
                     "cost": float(g["cost"].sum()),
                     "oil": float((g["oil"] * g["up"]).sum())})
    s = pd.DataFrame(rows)
    out = {"summary": s, "grids": grids, "layers": lay,
           "life_at_ref": lay.life_at_ref()}
    if len(s) == 2:
        out["d_nno"] = int(s["nno_days"].iloc[1] - s["nno_days"].iloc[0])
        out["d_npv"] = float(s["npv"].iloc[1] - s["npv"].iloc[0])
    return out


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
#: What a sweep may vary.  ``branch:<field>`` targets the *alternative* branch only, which is
#: what a УВЧ study wants: hold the base case and move the intervention.
SWEEPABLE = ("field", "sour", "downtime_days", "discount", "decline_6m", "bend",
             "wcut0", "netback", "layers_key",
             "branch0:ql0", "branch0:freq", "branch0:nominal0", "branch0:contractor",
             "branch1:ql0", "branch1:freq", "branch1:nominal0", "branch1:contractor",
             "ctrl:t50", "ctrl:t60", "ctrl:t70", "ctrl:k02", "ctrl:k12",
             "ctrl:kpl_lo", "ctrl:kpl_hi")


def _apply(cfg: SheetConfig, ctrl: dict, key: str, value):
    if key.startswith("ctrl:"):
        ctrl[key.split(":", 1)[1]] = value
        return cfg
    if key.startswith("branch"):
        idx = int(key[6])
        attr = key.split(":", 1)[1]
        brs = list(cfg.branches)
        brs[idx] = replace(brs[idx], **{attr: value})
        return replace(cfg, branches=tuple(brs))
    return replace(cfg, **{key: value})


def sweep(cfg: SheetConfig, *, tune: dict | None = None, qnom_block: dict | None = None,
          ctrl: dict | None = None, keep_grids: bool = False, **axes) -> pd.DataFrame:
    """Cartesian sweep over any of :data:`SWEEPABLE`.

    Returns one row per (point × branch) with ННО, failure count, NPV and oil, plus the Δ's
    between the two branches so a УВЧ study reads straight off it::

        sweep(cfg, **{"branch1:freq": [50, 55, 60, 65], "branch0:ql0": [400, 800]})
    """
    bad = [k for k in axes if k not in SWEEPABLE]
    if bad:
        raise KeyError(f"not sweepable: {bad}; allowed {SWEEPABLE}")
    names = list(axes)
    rows, grids = [], {}
    for point in product(*(axes[n] for n in names)):
        c, ct = cfg, dict(ctrl or CTRL_DEFAULT)
        for k, v in zip(names, point):
            c = _apply(c, ct, k, v)
        res = simulate(c, tune=tune, qnom_block=qnom_block, ctrl=ct)
        tag = dict(zip(names, point))
        for _, r in res["summary"].iterrows():
            rows.append({**tag, **r.to_dict(),
                         "d_nno": res.get("d_nno"), "d_npv": res.get("d_npv"),
                         "life_at_ref": res["life_at_ref"]})
        if keep_grids:
            grids[point] = res["grids"]
    out = pd.DataFrame(rows)
    if keep_grids:
        out.attrs["grids"] = grids
    return out


# ---------------------------------------------------------------------------
# Port validation
# ---------------------------------------------------------------------------
def validate_against_workbook(path: str | Path, *, layout: str = "npv", tol_days: int = 1,
                              qnom_block: dict | None = None, npv_cells=("P4", "Q4"),
                              **kw) -> pd.DataFrame:
    """Compare the port against the workbook's own cached results.

    Defaults to :data:`QNOM_PRECLAMP`, i.e. the parameters a not-yet-rewired workbook is
    running, so a green row means "the port reproduces Excel" rather than "the port and Excel
    happen to disagree by the size of a change we made on purpose".  Pass ``qnom_block=QNOM``
    to measure what the sour clamp costs.

    ⚠ :func:`lookup_nominal` is a second deliberate divergence: a case whose Ql runs above the
    ladder will not match, and should not — that is the bug this port refuses to reproduce.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    p = wb["ПрогнозРемонтов"]
    if layout == "npv":
        first_row, flags, nno_cells = 11, (9, 20), ("N4", "O4")
    else:
        first_row, flags, nno_cells = 15, (9, 20), ("R4", "Z4")
    cfg = load_config(path, layout=layout)
    res = simulate(cfg, qnom_block=dict(qnom_block or QNOM_PRECLAMP), **kw)
    rows = []
    for i, (col, cell) in enumerate(zip(flags, nno_cells)):
        cached_fail = sum(1 for r in range(first_row, first_row + cfg.days)
                          if p.cell(r, col).value == 0)
        cached_nno = p[cell].value
        npv_sheet = p[npv_cells[i]].value if i < len(npv_cells) else None
        s = res["summary"].iloc[i]
        rows.append({"branch": s["branch"],
                     "nno_sheet": cached_nno, "nno_python": s["nno_days"],
                     "failures_sheet": cached_fail, "failures_python": s["failures"],
                     "npv_sheet": npv_sheet, "npv_python": round(float(s["npv"]), 2),
                     "npv_rel_err": (abs(float(npv_sheet) - float(s["npv"]))
                                     / max(abs(float(npv_sheet)), 1.0)
                                     if isinstance(npv_sheet, (int, float)) else np.nan),
                     "ok": (abs(float(cached_nno) - s["nno_days"]) <= tol_days
                            and cached_fail == int(s["failures"]))})
    wb.close()
    return pd.DataFrame(rows)
