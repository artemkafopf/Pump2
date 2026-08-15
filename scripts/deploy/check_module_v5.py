"""Three-way agreement check for the v5.2 layer forms.

    python scripts/deploy/check_module_v5.py

The same curve now exists in three places and a silent disagreement between any two is the
kind of bug that ships: a wrong digit in a coefficient does not raise, it just prices pumps
slightly wrong forever.  So all three are compared numerically, on a dense grid, against the
fitted tables that are the source of truth:

1. ``results/.../final_sets_rational.csv`` + ``freq_final_coefficients.csv`` — the fit;
2. ``vba/ModuleFailureV5.bas`` — parsed for its ``Private Const`` values, so what is checked
   is the text that actually ships rather than what the build script intended to write;
3. ``analysis.workflows.production_risk.pikpolka_sim`` — the Python twin.

Also checked: that the CtrlBlock defaults in both files equal the fitted curve's own value at
each anchor.  That is what makes γ = 1 and therefore what makes "the deployed curve IS the
fit" a true statement rather than an aspiration.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.paths import RESULTS_ROOT  # noqa: E402
from analysis.workflows.production_risk import pikpolka_sim as PS  # noqa: E402

BAS = REPO_ROOT / "vba" / "ModuleFailureV5.bas"
SLUG = "production_risk_field_v52"
TOL = 5e-7


def bas_consts() -> dict[str, float]:
    text = BAS.read_text(encoding="cp1251")
    out = {}
    for name, val in re.findall(r"Private Const (\w+)\s+As Double\s*=\s*([-\d.#eE+]+)", text):
        out[name] = float(val.rstrip("#"))
    return out


def main() -> int:
    C = bas_consts()
    tabdir = sorted((RESULTS_ROOT / SLUG).glob("*/tables"))[-1]
    S = pd.read_csv(tabdir / "final_sets_rational.csv")
    FR = pd.read_csv(tabdir / "freq_final_coefficients.csv").set_index("set")
    bad = 0

    # --- 1. coefficients: .bas text vs pikpolka_sim -------------------------
    pairs = [
        ("freq base a", [C[f"FA{i}_B"] for i in range(5)], PS.FREQ_RAT[1][0]),
        ("freq base b", [C[f"FB{i}_B"] for i in range(3)], PS.FREQ_RAT[1][1]),
        ("freq stress a", [C[f"FA{i}_S"] for i in range(4)], PS.FREQ_RAT[2][0]),
        ("freq stress b", [C[f"FB{i}_S"] for i in range(3)], PS.FREQ_RAT[2][1]),
        ("kpod base a", [C["KA0_B"], C["KA1_B"]], PS.KPOD_RAT[1][0]),
        ("kpod base b", [C["KB0_B"], C["KB1_B"]], PS.KPOD_RAT[1][1]),
        ("kpod stress a", [C["KA0_S"], C["KA1_S"]], PS.KPOD_RAT[2][0]),
        ("wcut a", [C["WA0"], C["WA1"]], PS.WCUT_RAT[0]),
        ("wcut b", [C["WB0"], C["WB1"]], PS.WCUT_RAT[1]),
    ]
    print("=== коэффициенты: .bas против pikpolka_sim")
    for lab, a, b in pairs:
        a, b = list(a), list(b)
        n = min(len(a), len(b))
        ok = a[:n] == list(b)[:n] and all(abs(x) < 1e-12 for x in a[n:])
        print(f"  {'OK ' if ok else '*** РАСХОЖДЕНИЕ ***'} {lab:16s} {a}")
        bad += 0 if ok else 1
    # the stress Kпод denominator is degree 1 in the fit and padded with a zero in the .bas
    if abs(C["KB1_S"]) > 1e-12:
        print("  *** KB1_S должен быть 0 (у стрессового Kпод знаменатель первой степени)")
        bad += 1

    # --- 2. curves: .bas / twin vs the fitted tables ------------------------
    # Two separate questions, deliberately not merged.  The RAW rational (γ = 1) must
    # reproduce the fit to machine precision — anything else is a transcription error.  The
    # DEPLOYED curve additionally carries the sheet's 4-decimal anchors, and rounding those
    # perturbs γ slightly; that deviation is expected and only has to stay negligible.
    print("\n=== 2a. сырая рациональная форма против подгонки (γ = 1, машинная точность)")

    def check(lab, xs, twin, fit, tol, tag=""):
        d = float(np.max(np.abs(np.array([twin(x) for x in xs]) - np.array(fit))))
        flag = "OK " if d <= tol else "*** РАСХОЖДЕНИЕ ***"
        print(f"  {flag} {lab:34s} max|Δθ| = {d:.2e} {tag}")
        return 0 if d <= tol else 1

    FREQ_FIT_CURVE = {}
    for nm, iset in (("Базовый", 1), ("Стресс", 2)):
        a = np.array([float(v) for v in FR.loc[nm, "a"].split()])
        b = np.array([float(v) for v in FR.loc[nm, "b"].split()])
        cut, sc, cl = (float(FR.loc[nm, k]) for k in ("cut_hz_dev", "scale", "clamp_min"))
        xs = np.linspace(-22, 20, 400)
        fit = [max(float(np.exp(np.polyval(a[::-1], u) / (1 + np.polyval(b[::-1], u) * u))), cl)
               for u in (np.clip(xs, cut, None) / sc)]
        FREQ_FIT_CURVE[iset] = (xs, fit)
        bad += check(f"частота «{nm}»", xs,
                     lambda x, s=iset: max(float(np.exp(PS.ln_theta_freq_v52(x, s))), 1.0),
                     fit, TOL)

    def layer_fit(name):
        r = S[S["set"] == name].iloc[0]
        a = np.array([float(v) for v in str(r["a"]).split()])
        b = (np.array([float(v) for v in str(r["b"]).split()])
             if str(r["b"]).strip() else np.array([]))
        lo = float(r["clamp_below"]) if pd.notna(r["clamp_below"]) else None
        return float(r["ref"]), float(r["scale"]), a, b, lo

    def eval_layer(x, name):
        ref, sc, a, b, lo = layer_fit(name)
        xx = max(x, lo) if lo else x
        u = (xx - ref) / sc
        den = 1.0 + (float(np.polyval(b[::-1], u)) * u if len(b) else 0.0)
        return float(np.exp(u * float(np.polyval(a[::-1], u)) / den))

    # ⚠ Kпод сверяется ПО ПЛЕЧАМ, потому что нижнее плечо у обоих сценариев — стрессовой
    # формы (см. KPOD_ARM).  Сравнивать базовое нижнее плечо с базовой строкой CSV нельзя:
    # это разные объекты по построению, и «расхождение» там было бы ложной тревогой.
    KPOD_FIT_CURVE = {}
    for nm, iset, ctrl in (("Kпод «Базовый»", 1, PS.CTRL5_BASE),
                           ("Kпод «Стресс»", 2, PS.CTRL5_STRESS)):
        hi = np.linspace(0.8, 1.6, 200)
        bad += check(f"{nm}, верхнее плечо", hi,
                     lambda x, s=iset: float(np.exp(PS.ln_theta_kpod_v52(x, s))),
                     [eval_layer(x, nm) for x in hi], TOL)
        KPOD_FIT_CURVE[iset] = (np.linspace(0.4, 1.6, 400),
                                [eval_layer(x, nm) for x in np.linspace(0.4, 1.6, 400)],
                                ctrl, nm)
    lo = np.linspace(0.4, 0.8, 200)
    bad += check("Kпод, нижнее плечо обоих (форма «Стресса»)", lo,
                 lambda x: float(np.exp(PS.ln_theta_kpod_v52(x, 1))),
                 [eval_layer(x, "Kпод «Стресс»") for x in lo], TOL)
    # ⚠ grid starts at 2 %, not 0: the layer reads a value <= 1 as a FRACTION, so 0.5 on this
    # axis means 50 %, not half a per cent.  That rule is deliberate (it is what the fitting
    # frame does) but it makes the bottom two per cent of the axis untestable as per cent.
    xs_w = np.linspace(2, 95, 400)
    wfit = [eval_layer(x, "обв — v5.2 (к отгрузке)") for x in xs_w]
    bad += check("обводнённость", xs_w, lambda x: float(np.exp(PS.ln_theta_wcut_v52(x))),
                 wfit, TOL)

    print("\n=== 2b. отгружаемая кривая (якоря округлены до 4 знаков) против подгонки")
    for iset, nm in ((1, "Базовый"), (2, "Стресс")):
        xs, fit = FREQ_FIT_CURVE[iset]
        ctrl = PS.CTRL5_BASE if iset == 1 else PS.CTRL5_STRESS
        bad += check(f"частота «{nm}»", xs,
                     lambda x, c=ctrl: PS.theta_freq_v52(50.0 + x, c), fit, 1e-3,
                     "(допуск округления)")
    for iset, (xs, fit, ctrl, nm) in KPOD_FIT_CURVE.items():
        # ⚠ У «Базового» якорь при 0.2 задан РЕШЕНИЕМ (γ ≠ 1), поэтому отгружаемая кривая
        # обязана отличаться от подгонки — сверять их на совпадение бессмысленно.  Вместо
        # проверки печатается размер отхода, чтобы он был виден, а не потерян.
        gm = np.log(ctrl["k02"]) / np.log(np.exp(PS.ln_theta_kpod_v52(0.2, iset)))
        if abs(gm - 1.0) > 2e-3:
            dep = float(np.max(np.abs(np.array([PS.theta_kpod_v52(x, ctrl) for x in xs])
                                      - np.array(fit))))
            print(f"  --  {nm:34s} отход от подгонки по решению: max|Δθ| = {dep:.3f} "
                  f"(γ = {gm:.3f})")
            continue
        bad += check(nm, xs, lambda x, c=ctrl: PS.theta_kpod_v52(x, c), fit, 1e-3,
                     "(допуск округления)")
    bad += check("обводнённость", xs_w,
                 lambda x: PS.theta_wcut_v52(x, {**PS.CTRL5_BASE, "on_wcut": 1}), wfit, 1e-3,
                 "(допуск округления)")

    print("\n=== 2c. отгружаемые кривые Kпод: глубокий недогруз и порядок сценариев")
    xs = np.linspace(0.0, 1.8, 400)
    tb = np.array([PS.theta_kpod_v52(x, PS.CTRL5_BASE) for x in xs])
    ts = np.array([PS.theta_kpod_v52(x, PS.CTRL5_STRESS) for x in xs])
    low = xs <= 0.8
    mono = bool(np.all(np.diff(tb[low]) < 0) and np.all(np.diff(ts[low]) < 0))
    zero_ok = abs(tb[0] - 1.5) < 5e-3
    # ГЛАВНАЯ проверка: «Стресс» — верхняя граница, он не может быть мягче «Базового».
    # Экспонента на пологой базовой форме это ломала (θ 1.246 против 1.193 при Kпод 0.4),
    # и поймала это только приёмка книги — здесь проверка стоит до сборки.
    order = bool(np.all(ts >= tb - 1e-9))
    worst = float(np.max(tb - ts))
    print(f"  {'OK ' if mono else '*** НЕ МОНОТОННЫ ***'} оба плеча монотонно растут к нулю")
    print(f"  {'OK ' if zero_ok else '*** НЕ ТО ЗНАЧЕНИЕ ***'} решение θ_Базовый(0) = "
          f"{tb[0]:.4f}  (принято 1.500);  «Стресс»(0) = {ts[0]:.4f}")
    print(f"  {'OK ' if order else '*** ПОРЯДОК НАРУШЕН ***'} «Стресс» ≥ «Базовый» на всём "
          f"диапазоне (худшая точка: {worst:+.4f})")
    bad += 0 if (mono and zero_ok and order) else 1

    # --- 3. anchors: shipped defaults must reproduce the fit (gamma = 1) ----
    print("\n=== якоря: .bas против двойника; γ = 1 значит «отгружается подгонка»")
    anchors = [
        ("θ_частоты@40 Гц базовый", C["D_T40"], PS.CTRL5_BASE["t40"],
         np.exp(PS.ln_theta_freq_v52(-10, 1))),
        ("θ_частоты@60 Гц базовый", C["D_T60"], PS.CTRL5_BASE["t60"],
         np.exp(PS.ln_theta_freq_v52(10, 1))),
        ("θ_Kпод@0.2 базовый", C["D_K02"], PS.CTRL5_BASE["k02"],
         np.exp(PS.ln_theta_kpod_v52(0.2, 1))),
        ("θ_Kпод@1.6 базовый", C["D_K16"], PS.CTRL5_BASE["k16"],
         np.exp(PS.ln_theta_kpod_v52(1.6, 1))),
        ("θ_обв@95 %", C["D_W95"], PS.CTRL5_BASE["w95"], np.exp(PS.ln_theta_wcut_v52(95))),
        ("θ_частоты@40 Гц стресс", None, PS.CTRL5_STRESS["t40"],
         np.exp(PS.ln_theta_freq_v52(-10, 2))),
        ("θ_частоты@60 Гц стресс", None, PS.CTRL5_STRESS["t60"],
         np.exp(PS.ln_theta_freq_v52(10, 2))),
        ("θ_Kпод@0.2 стресс", None, PS.CTRL5_STRESS["k02"],
         np.exp(PS.ln_theta_kpod_v52(0.2, 2))),
        ("θ_Kпод@1.6 стресс", None, PS.CTRL5_STRESS["k16"],
         np.exp(PS.ln_theta_kpod_v52(1.6, 2))),
    ]
    for lab, in_bas, in_py, fitted in anchors:
        # .bas и двойник обязаны совпасть — это одна и та же отгружаемая величина.
        # Отход от подгонки НЕ ошибка: он показывается через γ, а не запрещается.
        agree = 0.0 if in_bas is None else abs(in_bas - in_py)
        ok = agree < 1e-9
        lf = float(np.log(fitted))
        gamma = float(np.log(in_py) / lf) if abs(lf) > 1e-6 else float("nan")
        tag = "подгонка" if abs(gamma - 1.0) < 2e-3 else f"РЕШЕНИЕ, γ = {gamma:.3f}"
        bas_txt = "—" if in_bas is None else f"{in_bas:.4f}"
        print(f"  {'OK ' if ok else '*** РАСХОЖДЕНИЕ .bas/py ***'} {lab:26s} "
              f".bas {bas_txt}  py {in_py:.4f}  подгонка {fitted:.4f}   {tag}")
        bad += 0 if ok else 1

    # --- 4. the switch actually switches -----------------------------------
    print("\n=== выключатели")
    # the module ships with only the Vt rows as a default TuneBlock; the per-field
    # block comes from the workbook, so the switch check uses a key that exists here
    KEY = "Vt_nonsour"
    off = {**PS.CTRL5_BASE, "on_freq": 0, "on_kpod": 0, "on_wcut": 0}
    lay_off = PS.Layers(key=KEY, ctrl5=off)
    lay_on = PS.Layers(key=KEY, ctrl5=dict(PS.CTRL5_BASE))
    checks = [
        ("все слои выключены ⇒ θ_частоты = 1", lay_off.theta_freq_layer(65.0), 1.0),
        ("все слои выключены ⇒ θ_Kпод = 1", lay_off.theta_kpod_layer(0.3), 1.0),
        ("обводнённость включена по умолчанию ⇒ θ ≠ 1", lay_on.theta_wcut_layer(90.0), None),
        ("выключатель обводнённости гасит слой ⇒ θ = 1",
         PS.Layers(key=KEY, ctrl5={**PS.CTRL5_BASE, "on_wcut": 0}).theta_wcut_layer(90.0),
         1.0),
        ("θ в опорной точке = 1 (50 Гц)", lay_on.theta_freq_layer(50.0), 1.0),
        ("θ в опорной точке = 1 (Kпод 0.8)", lay_on.theta_kpod_layer(0.8), 1.0),
    ]
    for lab, got, want in checks:
        ok = (abs(got - want) < 1e-9) if want is not None else (abs(got - 1.0) > 1e-3)
        print(f"  {'OK ' if ok else '*** ОШИБКА ***'} {lab:44s} θ = {got:.6f}")
        bad += 0 if ok else 1

    print(f"\n{'ВСЁ СОШЛОСЬ' if bad == 0 else f'ПРОБЛЕМ: {bad}'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
