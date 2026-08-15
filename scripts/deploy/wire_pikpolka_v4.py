"""Wire ``ModuleFailureV4`` into the ПикПолка calculator and verify it against Python.

    python scripts/deploy/wire_pikpolka_v4.py --check     # compare only, no Excel
    python scripts/deploy/wire_pikpolka_v4.py             # inject + verify + save

The module is a *deploy form*: it must reproduce the Python model exactly, not approximately.
So this script carries an independent Python mirror of the VBA arithmetic and diffs the two on
a grid before the workbook is saved — a port that "looks right" in the sheet has been wrong
before, and only a numeric diff catches it.

Known Excel-COM traps this script works around (all three have bitten this workbook):
  * a ``.bas`` with non-ASCII bytes imports as mojibake — the module is kept pure ASCII and
    asserted so here, with Cyrillic built via ``ChrW()`` inside the VBA;
  * an orphan ``EXCEL.EXE`` makes ``Workbooks.Open`` return a silently **ReadOnly** book, so
    the save is a no-op — checked explicitly after opening;
  * ``Sheets.Delete`` no-ops once the VBProject has been edited in the same session.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

WORKBOOK = Path.home() / "Downloads" / "2026_07_Калькулятор_ПикПолка_v3_fix.xlsm"
MODULE = REPO_ROOT / "vba" / "ModuleFailureV4.bas"
MODULE_NAME = "ModuleFailureV4"

# --- the parameter set being wired in --------------------------------------
FREQ = (1.8, 0.9, 1.3, 3.75)                      # θ@40, θ@50, θ@60, θ@70
KPOD = (2.0, 1.25, 0.70, 0.95, 2.0, 2.0)          # θ@0.2, θ@1.2, plateau lo/hi, shape lo/hi

QNOM_KNOTS = (60.0, 100.0, 160.0, 250.0, 400.0, 640.0, 1000.0, 1600.0)
QNOM_NS = (0.817254, 0.754386, 0.919780, 1.0, 1.304720, 1.635608, 1.638240, 1.826061)
QNOM_SR = (0.555767, 0.790572, 1.051675, 1.0, 1.115540, 1.334801, 1.366776, 1.203401)
BASE = {False: (1.2511259, 517.84469), True: (1.3447649, 200.64525)}
CONTRACTOR = {"brt": 1.0, "slb": 1.19994, "oth": 1.90976}

FREQ_CLAMP = (35.0, 70.0)
KPOD_CLAMP = (0.15, 1.8)
KPOD_B2 = 0.3


# ---------------------------------------------------------------------------
# Python mirror of the VBA (deliberately written from the .bas, not imported
# from the analysis package — two independent expressions of the same model)
# ---------------------------------------------------------------------------
def theta_qnom(is_sour: bool, qnom: float) -> float:
    th = QNOM_SR if is_sour else QNOM_NS
    if qnom <= 0:
        return 1.0
    q = float(np.clip(qnom, QNOM_KNOTS[0], QNOM_KNOTS[-1]))
    return float(np.exp(np.interp(np.log(q), np.log(QNOM_KNOTS), np.log(th))))


def theta_freq(f: float, t40: float, t50: float, t60: float, t70: float = 0.0) -> float:
    if f <= 0 or min(t40, t50, t60) <= 0:
        return 1.0
    h = 10.0
    la, lb, lc = np.log(t40), np.log(t50), np.log(t60)
    a = lb
    c = (la + lc - 2.0 * lb) / (2.0 * h * h)
    b = (lc - la) / (2.0 * h)
    d = 0.0
    if t70 > 0:
        u4 = 20.0
        den = u4 ** 3 - h * h * u4
        if abs(den) > 1e-9:
            d = (np.log(t70) - (a + b * u4 + c * u4 * u4)) / den
    b -= d * h * h
    u = float(np.clip(f, *FREQ_CLAMP)) - 50.0
    return float(np.exp(a + b * u + c * u * u + d * u ** 3))


def theta_kpod(k: float, k02: float, k12: float, pl_lo: float, pl_hi: float,
               sh_lo: float, sh_hi: float) -> float:
    if k <= 0:
        return 1.0
    p_lo, p_hi = min(pl_lo, pl_hi), max(pl_lo, pl_hi)
    pl, ph = max(sh_lo, 1.0), max(sh_hi, 1.0)
    kk = float(np.clip(k, *KPOD_CLAMP))
    if p_lo <= kk <= p_hi:
        return 1.0
    v_lo, v_hi = abs(np.log(0.2 / p_lo)), abs(np.log(1.2 / p_hi))
    c_l = (k02 - 1.0) * (1.0 + KPOD_B2 * v_lo ** pl) / v_lo ** pl
    c_r = (k12 - 1.0) * (1.0 + KPOD_B2 * v_hi ** ph) / v_hi ** ph
    if kk < p_lo:
        v = np.log(p_lo / kk)
        return float(1.0 + c_l * v ** pl / (1.0 + KPOD_B2 * v ** pl))
    v = np.log(kk / p_hi)
    return float(1.0 + c_r * v ** ph / (1.0 + KPOD_B2 * v ** ph))


def rmst730(eta: float, beta: float) -> float:
    """∫₀⁷³⁰ S(t)dt for a Weibull — the module's RMST730v4, via the same incomplete gamma."""
    from scipy.special import gammainc, gamma
    if eta <= 0 or beta <= 0:
        return 0.0
    a = 1.0 / beta
    x = (730.0 / eta) ** beta
    return float(min(730.0, eta * gamma(a + 1.0) * gammainc(a, x)))


def norm_name(s: str) -> str:
    """Lowercase, then keep only latin/cyrillic letters and digits — mirrors ``NormName4``."""
    s = s.strip().lower()
    return "".join(ch for ch in s
                   if ("a" <= ch <= "z") or ("0" <= ch <= "9")
                   or ("а" <= ch <= "я") or ch == "ё")


def contractor_mult(name: str) -> float:
    """brt / slb / oth from a free-text contractor name.

    The Cyrillic needles are lowercase because the name is normalised first — matching them
    against capital forms silently classified every "Борец" as ``oth``.
    """
    if not name.strip():                 # VBA tests Len() BEFORE normalising
        return CONTRACTOR["brt"]
    s = norm_name(name)
    if any(t in s for t in ("бор", "borets", "borec", "brt")):
        return CONTRACTOR["brt"]
    if any(t in s for t in ("шлю", "слай",
                            "слб", "slb", "schlumberger")):
        return CONTRACTOR["slb"]
    return CONTRACTOR["oth"]


def life_days(is_sour: bool, contractor: str, qnom: float, freq: float, kpod: float) -> float:
    b0, e0 = BASE[is_sour]
    theta = (contractor_mult(contractor)
             * theta_qnom(is_sour, qnom)
             * theta_freq(freq, *FREQ)
             * theta_kpod(kpod, *KPOD))
    return rmst730(e0 * theta ** (-1.0 / b0), b0)


# ---------------------------------------------------------------------------
#: Names as they actually appear in the workbook, not just the canonical codes — the
#: contractor classifier is where a silent, forecast-wide error hid once already.
CONTRACTOR_NAMES = ("brt", "Борец", "борец", "БОРЕЦ", "Borets", "slb", "Шлюмберже",
                    "СЛБ", "Schlumberger", "Новомет", "")


def grid() -> list[tuple]:
    """Verification grid: the operating-point corners on the canonical codes, plus a sweep
    over the contractor names as they are actually spelled in the workbook."""
    out = []
    for sour in (0, 1):
        for c in ("brt", "slb", "oth"):
            for q in (50.0, 60.0, 250.0, 640.0, 1600.0, 2500.0):
                for f in (30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 75.0):
                    for k in (0.1, 0.2, 0.7, 0.8, 0.95, 1.2, 1.8, 2.5):
                        out.append((sour, c, q, f, k))
        for c in CONTRACTOR_NAMES:
            out.append((sour, c, 250.0, 50.0, 0.825))
    return out


def check_only() -> None:
    print(f"freq (θ@40,θ@50,θ@60,θ@70) = {FREQ}")
    g = np.array([35, 40, 45, 50, 55, 60, 65, 70.0])
    print("  f :", " ".join(f"{x:8.0f}" for x in g))
    print("  θ :", " ".join(f"{theta_freq(x, *FREQ):8.4f}" for x in g))
    print(f"\nKpod (θ@0.2,θ@1.2,plateau,shape) = {KPOD}")
    gk = np.array([0.15, 0.2, 0.4, 0.6, 0.7, 0.8, 0.95, 1.0, 1.2, 1.5, 1.8])
    print("  k :", " ".join(f"{x:8.2f}" for x in gk))
    print("  θ :", " ".join(f"{theta_kpod(x, *KPOD):8.4f}" for x in gk))
    print("\nθ_Qnom knots")
    for lab, th in (("nonsour", QNOM_NS), ("sour", QNOM_SR)):
        print(f"  {lab:8s}", " ".join(f"{k:g}:{t:.3f}" for k, t in zip(QNOM_KNOTS, th)))
    print("\nlife at the reference point (brt, Qnom 250, 50 Hz, Kpod 0.825):")
    for sour, lab in ((0, "nonsour"), (1, "sour")):
        b0, e0 = BASE[bool(sour)]
        bare = rmst730(e0, b0)
        ref = life_days(bool(sour), "brt", 250.0, 50.0, 0.825)
        print(f"  {lab:8s} bare baseline (θ=1) {bare:6.1f} d   at reference {ref:6.1f} d"
              f"   (θ_freq(50) = {theta_freq(50.0, *FREQ):.3f})")


def deploy(check: bool) -> int:
    src = MODULE.read_text(encoding="utf-8")
    assert all(ord(ch) < 128 for ch in src), "module must be pure ASCII for the COM import"
    check_only()
    if check:
        return 0

    import shutil
    import win32com.client as win32

    if not WORKBOOK.exists():
        print(f"!! workbook not found: {WORKBOOK}")
        return 2
    # Never clobber an existing backup: re-running this script would otherwise overwrite the
    # pristine pre-change copy with an already-modified one, which is exactly when a backup
    # matters most.  Later runs get numbered siblings instead.
    backup = WORKBOOK.with_suffix(".PRE_UNIFIED_V4.xlsm")
    if backup.exists():
        i = 2
        while WORKBOOK.with_suffix(f".PRE_UNIFIED_V4.{i}.xlsm").exists():
            i += 1
        backup = WORKBOOK.with_suffix(f".PRE_UNIFIED_V4.{i}.xlsm")
    shutil.copy2(WORKBOOK, backup)
    print(f"\nbackup -> {backup.name}")

    app = win32.gencache.EnsureDispatch("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.AutomationSecurity = 1
    wb = None
    try:
        wb = app.Workbooks.Open(str(WORKBOOK), UpdateLinks=0)
        if wb.ReadOnly:                       # the orphan-EXCEL.EXE trap: saving would no-op
            raise RuntimeError("workbook opened READ-ONLY — close every Excel instance first")
        proj = wb.VBProject
        for comp in list(proj.VBComponents):
            if comp.Name == MODULE_NAME:
                proj.VBComponents.Remove(comp)
                print(f"removed old {MODULE_NAME}")
                break
        proj.VBComponents.Import(str(MODULE))
        print(f"imported {MODULE_NAME}")

        bad, worst = 0, 0.0
        for sour, c, q, f, k in grid():
            got = float(app.Run(f"{MODULE_NAME}.CM4_LifeDays", float(sour), c, q, f, k))
            want = life_days(bool(sour), c, q, f, k)
            diff = abs(got - want)
            worst = max(worst, diff)
            if diff > 1e-6:
                bad += 1
                if bad <= 8:
                    print(f"  MISMATCH sour={sour} {c} q={q} f={f} k={k}: "
                          f"VBA {got:.6f} vs py {want:.6f}")
        n = len(grid())
        print(f"\nCM4_LifeDays: {n - bad}/{n} exact, worst |Δ| = {worst:.2e} days")

        for f_ in (35.0, 40.0, 50.0, 60.0, 70.0):
            g_ = float(app.Run(f"{MODULE_NAME}.CM4_ThetaFreq", f_))
            assert abs(g_ - theta_freq(f_, *FREQ)) < 1e-9, f"θ_freq({f_}) {g_}"
        for k_ in (0.2, 0.7, 0.8, 0.95, 1.2, 1.8):
            g_ = float(app.Run(f"{MODULE_NAME}.CM4_ThetaKpod", k_))
            assert abs(g_ - theta_kpod(k_, *KPOD)) < 1e-9, f"θ_Kpod({k_}) {g_}"
        for s_ in (0, 1):
            for q_ in QNOM_KNOTS:
                g_ = float(app.Run(f"{MODULE_NAME}.CM4_ThetaQnom", float(s_), q_))
                assert abs(g_ - theta_qnom(bool(s_), q_)) < 1e-9, f"θ_Qnom({s_},{q_}) {g_}"
        print("layer UDFs (θ_freq / θ_Kpod / θ_Qnom): exact at every anchor and knot")

        for s_, lab in ((0, "nonsour"), (1, "sour")):
            print(f"  CM4_LifeAtRef({lab}) = "
                  f"{float(app.Run(f'{MODULE_NAME}.CM4_LifeAtRef', float(s_))):.2f} d   "
                  f"CM4_RmstRef = {float(app.Run(f'{MODULE_NAME}.CM4_RmstRef', float(s_))):.2f} d")

        if bad:
            print("\n!! NOT SAVED — VBA disagrees with Python")
            return 1
        wb.Save()
        print(f"\nsaved {WORKBOOK.name}")
        return 0
    finally:
        if wb is not None:
            wb.Close(SaveChanges=False)
        app.Quit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="print the layers, do not touch Excel")
    return deploy(ap.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
