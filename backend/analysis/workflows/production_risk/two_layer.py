"""Two-layer failure model: NORMAL (plannable) + INFANT (risk uplift).

Why two layers
--------------
A workover schedule is a resource commitment — you cannot *plan* for a new pump to
fail on day 8.  But those failures still consume rigs, pumps and oil, so they must
appear in DEMAND.  One curve cannot serve both jobs, and the shipped single Weibull
proves it: it reproduces the fleet total correctly (fact/model 0.98–1.05) only by
smearing an infant spike it cannot represent into a plateau it then over-states.
Adding an infant multiplier on top therefore double-counts (it pushed Vt to 1.18).

So split the OUTPUT rather than fight the schema:

  NORMAL  — Weibull with FREE beta, fitted on runs left-truncated past the infant
            window (i.e. conditional on surviving it).  Drives the SCHEDULE: the
            per-well `0`s, rig and crew booking.
  INFANT  — a probability per INSTALLATION, not per calendar day.  Every install —
            a new well, or the pump that follows any workover — carries it.  Drives
            DEMAND and oil loss as an aggregate uplift, never a dated row.

What the data says about the two layers (2026-07-17, corrected population)
--------------------------------------------------------------------------
* ~20% of failures land in the first 30 days, of which ~10–18% is genuine EXCESS
  over what the normal curve would produce on the same exposure.
* The infant timescale is a parameter, not a constant: freeing it, Ya/Vt/Za/Mc
  collapse to a day-0 startup mass (w1 ≈ 2–3.5%) while Ic finds a real 17.5-day
  window carrying 13%.  Hence `cut_days` is a knob, not a hard-coded 30.
* NORMAL is NOT a plateau: with beta free it fits **0.74–0.86**, a slowly declining
  hazard.  It is also NOT wear-out — four independent methods agree (raw age bands,
  left truncation, free mixture, and a shared gamma-frailty model that lifts Ya's
  beta only 0.717 → 0.740 despite real well-level frailty, theta=0.11, p=2.5e-05).
  So `beta > 1` is not available on the well-evidenced fields; Ic is the exception
  (beta 1.13).

Consequence worth stating to planners: beta < 1 means expected remaining life GROWS
slowly with age, so the oldest pumps are the LEAST urgent — the opposite of
"replace the oldest first".  Prioritise by risk = E x oil.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DEFAULT_CUT_DAYS = 30.0
CUT_CHOICES = (7.0, 30.0, 60.0, 90.0)


@dataclass
class Layers:
    cut_days: float
    # --- normal (plannable) ---
    beta: float
    eta: float
    b50: float
    n_runs_normal: int
    n_events_normal: int
    # --- infant (risk uplift) ---
    p_infant: float          # excess probability of an early failure PER INSTALL
    n_events_infant: int     # observed failures inside [0, cut)
    n_expected_normal: float # what the normal curve alone gives inside [0, cut)
    excess: float            # infant failures that the normal curve cannot explain
    n_runs_total: int
    n_events_total: int

    def as_dict(self) -> dict:
        return asdict(self)


def _fit_truncated_weibull(t: np.ndarray, e: np.ndarray, cut: float, restarts: int = 14) -> tuple[float, float]:
    """Weibull MLE conditional on survival past `cut` — the NORMAL layer.

    Conditioning is what keeps the infant mode out of the fit: a Weibull's hazard
    past `cut` is unchanged by the conditioning, so (beta, eta) describe the normal
    process and can be evaluated from age 0 as the "had there been no infant mode"
    counterfactual the schedule needs.
    """
    m = t > cut
    t, e = t[m], e[m]
    if e.sum() < 5:
        raise ValueError(f"only {int(e.sum())} events past cut={cut}; too few to fit")

    def nll(x: np.ndarray) -> float:
        b, eta = np.exp(np.clip(x, -20, 20))
        z = (np.maximum(t, 1e-9) / eta) ** b
        zc = (cut / eta) ** b if cut > 0 else 0.0
        ll = np.sum(np.where(e == 1, np.log(b / eta) + (b - 1) * np.log(np.maximum(t, 1e-9) / eta) - z, -z))
        ll += len(t) * zc  # left truncation: divide by S(cut)
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for s in range(restarts):
        rng = np.random.default_rng(s)
        x0 = [np.log(rng.uniform(0.4, 2.5)), np.log(rng.uniform(200, 2000))]
        r = minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=12000, fatol=1e-10, xatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    b, eta = np.exp(best.x)
    return float(b), float(eta)


def survival(t, beta: float, eta: float) -> np.ndarray:
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    return np.exp(-((t / eta) ** beta))


def fit(pop: pd.DataFrame, cut_days: float = DEFAULT_CUT_DAYS) -> Layers:
    """Split a run population into the normal and infant layers."""
    t = pop["tte"].to_numpy(float)
    e = pop["event"].to_numpy(int)
    beta, eta = _fit_truncated_weibull(t, e, float(cut_days))

    # Infant excess = what actually happened inside the window MINUS what the normal
    # curve alone would have produced on exactly the same exposure.
    observed = int(((t < cut_days) & (e == 1)).sum())
    horizon = np.minimum(t, cut_days)
    expected = float(np.sum(1.0 - survival(horizon, beta, eta)))
    excess = observed - expected
    p_infant = float(np.clip(excess / max(len(t), 1), 0.0, 1.0))

    return Layers(
        cut_days=float(cut_days),
        beta=beta,
        eta=eta,
        b50=float(eta * np.log(2) ** (1.0 / beta)),
        n_runs_normal=int((t > cut_days).sum()),
        n_events_normal=int(((t > cut_days) & (e == 1)).sum()),
        p_infant=p_infant,
        n_events_infant=observed,
        n_expected_normal=expected,
        excess=float(excess),
        n_runs_total=int(len(t)),
        n_events_total=int(e.sum()),
    )


def normal_registry_params(layers: Layers) -> dict[str, float]:
    """The NORMAL layer in the shipped 5-column registry form (a plain k1 Weibull).

    Evaluated from age 0 on purpose: this is the schedule's counterfactual — what
    the fleet would do if no pump ever died in infancy.
    """
    return {
        "w1": 0.0,
        "beta1": layers.beta,
        "eta1": layers.eta,
        "beta2": layers.beta,
        "eta2": layers.eta,
    }


def infant_uplift(n_installs: float, layers: Layers) -> float:
    """Extra (unschedulable) failures generated by `n_installs` new pumps.

    Recursive: an infant failure is itself a workover, whose replacement pump carries
    the same risk — so the geometric sum, not a single pass.
    """
    p = float(np.clip(layers.p_infant, 0.0, 0.95))
    return float(n_installs) * p / (1.0 - p)


def _rmst(tte: np.ndarray, pulled: np.ndarray, tau_pct: float = 95.0) -> tuple[float, float]:
    """Restricted mean survival time off the ALL-CAUSE KM — the honest mean cycle.

    Integrates the KM out to the `tau_pct` percentile of observed times, so no tail
    is invented beyond the data.  Returns (rmst, tau).
    """
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(tte, pulled)
    tau = float(np.percentile(tte, tau_pct))
    sf = km.survival_function_
    x = sf.index.to_numpy(dtype=float)
    y = sf.iloc[:, 0].to_numpy(dtype=float)
    m = x <= tau
    if m.sum() < 2:
        return float("nan"), tau
    return float(np.trapezoid(y[m], x[m]) if hasattr(np, "trapezoid") else np.trapz(y[m], x[m])), tau


def mrp_days(pop: pd.DataFrame, install_from: str | None = None) -> dict[str, float]:
    """МРП per stratum — mean days between pump pulls, ALL causes.

    Estimated as the **RMST of the all-cause Kaplan-Meier** (event = any pull,
    censored = still running).

    NOT `Σ наработка / N подъёмов`.  That ratio dumps the time of pumps that have
    never been pulled into the numerator while only ended runs reach the
    denominator, so it is unbiased ONLY under a constant hazard — and the pull
    hazard here is not constant (beta ~ 0.9).  Its error scales with the censored
    fraction, which is worst exactly where it matters most:

        Mc, installs 2024+ (35% still running):  Σt/N = 305  vs  RMST = 241  (+27%)
        Mc, all history    (29% still running):  Σt/N = 356  vs  RMST = 293  (+22%)
        Ya                 ( 8% still running):  Σt/N = 412  vs  RMST = 381  ( +8%)

    The same flaw infects `Σt / N_отказов` ("ННО"), and worse — only ~26% of Mc runs
    end in a failure, which is how it reaches 969 d against an observed failed-run
    mean of 191.  Use it as a RATE if at all, never as a life.

    Clock: `tte` is Свод «Наработка (сут)» ~= CALENDAR (median tte/calendar = 0.986
    over 1644 completed runs).  True op-days are ~0.90 of it (Mc 0.896, Ya 0.912).

    `install_from` filters on INSTALL date, not exposure: left-truncating exposure
    keeps pre-window installs' recent time and hides that recent runs are shorter
    (Mc: ГТМ mean 295 -> 182 once filtered properly).
    """
    g = pop.copy()
    if install_from is not None:
        g = g[g["install"] >= pd.Timestamp(install_from)]
    out: dict[str, float] = {}
    for stratum, x in g.groupby("stratum"):
        pulled = x["end"].notna().astype(int).to_numpy()
        if int(pulled.sum()) < 5:
            continue
        rmst, _ = _rmst(x["tte"].to_numpy(dtype=float), pulled)
        if np.isfinite(rmst) and rmst > 0:
            out[str(stratum)] = rmst
    return out


def conventional_events(age: float, mrp: float, horizon_days: float) -> list[float]:
    """Days-from-forecast-start of each workover under the RESOURCE convention.

    Rule: a pump is replaced every `mrp` operating days, so the next one falls due
    at `mrp - age`; a pump already past `mrp` is overdue and lands at day 0.

    This is a PLANNING POLICY, not a forecast.  The data says the opposite: with
    beta<1 the remaining life GROWS with age (Ya: 828 d at age 0 -> 1069 d at age
    900), and 33-35% of pumps outlive the mean by definition, so "overdue" is not a
    physical state.  The convention buys monotone, explainable rows — an old pump is
    scheduled before a new one — at the cost of aiming crews at the pumps LEAST
    likely to fail.  Use it because a schedule must be defensible, not because it
    predicts; label it as a convention wherever it is shown.
    """
    if mrp <= 0 or not np.isfinite(mrp):
        return []
    out: list[float] = []
    t = max(0.0, float(mrp) - max(float(age), 0.0))
    while t <= horizon_days:
        out.append(t)
        t += float(mrp)
    return out


def summary_table(layers_by_stratum: dict[str, Layers]) -> pd.DataFrame:
    rows = []
    for name, l in layers_by_stratum.items():
        rows.append({
            "страта": name,
            "пробегов": l.n_runs_total,
            "отказов": l.n_events_total,
            "окно, сут": l.cut_days,
            "beta (норма)": round(l.beta, 3),
            "eta (норма)": round(l.eta, 1),
            "B50 (норма)": round(l.b50),
            "отказов в окне": l.n_events_infant,
            "ожидалось нормой": round(l.n_expected_normal, 1),
            "избыток (младенч.)": round(l.excess, 1),
            "p_младенч / установку": round(l.p_infant, 4),
            "доля младенч.": f"{100 * l.excess / max(l.n_events_total, 1):.0f}%",
        })
    return pd.DataFrame(rows)
