"""Weibull-скан по стратам: c x k x эстиманд x часы, с отчётом RMST(0)/MRL(0).

Реализация типового процесса из ``docs/notes/production_risk_weibull_fit_workflow.md``
для ПРОИЗВОЛЬНОЙ страты или месторождения.

Ключевые решения, зашитые здесь (обоснование — в заметке):

* **Эстиманд — явный параметр, а не умолчание.** ``ESTIMAND_ALL_CAUSE`` (любой подъём =
  событие) даёт МРП для графика бригад; ``ESTIMAND_CAUSE_SPECIFIC`` (ГТМ/ППР цензурируется)
  даёт интенсивность отказов для KPI.  Это РАЗНЫЕ величины: на Ya они разошлись в 1.8x.
  Оба эстиманда цензурируют работающие насосы на ``as_of`` — это делает
  ``esp_population.build``.
* **Отсечка c режет ранние ОТКАЗЫ, цензуру оставляет** (определение A).  Левое усечение
  ``t > c`` выбросило бы и короткие цензурированные пробеги — это другая величина.
* **c=60/90 не поддерживаются**: член ``-n*log S(c)`` разваливается на клипе 1e-300.
* **Отчёт — RMST(0) и MRL(0), не B50** (стандартное требование, см. память проекта):
  ``MRL(0)`` = средний ресурс со ВСЕМ хвостом, ``RMST(0)`` = среднее до ``tau``=p95
  наблюдённых, хвост не выдумываем.  ``rmst_km`` — непараметрический контроль по KM:
  ``rmst_model/rmst_km`` ~ 1.00 подтверждает форму, расхождение >10% означает, что
  Вейбулл искажает кривую.
* **k2 требует ограничений** ``beta2 > max(1,beta1)``, ``eta2 >= 2*eta1``: без них
  компоненты меняются местами.  Если ``b2 - max(1,b1) < 0.05`` (``b2_at_bound``), ограничение
  ПРИЖАТО — ``beta2`` тогда регуляризатор, а не измерение, и читать его как износ нельзя.
  ``w1_degenerate`` (w1>0.75 или <0.05) означает, что смесь схлопнулась в одну компоненту.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gamma as _gamma

from analysis.workflows.production_risk import esp_population as P

ESTIMAND_ALL_CAUSE = "all_cause"
ESTIMAND_CAUSE_SPECIFIC = "cause_specific"
SUPPORTED_CUTS = (0, 3, 7, 30)
B2_BOUND_TOL = 0.05
W1_DEGENERATE_HI = 0.75
W1_DEGENERATE_LO = 0.05
RMST_TAU_PCT = 95.0


@dataclass
class ScanRow:
    stratum: str
    estimand: str
    clock: str
    c: int
    k: int
    n: int
    events: int
    censored: int
    beta: float          # k=1: shape; k=2: beta ДОМИНИРУЮЩЕЙ компоненты
    eta: float           # k=1: scale; k=2: eta доминирующей компоненты
    mrl0: float
    rmst0: float
    rmst_km: float
    tau: float
    model_over_km: float
    w1: float = float("nan")
    beta1: float = float("nan")
    eta1: float = float("nan")
    beta2: float = float("nan")
    eta2: float = float("nan")
    b2_at_bound: bool = False
    w1_degenerate: bool = False
    converged: bool = True
    note: str = ""


def build_stratum(
    as_of: date | pd.Timestamp,
    field: str,
    h2s: str = "nonsour",
    contractor: str | list[str] | None = None,
    estimand: str = ESTIMAND_ALL_CAUSE,
    pop: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Вернуть (tte, event, meta) для страты на КАЛЕНДАРНОМ клоке.

    ``contractor`` — один код ("brt"), список (["brt", "slb"] — объединённая страта)
    или None (все подрядчики).

    Работающие насосы цензурируются на ``as_of`` — они ОБЯЗАТЕЛЬНЫ: выбросить их
    (как делает витрина V03) значит выбросить самых долгоживущих и занизить ресурс.
    """
    if estimand not in (ESTIMAND_ALL_CAUSE, ESTIMAND_CAUSE_SPECIFIC):
        raise ValueError(f"estimand must be one of the ESTIMAND_* constants; got {estimand!r}")
    gtm_is_failure = estimand == ESTIMAND_ALL_CAUSE
    if pop is None:
        pop = P.build(as_of, gtm_is_failure=gtm_is_failure)
    # Фильтруем по подрядчикам сами — P.select берёт только одного.
    g = P.select(pop, field, h2s=h2s, contractor=None)
    if contractor is not None:
        groups = [contractor] if isinstance(contractor, str) else list(contractor)
        g = g[g["contractor_group"].isin(groups)]
    g = g[g["tte"] > 0]
    t = g["tte"].to_numpy(float)
    e = g["event"].to_numpy(int)
    meta = {
        "n": len(t),
        "events": int(e.sum()),
        "censored": int((e == 0).sum()),
        "running": int(g["end"].isna().sum()),
        "clock": "cal",
    }
    return t, e, meta


def apply_cut(t: np.ndarray, e: np.ndarray, c: int) -> tuple[np.ndarray, np.ndarray]:
    """Определение A: выбросить пробеги, ОТКАЗАВШИЕ раньше c; цензуру оставить."""
    if c not in SUPPORTED_CUTS:
        raise ValueError(f"c={c} not supported; use one of {SUPPORTED_CUTS} (60/90 численно ломаются)")
    keep = ~((e == 1) & (t < c))
    return t[keep], e[keep]


def _surv_k1(beta: float, eta: float):
    def S(x):
        x = np.maximum(np.asarray(x, float), 0.0)
        return np.exp(-((x / eta) ** beta))
    return S


def _surv_k2(w1: float, b1: float, e1: float, b2: float, e2: float):
    def S(x):
        x = np.maximum(np.asarray(x, float), 0.0)
        return w1 * np.exp(-((x / e1) ** b1)) + (1 - w1) * np.exp(-((x / e2) ** b2))
    return S


def _rmst_km(t: np.ndarray, e: np.ndarray, tau: float) -> float:
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(t, e)
    sf = km.survival_function_
    x = sf.index.to_numpy(float)
    y = sf.iloc[:, 0].to_numpy(float)
    m = x <= tau
    if m.sum() < 2:
        return float("nan")
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(y[m], x[m]))


def fit(
    t: np.ndarray,
    e: np.ndarray,
    k: int,
    stratum: str,
    estimand: str,
    c: int,
    clock: str = "cal",
    num_starts: int = 60,
) -> ScanRow:
    """Одна ячейка скана: подогнать k=1 или k=2 и посчитать RMST(0)/MRL(0)."""
    tau = float(np.percentile(t, RMST_TAU_PCT))
    common = dict(stratum=stratum, estimand=estimand, clock=clock, c=c, k=k,
                  n=len(t), events=int(e.sum()), censored=int((e == 0).sum()), tau=tau)

    if k == 1:
        from lifelines import WeibullFitter

        wf = WeibullFitter().fit(t, e)
        beta, eta = float(wf.rho_), float(wf.lambda_)
        S = _surv_k1(beta, eta)
        mrl0 = eta * _gamma(1.0 + 1.0 / beta)
        row = dict(beta=beta, eta=eta, w1=float("nan"))
    elif k == 2:
        from analysis.models.survival.weibull_em import fit_latent_weibull_em

        r = fit_latent_weibull_em(t, e, num_starts=num_starts)
        m = r.model
        w1 = float(m.weight_1)
        b1, e1 = float(m.component_1.beta), float(m.component_1.eta)
        b2, e2 = float(m.component_2.beta), float(m.component_2.eta)
        S = _surv_k2(w1, b1, e1, b2, e2)
        # E[T] смеси = взвешенная сумма средних компонент
        mrl0 = w1 * e1 * _gamma(1.0 + 1.0 / b1) + (1 - w1) * e2 * _gamma(1.0 + 1.0 / b2)
        dominant_1 = w1 >= 0.5
        row = dict(
            beta=b1 if dominant_1 else b2,       # читаем ДОМИНИРУЮЩУЮ, не beta2
            eta=e1 if dominant_1 else e2,
            w1=w1, beta1=b1, eta1=e1, beta2=b2, eta2=e2,
            b2_at_bound=bool((b2 - max(1.0, b1)) < B2_BOUND_TOL),
            w1_degenerate=bool(w1 > W1_DEGENERATE_HI or w1 < W1_DEGENERATE_LO),
            converged=bool(r.success),
            note=str(r.message or "")[:60],
        )
    else:
        raise ValueError(f"k must be 1 or 2; got {k}")

    rmst0 = float(quad(S, 0.0, tau, limit=200)[0])
    km = _rmst_km(t, e, tau)
    return ScanRow(mrl0=float(mrl0), rmst0=rmst0, rmst_km=km,
                   model_over_km=rmst0 / km if km and np.isfinite(km) else float("nan"),
                   **common, **row)


def scan(
    as_of: date | pd.Timestamp,
    strata: list[tuple[str, str, str | list[str] | None]],
    estimand: str = ESTIMAND_ALL_CAUSE,
    cuts: tuple[int, ...] = SUPPORTED_CUTS,
    ks: tuple[int, ...] = (1, 2),
    num_starts: int = 60,
    pop: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Полный скан. ``strata`` = [(label, field, contractor|None), ...].

    k=2 стоит минуты на тысячах пробегов — для больших популяций гоните ячейки
    параллельно (см. ``scripts/run/weibull_scan.py``).
    """
    if pop is None:
        pop = P.build(as_of, gtm_is_failure=estimand == ESTIMAND_ALL_CAUSE)
    out: list[dict] = []
    for label, field, contractor in strata:
        t, e, _meta = build_stratum(as_of, field, contractor=contractor,
                                    estimand=estimand, pop=pop)
        for c in cuts:
            tc, ec = apply_cut(t, e, c)
            if ec.sum() < 5:
                continue
            for k in ks:
                out.append(asdict(fit(tc, ec, k, label, estimand, c, num_starts=num_starts)))
    return pd.DataFrame(out)
