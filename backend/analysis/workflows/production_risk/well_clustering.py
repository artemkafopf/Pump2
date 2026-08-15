"""How much of Ya's nominal sample size is real? — well-level clustering diagnostic.

Every run-level fit in this repo treats runs as independent.  Ya averages ~4.5 runs per
well (475 wells, 2127 runs), so if wells differ systematically the nominal ``n`` overstates
the information and every interval is too narrow.  This module measures the intra-well
correlation and reports *where* the understatement actually lands.

It answers three separate questions, because they have different answers:

1. **Is there a well effect?**  Shared gamma-frailty Weibull vs the independent fit
   (:mod:`analysis.models.survival.shared_frailty`), boundary-corrected LR test.
2. **Does it move the covariate conclusions?**  Cox with ``cluster_col`` (robust sandwich)
   against the naive fit, per covariate.  The inflation here is *not* the marginal design
   effect — a covariate that varies within a well barely inflates at all, and a slope can
   even tighten.  This has to be measured per covariate, never applied as a blanket factor.
3. **Does it move the deliverable?**  RMST(0, 730) — the ННО quantity — bootstrapped two
   ways: resampling runs (naive) and resampling *wells* (cluster).  This is where the
   understatement is real, because the baseline level is exactly the quantity the frailty
   variance attaches to.

Measured 2026-07-28 (see ``results/production_risk_well_clustering/``):

* Ya rho = 0.091 (p = 7.6e-08) — real, highly significant, modest in size.
* Covariate slope inflation 0.98–1.18x.  Vt_nonsour rho = 0.041 (p = 0.25).  The
  clustering asymmetry between the two fields is therefore ~1.1x on slopes and does **not**
  account for effects that hold on Ya and fail on Vt.
* RMST(0,730) SE inflation 1.31x on Ya vs 1.11x on Vt_nonsour — the forecast interval, not
  the covariate verdicts, is what clustering was hiding.
* Weibull shape 0.791 (independent) vs 0.828 (frailty-adjusted): the infant-hazard shape is
  **not** an unmodelled-heterogeneity artifact.

**Between-well variation dilutes the Qnom effect** — within > pooled in 6 of 6
population x sample-rule combinations, ratio 1.3-2.1x (:func:`bare_within`, single
covariate, identical sample rule across populations)::

    qnom complete case          beta_within   beta_pooled   wells / events
      Ya                          0.383         0.287        331 / 1182
      Vt_nonsour                  0.529         0.248         88 /  187
      Vt_ALL                      0.459         0.277        134 /  292
    restricted to 7-cov case
      Ya                          0.538         0.401        236 /  620
      Vt_nonsour                  0.443         0.266         63 /  120
      Vt_ALL                      0.405         0.295         99 /  197

Same direction as the pump-sizing landmark result (2.01 within vs 1.46 between), now
replicated on a second field.  The within-well estimates are statistically
indistinguishable across populations (Ya vs Vt_nonsour: p = 0.42 max-power, p = 0.70
restricted), so the pump-size response itself transfers.

⚠ **The apparent Ya-vs-Vt difference on Qnom is a Vt_nonsour identification failure, not
physics.**  It only appears in *multivariable* fits.  On a fixed 63-well / 120-event
sample the within-well coefficient runs 0.443 -> 1.519 -> 2.986 across ``SPEC_LADDER``
(Ya, on a fixed 236-well / 620-event sample: 0.538 -> 0.689 -> 0.788).  Qnom/Kpod/Ql are
~0.65 correlated *within* well and Vt_nonsour has too few informative wells to separate
them.  In the **bare** spec the pooled estimates agree across fields too (0.287 vs 0.248,
p = 0.68) — so multivariable Vt_nonsour numbers, not the fields, produced the discrepancy.

Practical consequence: **only Ya identifies a multivariable within-well fit.**  Ladder
drift across ``SPEC_LADDER``, each on its own fixed sample::

    Ya          0.538 -> 0.689 -> 0.788    1.5x   236 wells / 620 events
    Vt_ALL      0.405 -> 0.853 -> 1.942    4.8x    99 wells / 197 events
    Vt_nonsour  0.443 -> 1.519 -> 2.986    6.7x    63 wells / 120 events

Admitting the sour stratum roughly halves the instability but does not remove it.  Any
Vt-side verification should therefore be done one covariate at a time; a multivariable
Vt within-well coefficient is not a measurement whichever stratum set is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.models.survival.shared_frailty import fit_shared_frailty
from analysis.paths import results_dir
from analysis.workflows.production_risk import unified_v4 as U

__all__ = ["FIELDS", "COVARIATES", "frailty_table", "cluster_se_table", "rmst_table", "run"]

CLOCK = U.CLOCK
EVENT_COL = U.EVENT_COL

#: Stratum selections keyed by report label.  ``None`` means the whole field.
FIELDS: dict[str, tuple[str, str | None]] = {
    "Ya": ("Ya", None),
    "Vt_nonsour": ("Vt", "Vt_nonsour"),
    # Whole Vt is carried only as the fallback verification population: Vt_nonsour alone
    # has too few informative wells for a multivariable within-well fit (see module docs).
    "Vt_ALL": ("Vt", None),
}

#: Entered linearly on the log-hazard.  These are screening terms for the *inflation
#: ratio* — the shipped layers are polylines, so a null here is not a null for the layer
#: (a linear term cannot see the frequency valley).
COVARIATES = ["log_qnom", "kpod_run", "freq_dev", "wcut", "log_ql_t0", "slb", "oth"]

RMST_TAU = 730.0


def _prepare(field: str, stratum_prefix: str | None) -> pd.DataFrame:
    d = U.prepare_field(field)
    if stratum_prefix:
        d = d[d["stratum"].str.startswith(stratum_prefix)]
    d = d[d[CLOCK] > 0].copy()
    d["log_qnom"] = np.log(d["qnom"])
    d["log_ql_t0"] = np.log(d["ql"].clip(lower=1.0))
    return d


def frailty_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Frailty variance, ICC and design effect per field."""
    rows = []
    for label, d in frames.items():
        f = fit_shared_frailty(
            d, duration_col=CLOCK, event_col=EVENT_COL, cluster_col="code"
        )
        s = f.summary().set_index("metric")["value"]
        rows.append(pd.Series(s, name=label))
    return pd.DataFrame(rows)


def cluster_se_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Naive vs cluster-robust Cox standard errors, per covariate per field.

    ``frac_between`` is the share of the covariate's variance that lies between wells —
    it explains why the inflation differs so much across covariates.
    """
    from lifelines import CoxPHFitter

    rows = []
    for label, d in frames.items():
        keep = COVARIATES + [CLOCK, EVENT_COL, "code"]
        sub = d.dropna(subset=keep).copy()
        for c in COVARIATES:  # standardise so the betas are comparable across covariates
            if sub[c].std() > 0:
                sub[c] = (sub[c] - sub[c].mean()) / sub[c].std()
        naive = CoxPHFitter().fit(sub[COVARIATES + [CLOCK, EVENT_COL]], CLOCK, EVENT_COL)
        robust = CoxPHFitter().fit(sub[keep], CLOCK, EVENT_COL, cluster_col="code")
        for c in COVARIATES:
            gm = sub.groupby("code")[c].transform("mean")
            vb, vw = gm.var(), (sub[c] - gm).var()
            rows.append(
                dict(
                    field=label, covariate=c, n=len(sub), events=int(sub[EVENT_COL].sum()),
                    beta=naive.params_[c],
                    se_naive=naive.standard_errors_[c],
                    se_cluster=robust.standard_errors_[c],
                    se_inflation=robust.standard_errors_[c] / naive.standard_errors_[c],
                    p_naive=naive.summary.loc[c, "p"],
                    p_cluster=robust.summary.loc[c, "p"],
                    frac_between=vb / (vb + vw),
                )
            )
    return pd.DataFrame(rows)


def mundlak_table(frames: dict[str, pd.DataFrame], *, min_runs: int = 2) -> pd.DataFrame:
    """Split each covariate into its within-well and between-well effect.

    Uses the Mundlak device: enter the covariate raw *and* add its well mean.  In that
    parameterisation the raw coefficient is the **within**-well effect (identified only by
    wells that changed the covariate between runs), and the coefficient on the well mean is
    ``beta_between - beta_within`` — so its p-value is a direct test of whether the two
    differ, i.e. whether the effect is confounded by which wells got what.

    ``beta_within`` is the actionable one: it says what happens when you change the
    covariate *in a given well*.  ``beta_between`` mixes that with well selection.

    All standard errors are cluster-robust on ``code`` — the naive ones are understated,
    per :func:`cluster_se_table`.
    """
    from lifelines import CoxPHFitter

    rows = []
    for label, d in frames.items():
        keep = COVARIATES + [CLOCK, EVENT_COL, "code"]
        sub = d.dropna(subset=keep).copy()
        for c in COVARIATES:
            if sub[c].std() > 0:
                sub[c] = (sub[c] - sub[c].mean()) / sub[c].std()

        sizes = sub.groupby("code")["code"].transform("size")
        for c in COVARIATES:
            mean_col = f"{c}__wellmean"
            work = sub.copy()
            work[mean_col] = work.groupby("code")[c].transform("mean")
            if work[mean_col].std() <= 0:
                continue
            cols = COVARIATES + [mean_col, CLOCK, EVENT_COL, "code"]
            try:
                fit = CoxPHFitter().fit(work[cols], CLOCK, EVENT_COL, cluster_col="code")
            except Exception:  # separation / singular design on a thin covariate
                continue

            b_within = fit.params_[c]
            delta = fit.params_[mean_col]  # beta_between - beta_within
            # Wells that can actually identify the within effect: repeat runs, covariate
            # actually varies, and at least one event to contribute to the partial likelihood.
            dev = (work[c] - work[mean_col]).abs()
            informative = work[(sizes >= min_runs) & (dev > 1e-9)].groupby("code")[EVENT_COL].sum()
            rows.append(
                dict(
                    field=label, covariate=c,
                    beta_within=b_within,
                    beta_between=b_within + delta,
                    se_within=fit.standard_errors_[c],
                    p_within=fit.summary.loc[c, "p"],
                    delta=delta,
                    p_delta=fit.summary.loc[mean_col, "p"],
                    n_wells_informative=int((informative > 0).sum()),
                    events_informative=int(informative.sum()),
                )
            )
    return pd.DataFrame(rows)


def within_well_stratified(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cox stratified by well — the pure fixed-effects within estimate, as a cross-check.

    Stratifying on ``code`` removes every time-constant well characteristic by
    construction, at the cost of discarding wells with no within-stratum event variation.
    It should agree with ``beta_within`` from :func:`mundlak_table`; where it does not, the
    Mundlak random-effects assumption is doing work and the stratified answer is the safer one.
    """
    from lifelines import CoxPHFitter

    rows = []
    for label, d in frames.items():
        keep = COVARIATES + [CLOCK, EVENT_COL, "code"]
        sub = d.dropna(subset=keep).copy()
        for c in COVARIATES:
            if sub[c].std() > 0:
                sub[c] = (sub[c] - sub[c].mean()) / sub[c].std()
        sub = sub[sub.groupby("code")["code"].transform("size") >= 2]
        # A stratum with no event contributes nothing to the partial likelihood.
        sub = sub[sub.groupby("code")[EVENT_COL].transform("sum") > 0]
        if sub.empty:
            continue
        try:
            fit = CoxPHFitter().fit(
                sub[keep], CLOCK, EVENT_COL, strata=["code"]
            )
        except Exception:
            continue
        for c in COVARIATES:
            rows.append(
                dict(
                    field=label, covariate=c,
                    beta_strat=fit.params_[c],
                    se_strat=fit.standard_errors_[c],
                    p_strat=fit.summary.loc[c, "p"],
                    n_runs=len(sub), n_wells=sub["code"].nunique(),
                    events=int(sub[EVENT_COL].sum()),
                )
            )
    return pd.DataFrame(rows)


#: Nested specifications for :func:`spec_ladder`, narrowest first.  Qnom, Kpod and Ql are
#: ~0.86 correlated raw and ~0.65 within-well, so a multivariable within-well fit inflates
#: whichever of them the optimiser leans on.  The bare fit is the trustworthy one.
SPEC_LADDER = [
    ["log_qnom"],
    ["log_qnom", "kpod_run"],
    COVARIATES,
]


def spec_ladder(frames: dict[str, pd.DataFrame], target: str = "log_qnom") -> pd.DataFrame:
    """Track one covariate's within-well estimate as the model widens.

    A coefficient that is stable across the ladder is identified; one that moves several
    fold is being carried by collinearity with whatever was just added, and only the bare
    entry should be quoted.
    """
    from lifelines import CoxPHFitter

    rows = []
    for label, d in frames.items():
        # One fixed sample for every rung — the complete case over the *widest* spec.
        # Dropping NAs per rung would let the sample move with the specification and the
        # drift could no longer be attributed to the added terms.
        widest = list(dict.fromkeys([c for spec in SPEC_LADDER for c in spec]))
        base = d.dropna(subset=widest + [CLOCK, EVENT_COL, "code"]).copy()
        for c in widest:
            if base[c].std() > 0:
                base[c] = (base[c] - base[c].mean()) / base[c].std()
        base = base[base.groupby("code")["code"].transform("size") >= 2]
        base = base[base.groupby("code")[EVENT_COL].transform("sum") > 0]
        if base.empty:
            continue
        for spec in SPEC_LADDER:
            if target not in spec:
                continue
            keep = list(dict.fromkeys(spec)) + [CLOCK, EVENT_COL, "code"]
            sub = base
            fit = CoxPHFitter().fit(sub[keep], CLOCK, EVENT_COL, strata=["code"])
            rows.append(
                dict(
                    field=label, target=target, n_terms=len(spec),
                    spec="+".join(spec),
                    beta=fit.params_[target], se=fit.standard_errors_[target],
                    p=fit.summary.loc[target, "p"],
                    n_runs=len(sub), n_wells=sub["code"].nunique(),
                    events=int(sub[EVENT_COL].sum()),
                )
            )
    return pd.DataFrame(rows)


def bare_within(
    frames: dict[str, pd.DataFrame],
    target: str = "log_qnom",
    *,
    restrict_to: list[str] | None = None,
) -> pd.DataFrame:
    """Single-covariate within-well estimate, one consistent sample rule per population.

    The cross-field comparison only means something if the sample is defined the same way
    everywhere.  ``restrict_to=None`` uses each population's complete case on ``target``
    alone (maximum power); passing the full covariate list instead restricts every
    population to the same complete case as the multivariable fits, which costs power but
    makes the numbers directly comparable to :func:`within_well_stratified`.

    ``beta_pooled`` is the same covariate with no well stratification on the same rows —
    the between- and within-well variation fused, i.e. what the shipped models see.
    """
    from lifelines import CoxPHFitter

    need = list(dict.fromkeys((restrict_to or []) + [target]))
    rows = []
    for label, d in frames.items():
        sub = d.dropna(subset=need + [CLOCK, EVENT_COL, "code"]).copy()
        if sub[target].std() > 0:
            sub[target] = (sub[target] - sub[target].mean()) / sub[target].std()
        cols = [target, CLOCK, EVENT_COL, "code"]
        pooled = CoxPHFitter().fit(sub[cols], CLOCK, EVENT_COL, cluster_col="code")
        keep = sub[(sub.groupby("code")["code"].transform("size") >= 2)
                   & (sub.groupby("code")[EVENT_COL].transform("sum") > 0)]
        if keep.empty:
            continue
        within = CoxPHFitter().fit(keep[cols], CLOCK, EVENT_COL, strata=["code"])
        rows.append(
            dict(
                population=label, target=target,
                beta_within=within.params_[target], se_within=within.standard_errors_[target],
                p_within=within.summary.loc[target, "p"],
                beta_pooled=pooled.params_[target], se_pooled=pooled.standard_errors_[target],
                p_pooled=pooled.summary.loc[target, "p"],
                n_runs_within=len(keep), n_wells_within=keep["code"].nunique(),
                events_within=int(keep[EVENT_COL].sum()),
                n_runs_pooled=len(sub), events_pooled=int(sub[EVENT_COL].sum()),
            )
        )
    return pd.DataFrame(rows)


def _rmst(t: np.ndarray, e: np.ndarray, tau: float = RMST_TAU) -> float:
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(t, e)
    s = km.survival_function_.iloc[:, 0]
    s = s[s.index <= tau]
    if not len(s):
        return float(tau)
    x = np.r_[0.0, s.index.values, tau]
    y = np.r_[1.0, s.values, s.values[-1]]
    return float(np.trapezoid(y[:-1], x[:-1]) + y[-2] * (tau - x[-2]))


def rmst_table(frames: dict[str, pd.DataFrame], *, n_boot: int = 400, seed: int = 11) -> pd.DataFrame:
    """RMST(0, tau) with run-resampled vs well-resampled bootstrap intervals."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, d in frames.items():
        point = _rmst(d[CLOCK].values, d[EVENT_COL].values)
        by_well = d.groupby("code").indices
        wells = np.array(list(by_well.keys()))
        naive, clust = [], []
        for _ in range(n_boot):
            s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
            naive.append(_rmst(s[CLOCK].values, s[EVENT_COL].values))
            pick = rng.choice(len(wells), len(wells), replace=True)
            c = d.iloc[np.concatenate([by_well[wells[i]] for i in pick])]
            clust.append(_rmst(c[CLOCK].values, c[EVENT_COL].values))
        se_n, se_c = np.std(naive, ddof=1), np.std(clust, ddof=1)
        rows.append(
            dict(
                field=label, n_runs=len(d), n_wells=len(wells),
                rmst_730=point, se_naive=se_n, se_cluster=se_c,
                se_inflation=se_c / se_n,
                lo_naive=point - 1.96 * se_n, hi_naive=point + 1.96 * se_n,
                lo_cluster=point - 1.96 * se_c, hi_cluster=point + 1.96 * se_c,
            )
        )
    return pd.DataFrame(rows)


def run(*, n_boot: int = 400, slug: str = "production_risk_well_clustering") -> dict[str, pd.DataFrame]:
    """Run all three diagnostics and write them under ``results/<slug>/<date>/tables``."""
    frames = {label: _prepare(f, s) for label, (f, s) in FIELDS.items()}
    out = {
        "frailty": frailty_table(frames),
        "cluster_se": cluster_se_table(frames),
        "mundlak": mundlak_table(frames),
        "stratified": within_well_stratified(frames),
        "spec_ladder": spec_ladder(frames),
        "bare_within": bare_within(frames),
        "bare_within_restricted": bare_within(frames, restrict_to=COVARIATES),
        "rmst": rmst_table(frames, n_boot=n_boot),
    }
    dest = results_dir(slug) / "tables"
    out["frailty"].to_csv(dest / "frailty_by_field.csv")
    out["cluster_se"].to_csv(dest / "cluster_robust_se.csv", index=False)
    out["mundlak"].to_csv(dest / "within_between_mundlak.csv", index=False)
    out["stratified"].to_csv(dest / "within_well_stratified.csv", index=False)
    out["spec_ladder"].to_csv(dest / "spec_ladder_qnom.csv", index=False)
    out["bare_within"].to_csv(dest / "bare_within_qnom.csv", index=False)
    out["bare_within_restricted"].to_csv(dest / "bare_within_qnom_restricted.csv", index=False)
    out["rmst"].to_csv(dest / "rmst_bootstrap.csv", index=False)
    return out
