"""ESP run population on the Свод clock, reconstructable at any analysis date.

One row per installed pump run (`install → pull`), which is how the survival fit
treats them: after a ГТМ the same pump comes back only ~4% of the time, so each
install is a fresh unit starting at age 0.

The population can be rebuilt "as of" a historical date T — every run installed
before T, censored at T if it had not been pulled yet.  That makes it possible to
ask what a well-observed stratum's model would have looked like back when it had
only as much data as a young field has today.

Fixes carried over from the audit of the shipped refit:
  * day-0 failures are KEPT (clipped to 0.5 d), not dropped by a ``tte > 0`` filter;
  * Свод is canonical — no within-source dedup (runs 2 days apart are real);
  * pull reason, not the presence of a fail date, decides event vs censor.

The population is **Свод + the Big runs Свод does not already have** (see
``load_big_runs``).  Both halves of that are load-bearing: Big-only *closed* runs
carry failures Свод never recorded (619 for Ya — a third of its events), and Big
rows matching a Свод run must be dropped or they double-count as free censored
exposure.  Getting either wrong biases the fitted hazard **down**, which shows up
as a model that under-predicts the plan.  Cross-check any new field's run/event
counts against ``esp_models.csv`` before trusting a fit.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data.equipment_big import load_equipment_big
from analysis.paths import WAREHOUSE_DIR
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk

WORKOVER_REASONS = {"гтм", "ппр"}
GENUINE_FAILURE_REASONS = {
    "снижение изоляции",
    "снижение изоляции (r-0)",
    "r-0",
    "отсутствие подачи",
    "нет подачи, токи х.х.",
    "нет подачи токи х.х.",
    "нет звезды",
    "клин",
    "клин эцн",
    "снижение подачи",
}
DAY_ZERO_TTE = 0.5


def _norm(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split()).strip()


def contractor_group(value: object) -> str:
    text = str(value or "")
    if "Борец" in text:
        return "brt"
    if "Шлюмберже" in text:
        return "slb"
    return "oth"


def classify(pull_reason: object, has_failed_unit: bool, gtm_is_failure: bool) -> int:
    """1 = failure event, 0 = censored.

    ``gtm_is_failure`` switches the estimand: False (default) censors ГТМ/ППР and
    yields the cause-specific failure hazard; True counts every pull as an event
    and yields all-cause pump replacement.
    """
    reason = _norm(pull_reason).casefold()
    if reason in WORKOVER_REASONS:
        return 1 if gtm_is_failure else 0
    if reason in GENUINE_FAILURE_REASONS:
        return 1
    return 1 if has_failed_unit else 0


def load_svod_runs(prediction_workbook_path: Path | None = None) -> pd.DataFrame:
    src = Path(prediction_workbook_path or crosswalk.resolve_prediction_workbook_path())
    sv = pd.read_excel(src, sheet_name="Свод")

    out = pd.DataFrame()
    out["code"] = sv["Скв."].map(crosswalk.norm_well)
    out["install"] = pd.to_datetime(sv["Дата монтажа"], errors="coerce")
    out["end"] = pd.to_datetime(sv["Дата остановки"], errors="coerce")
    out["tte_full"] = pd.to_numeric(sv["Наработка (сут)"], errors="coerce")
    out["pull_reason"] = sv["Причина остановки"]
    out["contractor_group"] = sv["Принадлежность"].map(contractor_group)
    has_unit = ~sv["Отказавший узел"].map(_norm).str.casefold().isin({"", "нет"})
    out["has_failed_unit"] = has_unit
    out["field"] = out["code"].map(_model_field)
    out["h2s_class"] = np.where(
        (out["field"] == "Vt") & (sv["Кислый/Некислый"].map(crosswalk.sour_group) == "sour"),
        "sour",
        "nonsour",
    )
    out["source"] = "svod"
    out = out.dropna(subset=["install"])

    # «Флаг отказа» = -1 marks runs with an unresolved outcome (mostly «Прочие») —
    # neither a clean event nor a clean censoring.  Excluded (user decision 2026-07-22).
    if "Флаг отказа" in sv.columns:
        flag = pd.to_numeric(sv["Флаг отказа"], errors="coerce").reindex(out.index)
        out = out[flag != -1]

    # Weird-date rows are excluded, not repaired.  Long демонтаж−остановка gaps are
    # NORMAL (the well waits for a crew); what is impossible is a stop before the
    # install or any date beyond SVOD_FUTURE_BOUND (e.g. Vt_7805, демонтаж 2026-11-04).
    demo = pd.to_datetime(sv["Дата демонтажа"], errors="coerce").reindex(out.index)
    future = pd.Timestamp(C.SVOD_FUTURE_BOUND)
    weird = (
        (out["end"].notna() & (out["end"] < out["install"]))
        | (out["install"] > future)
        | (out["end"] > future)
        | (demo > future)
    )
    out = out[~weird]

    # The reworked Свод carries OPEN (censored) rows: stop date empty, «Наработка»
    # empty.  Their techregime snapshot is C.SVOD_OPEN_ASOF, so the running pump's
    # age is frozen there (user decision) — ``build`` then censors at
    # min(that, as_of age) exactly as for any censored run.  A CLOSED row without
    # «Наработка» has no usable clock.
    open_rows = out["end"].isna()
    asof_age = (pd.Timestamp(C.SVOD_OPEN_ASOF) - out["install"]).dt.days.clip(lower=0).astype(float)
    out.loc[open_rows & out["tte_full"].isna(), "tte_full"] = asof_age
    closed_without_clock = (~open_rows) & out["tte_full"].isna()
    return out[~closed_without_clock]


def _model_field(code: object) -> str | None:
    text = str(code or "")
    prefix = text.split("_", 1)[0].upper() if "_" in text else text.upper()
    if prefix in C.EXPLICIT_GLOBAL_FALLBACK:
        return None
    return C.FIELD_PREFIX_MAP.get(prefix)


def _v03_install_dates() -> pd.Series:
    """Well → V03 install dates, the second dedup key the shipped refit uses."""
    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    try:
        v03 = pd.read_sql(
            "SELECT well_key, install_date FROM raw__v03_runs",
            con, parse_dates=["install_date"],
        ).dropna().drop_duplicates()
    finally:
        con.close()
    v03["code"] = v03["well_key"].map(crosswalk.norm_well)
    return v03.groupby("code")["install_date"].apply(list)


def _within_tol(target: pd.Timestamp, dates: list, tol_days: int) -> bool:
    return any(pd.notna(d) and abs((target - d).days) <= tol_days for d in dates)


def load_big_runs(
    svod: pd.DataFrame,
    tol_days: int = 7,
    equipment_big_path: Path | None = None,
) -> pd.DataFrame:
    """Big strict-ESP runs that Свод does NOT already carry — CLOSED only.

    Свод is canonical, so a Big row within ``tol_days`` of a Свод install for the
    same well is the same physical run and must not be counted twice.  V03 is the
    second dedup key, matching the shipped refit's population.

    Closed Big-only runs carry real failures Свод never saw — 619 of them for Ya,
    which is why a Свод-only population under-states Ya's hazard by a third of its
    events (Mc has 5, Vt 26, so the gap hid until Ya was fitted).

    Open Big rows are NOT taken (user decision 2026-07-22): the reworked Свод lists
    the running fleet from the techregime snapshot, and the Big open remainder was
    dominated by phantoms — brine bores folded onto oil codes (VT_2704), stale
    never-closed rows (YA_601), and water-field YAW_/BTW_ codes.
    """
    big = load_equipment_big(equipment_big_path) if equipment_big_path else load_equipment_big()
    big = big[big["is_esp_strict"] == True].copy()  # noqa: E712 — pandas mask
    big["code"] = big["well_key"].map(crosswalk.norm_well)
    big["install"] = pd.to_datetime(big["install_date"], errors="coerce")
    big["pull"] = pd.to_datetime(big["pull_date"], errors="coerce")
    big["nno"] = pd.to_numeric(big["nno_days"], errors="coerce")
    big = big.dropna(subset=["code", "install"])

    svod_dates = (
        svod.dropna(subset=["code", "install"]).groupby("code")["install"].apply(list)
    )
    v03_dates = _v03_install_dates()
    keep = [
        not (
            _within_tol(d, svod_dates.get(c, []), tol_days)
            or _within_tol(d, v03_dates.get(c, []), tol_days)
        )
        for c, d in zip(big["code"], big["install"])
    ]
    big = big[keep]
    # Closed only (see docstring); a closed run without a recorded operating time
    # has no usable clock.
    big = big[big["pull"].notna() & big["nno"].notna() & (big["nno"] >= 0)]

    closed = big["pull"].notna()
    out = pd.DataFrame()
    out["code"] = big["code"]
    out["install"] = big["install"]
    out["end"] = big["pull"]
    out["tte_full"] = np.where(closed, big["nno"], np.nan)
    out["pull_reason"] = big["pull_reason"]
    out["contractor_group"] = big["contractor"].map(contractor_group)
    # Big's own failure signal is a stamped fail date (it stamps one on workovers
    # too, so ``classify``'s workover demotion still applies).
    out["has_failed_unit"] = big["fail_date"].notna()
    out["field"] = out["code"].map(_model_field)
    # Placeholder — Big carries no «Кислый/Некислый», so the real class is
    # inherited from the well's Свод history in ``build``.
    out["h2s_class"] = "nonsour"
    out["source"] = np.where(closed, "big_closed", "big_open")
    return out.reset_index(drop=True)


def well_sour_map(svod: pd.DataFrame) -> pd.Series:
    """Well-level sour class from Свод.

    Sour is a property of the well's fluid, not of an individual run, so every
    run of a well shares it.
    """
    return svod.groupby("code")["h2s_class"].agg(
        lambda s: "sour" if (s == "sour").any() else "nonsour"
    )


def build(
    as_of: date | pd.Timestamp,
    gtm_is_failure: bool = False,
    svod: pd.DataFrame | None = None,
    big_runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Population as it stood on ``as_of``: runs installed by then, censored at it."""
    t = pd.Timestamp(as_of)
    svod = load_svod_runs() if svod is None else svod
    big_runs = load_big_runs(svod) if big_runs is None else big_runs

    frames = []
    for src in (svod, big_runs):
        g = src[src["install"] <= t].copy()
        age_at_t = (t - g["install"]).dt.days.astype(float)
        # Ended before the analysis date -> the real outcome; otherwise still
        # running at T -> right-censored at its age then.
        ended = g["end"].notna() & (g["end"] <= t)
        g["tte"] = np.where(ended, g["tte_full"], np.minimum(g["tte_full"].fillna(np.inf), age_at_t))
        g["event"] = [
            classify(r, u, gtm_is_failure) if e else 0
            for r, u, e in zip(g["pull_reason"], g["has_failed_unit"], ended)
        ]
        frames.append(g)

    pop = pd.concat(frames, ignore_index=True)
    # Big has no «Кислый/Некислый», so its runs arrive stamped "nonsour". Sour is a
    # property of the well's fluid, not of a run, so inherit it from that well's Свод
    # history — otherwise a Vt sour well's RUNNING pump lands in Vt_nonsour.
    sour = well_sour_map(svod)
    inherited = pop["code"].map(sour)
    # Big always inherits (it carries no «Кислый/Некислый»).  Under the v3.2 flag the
    # roll-up also relabels Свод runs — recovering running pumps that never got the flag
    # because it is written only from the failure/workover DB (see config note).
    eligible = (
        inherited.notna()
        if C.SOUR_WELL_LEVEL_ALL_RUNS
        else pop["source"].isin(("big_open", "big_closed")) & inherited.notna()
    )
    pop["h2s_class"] = np.where(eligible, inherited, pop["h2s_class"])
    pop["tte"] = pop["tte"].clip(lower=DAY_ZERO_TTE)  # keep day-0 startup failures
    pop = pop[np.isfinite(pop["tte"])]
    pop = drop_stale_open_runs(pop)
    pop["stratum"] = pop["field"].astype(str) + "_" + pop["h2s_class"] + "_" + pop["contractor_group"]
    return pop.reset_index(drop=True)


def drop_stale_open_runs(pop: pd.DataFrame, tol_days: int = 7) -> pd.DataFrame:
    """Drop OPEN runs contradicted by later activity on the same well.

    A run cannot still be running if the same well has a newer install, or a closed
    run that ended after this run began — the open row is a stale record the source
    never closed.  Known phantoms this catches: YA_601 (Big open row from 2016 with a
    closed run ending 2017) and VT_2704 (an open oil-coded row shadowed by later
    runs of the brine bore ``vt_2704рс`` that ``norm_well`` folds onto the same code).
    """
    is_open = pop["end"].isna()
    if not is_open.any():
        return pop
    tol = pd.Timedelta(days=tol_days)
    last_install = pop.groupby("code")["install"].max()
    last_closed_end = pop.loc[~is_open].groupby("code")["end"].max()
    stale = is_open & (
        (pop["install"] < pop["code"].map(last_install) - tol)
        | (pop["install"] < pop["code"].map(last_closed_end) - tol)
    )
    return pop[~stale]


def add_time_scales(pop: pd.DataFrame, as_of: date | pd.Timestamp) -> pd.DataFrame:
    """Attach the canonical time scales.  There are exactly **two model clocks**:

      ``t_cal``  — calendar.  ``pull - install`` (or ``as_of - install`` while running).
                   Two dates subtracted: always available, never reported, never
                   imputed.  This is the ``cal`` family of fits.
      ``t_op``   — operating time.  Attached by :func:`esp_optime.measure`, together
                   with ``t_mix`` (the servable version — see there).  This is the
                   ``op`` family.

    and one **reporting-only** figure:

      ``t_nno``  — «Наработка (сут)» / Big ``nno_days`` / ТР ННО, exactly as reported,
                   ``NaN`` where nobody reported one.  The business reads ННО, so it
                   must be carried, but it is NOT a fit clock: it is calendar-like
                   (median t_nno/t_cal = 0.95-0.99, versus 0.85-0.89 for true op-time)
                   yet inconsistently net of downtime, and it does not exist for
                   running pumps at all.

    Why ``t_nno`` is not a clock — the defect this replaces
    ------------------------------------------------------
    The legacy ``tte`` column silently carries *two* clocks: closed runs get ННО, but
    open runs fall through ``tte_full.fillna(inf)`` to the calendar age.  Measured:
    ``tte == t_cal`` for 60 of 61 open Mc runs, 138 of 138 open Vt runs, 166 of 167
    open Ya runs.  So **events are timed on ННО while censorings are timed on
    calendar** — and since ННО < calendar, censored rows get the longer clock, which
    biases fitted survival upward.  ``t_cal`` and ``t_op``/``t_mix`` each apply one
    clock to events and censorings alike, which is what fixes it.

    ``tte`` is kept untouched for the shipped bundle's consumers; new fits take
    ``t_cal`` or ``t_mix``.
    """
    t = pd.Timestamp(as_of)
    out = pop.copy()
    ended = out["end"].notna() & (out["end"] <= t)
    end_eff = out["end"].where(ended, t)
    out["t_cal"] = (end_eff - out["install"]).dt.days.astype(float).clip(lower=DAY_ZERO_TTE)
    # Reported ННО, with no calendar substitution: absent stays absent.
    out["t_nno"] = pd.to_numeric(out["tte_full"], errors="coerce")
    # Impossible values are a data-quality signal, not a measurement (19 Vt runs
    # report ННО exceeding the elapsed calendar span, one by 362 days).
    out["t_nno_valid"] = out["t_nno"].notna() & (out["t_nno"] <= out["t_cal"] + 1.0)
    return out


def add_calendar_tte(pop: pd.DataFrame, as_of: date | pd.Timestamp) -> pd.DataFrame:
    """Deprecated alias of :func:`add_time_scales` (kept for the Vt grid scripts)."""
    out = add_time_scales(pop, as_of)
    out["tte_cal"] = out["t_cal"]
    return out


def add_entry_age(pop: pd.DataFrame, window_start: date | pd.Timestamp) -> pd.DataFrame:
    """Age each run had reached at ``window_start`` — the left-truncation entry.

    Runs installed inside the window enter at 0; runs already turning enter at
    their age then, contributing only their in-window exposure.  Runs that had
    already ended before the window are dropped.
    """
    start = pd.Timestamp(window_start)
    out = pop.copy()
    out["entry"] = ((start - out["install"]).dt.days.clip(lower=0)).astype(float)
    return out[out["tte"] > out["entry"]].reset_index(drop=True)


EQUIPMENT_COLS = (
    "pump_exec_group",
    "pump_exec_lch",
    "pump_corr_class",
    "max_corr_class",
    "any_corr_protection",
    "type_corr_resistant",
    "setting_depth_m",
)


def attach_equipment(
    pop: pd.DataFrame,
    tolerance_days: int = 10,
    equipment_big_path: Path | None = None,
) -> pd.DataFrame:
    """Join Big's per-run equipment attributes onto the population.

    Свод and Big describe the same physical runs but agree only approximately on
    the install date, so runs are matched per well to the nearest Big install
    within ``tolerance_days``.  Unmatched runs keep NaN — they must be reported as
    coverage, never silently dropped, since Big coverage is not random.
    """
    big = load_equipment_big(equipment_big_path) if equipment_big_path else load_equipment_big()
    big = big[big["is_esp_strict"] == True].copy()  # noqa: E712 — pandas mask
    big["code"] = big["well_key"].map(crosswalk.norm_well)
    big["install"] = pd.to_datetime(big["install_date"], errors="coerce")
    cols = [c for c in EQUIPMENT_COLS if c in big.columns]
    big = big.dropna(subset=["install", "code"])[["code", "install", *cols]]

    left = pop.copy()
    left["_row"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values("install"),
        big.sort_values("install"),
        on="install",
        by="code",
        tolerance=pd.Timedelta(days=tolerance_days),
        direction="nearest",
    )
    return merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def mc_cohort_mask(pop: pd.DataFrame) -> pd.Series:
    """Rows passing the Мирнинский cohort rule — the mask behind :func:`apply_mc_cohort`.

    Exposed separately so a caller that has to *report* the filter (a selection funnel)
    counts the very same rows the filter drops, instead of re-deriving the rule.
    """
    if "install" not in pop.columns:
        raise KeyError("apply_mc_cohort needs an `install` column")
    start = pd.Timestamp(C.MC_INSTALL_COHORT_START)
    is_mc = pop["field"].isin(C.MC_COHORT_FIELDS)
    return ~is_mc | (pop["install"] >= start)


def apply_mc_cohort(pop: pd.DataFrame) -> pd.DataFrame:
    """Restrict Мирнинский to installs on/after ``C.MC_INSTALL_COHORT_START``.

    A cohort filter on the install date — NOT left truncation of exposure, which keeps
    pre-2024 runs' later exposure and hides that recent runs are shorter.  Rows of other
    fields pass through untouched.
    """
    return pop[mc_cohort_mask(pop)].copy()


def select(
    pop: pd.DataFrame,
    field: str,
    h2s: str = "nonsour",
    contractor: str | None = None,
    mc_cohort: bool = True,
) -> pd.DataFrame:
    """Pull one stratum.

    ``mc_cohort`` (default True) applies the Мирнинский installs-2024+ rule — every Mc
    statistic is built from that cohort.  Pass False only to reproduce an all-history
    number for contrast, and label it as such.
    """
    g = pop[(pop["field"] == field) & (pop["h2s_class"] == h2s)]
    if contractor is not None:
        g = g[g["contractor_group"] == contractor]
    if mc_cohort:
        g = apply_mc_cohort(g)
    return g.copy()
