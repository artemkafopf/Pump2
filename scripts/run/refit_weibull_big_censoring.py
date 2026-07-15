"""Workstream A — censoring-corrected, pull-reason-aware Weibull refit.

Rebuilds the latent-Weibull survival population so ONE fit reflects both of the
over-prediction levers found this session (see
``docs/notes/production_risk_joint_refit_prompt.md``):

  Lever 1 — the missing censored tail.  The shipped bundle fits on
  ``mart__weibull_input`` (V03 runs) and omits currently-running pumps.  The last
  run of each well in ``WellsArtificialLiftBig.xlsx`` IS the live pump → a
  legitimate right-censored observation (the ~500 open runs).

  Lever 2 — workovers mis-booked as failures.  Big AND the fresh Свод both stamp
  «Дата отказа» / Failure Flag = 1 on ГТМ/ППР workover pulls (pump serial evidence:
  the same physical pump returns only ~4% of the time after ГТМ).  So the event
  marker is the PULL REASON, not the presence of a fail date.

Population = fresh-Свод runs (canonical events) + Big-only strict-ESP runs absent
from Свод and V03 by ``(well_key, install_date ± 7 days)``.  Every run is one
survival unit (install → pull); events are classified by pull reason across the
WHOLE population.  Clock stays op-days (``ttf_mix`` / Наработка / ННО) — the
calendar mapping is Workstream B.

This is the FROZEN-BASELINE builder for Workstream A: it writes a candidate
bundle + audit tables + KM-overlay figures under
``results/esp_survival_big_censored_refit/<date>/`` but does NOT ship the bundle,
touch calibration factors, or rebuild the EXE.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.data.equipment_big import esp_vocab_audit, load_equipment_big
from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    latent_life_quantile,
    latent_survival,
)
from analysis.models.survival.weibull_model import fit_basic_weibull
from analysis.paths import (
    WAREHOUSE_DIR,
    resolve_equipment_big_path,
    resolve_prediction_workbook_path,
    results_dir,
)
from analysis.workflows.esp_survival import phase2_mixture
from analysis.workflows.esp_survival.data_mart import H2S_SOUR_THRESHOLD_MG_L, load_mart_df
from analysis.workflows.production_risk import config as PR_CONFIG
from analysis.workflows.production_risk import crosswalk

# ---------------------------------------------------------------------------
# Pull-reason classification (Lever 2) — applies to the WHOLE population.
# ---------------------------------------------------------------------------

# Casefolded pull reasons that mean the pump was pulled healthy for well work.
# A run with one of these is CENSORED at its operating time even when a fail date
# / Failure Flag = 1 is present (Big & Свод both mis-stamp these — verified this
# session: 453 Свод ГТМ rows carry Failure Flag = 1).
_WORKOVER_REASONS = {"гтм", "ппр"}

# Casefolded pull reasons that mean a genuine ESP failure → the run is an EVENT.
# Covers both the Big vocabulary («Снижение изоляции (R-0)», «Нет подачи, токи
# х.х.», «Нет звезды», «Клин ЭЦН») and the shorter Свод vocabulary («Снижение
# изоляции», «Клин»).
_GENUINE_FAILURE_REASONS = {
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


def _norm_reason(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return " ".join(str(v).split()).strip()


def naive_event(pull_reason: object, base_failure: bool) -> int:
    """Event marker BEFORE the workover demotion (Lever 2).

    A genuine-failure pull reason is always a failure; «Прочие»/blank fall back to
    a reliable per-source failure signal (``base_failure``): a named failed
    component «Отказавший узел» for Свод, a stamped fail date for Big.  Свод
    «Failure Flag» is deliberately NOT used — it is a sub-classification, not a
    failure/censor marker (FF = 0 rows routinely carry a named failed component).
    """
    r = _norm_reason(pull_reason).casefold()
    if r in _GENUINE_FAILURE_REASONS:
        return 1
    return 1 if base_failure else 0


def classify_event(pull_reason: object, base_failure: bool) -> int:
    """Event (1) vs censored (0) by PULL REASON, not by presence of a fail date.

    * {ГТМ, ППР}                     → 0 (workover, censored even if fail present)
    * everything else                → :func:`naive_event`
    """
    if _norm_reason(pull_reason).casefold() in _WORKOVER_REASONS:
        return 0
    return naive_event(pull_reason, base_failure)


def _field_from_well_key(well_key: object) -> str | None:
    if well_key is None or pd.isna(well_key):
        return None
    s = str(well_key).strip()
    prefix = s.split("_", 1)[0].upper() if "_" in s else s.upper()
    if prefix in PR_CONFIG.EXPLICIT_GLOBAL_FALLBACK:
        return None
    return PR_CONFIG.FIELD_PREFIX_MAP.get(prefix)


def _contractor_group(v: object) -> str:
    s = str(v or "")
    if "Борец" in s:
        return "brt"
    if "Шлюмберже" in s:
        return "slb"
    return "oth"


def _quantile(p: float, w1: float, b1: float, e1: float, b2: float, e2: float) -> float:
    model = TwoComponentLatentWeibullModel(
        weight_1=w1,
        component_1=WeibullParameters(beta=b1, eta=e1, label="C1"),
        component_2=WeibullParameters(beta=b2, eta=e2, label="C2"),
    )
    q = latent_life_quantile(p, model)
    return float(q) if np.isfinite(q) else float("nan")


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _source_v03_keys() -> pd.DataFrame:
    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    try:
        return (
            pd.read_sql(
                "SELECT well_key, install_date FROM raw__v03_runs",
                con,
                parse_dates=["install_date"],
            )
            .dropna()
            .drop_duplicates()
        )
    finally:
        con.close()


def _load_svod_runs(prediction_workbook_path: Path | None) -> pd.DataFrame:
    """Fresh-Свод runs as the canonical event side (op-day clock = Наработка)."""
    src = Path(prediction_workbook_path or resolve_prediction_workbook_path())
    sv = pd.read_excel(src, sheet_name="Свод")

    out = pd.DataFrame()
    out["well_key"] = sv["Скв."].map(crosswalk.norm_well).str.lower()
    out["install_date"] = pd.to_datetime(sv["Дата монтажа"], errors="coerce", dayfirst=True)
    out["end_date"] = pd.to_datetime(sv["Дата остановки"], errors="coerce", dayfirst=True)
    out["nno_days"] = pd.to_numeric(sv["Наработка (сут)"], errors="coerce")
    out["pull_reason"] = sv["Причина остановки"]
    out["failed_unit"] = sv["Отказавший узел"].map(_norm_reason)
    out["contractor"] = sv["Принадлежность"].astype(str)
    out["sour_raw"] = sv["Кислый/Некислый"].astype(str)
    out["source"] = "svod"

    out["field"] = out["well_key"].map(_field_from_well_key)
    out["contractor_group"] = out["contractor"].map(_contractor_group)
    out["h2s_class"] = np.where(
        (out["field"] == "Vt") & (out["sour_raw"].map(crosswalk.sour_group) == "sour"),
        "sour",
        "nonsour",
    )
    out["tte"] = out["nno_days"].astype(float)
    # base failure signal for Свод = a named failed component (not «нет»/blank).
    base = ~out["failed_unit"].str.casefold().isin({"", "нет"})
    # event_raw = naive (any failure signal, no workover demotion); event = gated.
    out["event_raw"] = [naive_event(pr, b) for pr, b in zip(out["pull_reason"], base)]
    out["event"] = [classify_event(pr, b) for pr, b in zip(out["pull_reason"], base)]
    return out


def _within_tol(target: pd.Timestamp, dates: list[pd.Timestamp], tol_days: int) -> bool:
    if pd.isna(target):
        return False
    for d in dates:
        if pd.notna(d) and abs((target - d).days) <= tol_days:
            return True
    return False


def _load_big_only(
    svod: pd.DataFrame,
    v03_keys: pd.DataFrame,
    mart: pd.DataFrame,
    equipment_big_path: Path | None,
    tol_days: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Big strict-ESP runs absent from Свод AND V03 by (well_key, install ± tol)."""
    big = load_equipment_big(equipment_big_path or resolve_equipment_big_path())
    big["install_date"] = pd.to_datetime(big["install_date"], errors="coerce")
    big["nno_days"] = pd.to_numeric(big["nno_days"], errors="coerce")

    cand = big[
        big["is_esp_strict"].fillna(False)
        & big["well_key"].notna()
        & big["install_date"].notna()
        & big["nno_days"].notna()
        & (big["nno_days"] > 0)
    ].copy()
    cand["field"] = cand["well_key"].map(_field_from_well_key)

    # ±tol dedup vs Свод and V03, per well.
    svod_dates = svod.dropna(subset=["well_key", "install_date"]).groupby("well_key")[
        "install_date"
    ].apply(list)
    v03 = v03_keys.copy()
    v03["well_key"] = v03["well_key"].astype(str).str.lower()
    v03_dates = v03.dropna(subset=["well_key", "install_date"]).groupby("well_key")[
        "install_date"
    ].apply(list)

    def _is_dup(row) -> bool:
        wk = row["well_key"]
        d = row["install_date"]
        return _within_tol(d, svod_dates.get(wk, []), tol_days) or _within_tol(
            d, v03_dates.get(wk, []), tol_days
        )

    cand["in_svod_or_v03"] = cand.apply(_is_dup, axis=1)
    big_only = cand[~cand["in_svod_or_v03"] & cand["field"].notna()].copy()

    # A3 — sour inheritance from mart by well_key then pad; unknown → nonsour.
    h2s = pd.to_numeric(mart["h2s_proxy_mg_l"], errors="coerce")
    by_well = mart.assign(_h=h2s).groupby("well_key")["_h"].median()
    by_pad = mart.assign(_h=h2s).groupby("pad")["_h"].median()
    proxy = big_only["well_key"].map(by_well)
    proxy = proxy.fillna(big_only["pad"].map(by_pad)) if "pad" in big_only.columns else proxy
    big_only["h2s_proxy_mg_l"] = proxy
    h2s_num = pd.to_numeric(big_only["h2s_proxy_mg_l"], errors="coerce")
    big_only["h2s_class"] = np.where(
        (big_only["field"] == "Vt") & (h2s_num > H2S_SOUR_THRESHOLD_MG_L),
        "sour",
        "nonsour",
    )
    big_only["contractor_group"] = big_only["contractor"].map(_contractor_group)
    big_only["tte"] = big_only["nno_days"].astype(float)
    big_only["end_date"] = pd.to_datetime(big_only.get("pull_date"), errors="coerce")
    # base failure signal for Big = a stamped fail date (Big stamps it on workovers
    # too, so the workover demotion in classify_event still applies).
    base = big_only["fail_date"].notna()
    big_only["event_raw"] = [naive_event(pr, b) for pr, b in zip(big_only["pull_reason"], base)]
    big_only["event"] = [classify_event(pr, b) for pr, b in zip(big_only["pull_reason"], base)]
    big_only["source"] = "big_only"

    audit = {
        "big_strict_esp_rows": int(big["is_esp_strict"].fillna(False).sum()),
        "big_strict_esp_candidate_nno_pos": int(len(cand)),
        "big_only_after_dedup": int((~cand["in_svod_or_v03"]).sum()),
        "big_only_prefix_mappable": int(len(big_only)),
        "big_only_events_genuine": int(big_only["event"].sum()),
        "big_only_censored": int((big_only["event"] == 0).sum()),
        "big_only_flip_event_to_censored": int(
            ((big_only["event_raw"] == 1) & (big_only["event"] == 0)).sum()
        ),
        "big_only_vt_sour": int(
            ((big_only["field"] == "Vt") & (big_only["h2s_class"] == "sour")).sum()
        ),
        "big_only_vt_missing_h2s_proxy": int(
            (big_only["field"] == "Vt").sum()
            - ((big_only["field"] == "Vt") & big_only["h2s_proxy_mg_l"].notna()).sum()
        ),
    }
    return big_only, audit


_KEEP_COLS = [
    "well_key", "install_date", "end_date", "tte", "pull_reason",
    "event", "event_raw", "field", "h2s_class",
    "contractor_group", "source",
]


def build_fit_population(
    *,
    prediction_workbook_path: Path | None = None,
    equipment_big_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (population, big_only, audit_dict)."""
    mart = load_mart_df(tte_col="ttf_mix")
    svod = _load_svod_runs(prediction_workbook_path)
    v03_keys = _source_v03_keys()
    big_only, big_audit = _load_big_only(
        svod, v03_keys, mart, equipment_big_path
    )

    pop = pd.concat([svod[_KEEP_COLS], big_only[_KEEP_COLS]], ignore_index=True)
    pop = pop[pop["tte"].notna() & (pop["tte"] > 0)].reset_index(drop=True)
    pop["stratum"] = pop["field"].astype(str) + "_" + pop["h2s_class"].astype(str)

    audit = {
        "svod_runs": int(len(svod)),
        "svod_runs_tte_pos": int((svod["tte"] > 0).sum()),
        "svod_events_pullreason": int(svod["event"].sum()),
        "svod_events_failflag_old": int(svod["event_raw"].sum()),
        "svod_flip_event_to_censored": int(
            ((svod["event_raw"] == 1) & (svod["event"] == 0)).sum()
        ),
        **big_audit,
        "population_runs": int(len(pop)),
        "population_events": int(pop["event"].sum()),
        "population_censored": int((pop["event"] == 0).sum()),
        "population_events_old_definition": int(pop["event_raw"].sum()),
        "population_total_flips_event_to_censored": int(
            ((pop["event_raw"] == 1) & (pop["event"] == 0)).sum()
        ),
    }
    return pop, big_only, audit


# ---------------------------------------------------------------------------
# Fitting (reuses the Phase-2 K=2 EM machinery + cascade schema)
# ---------------------------------------------------------------------------

def _registry_rows(
    df: pd.DataFrame, stratum_col: str, out_dir: Path, tag: str
) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    work["stratum"] = work[stratum_col]
    res = phase2_mixture.run(work, out_dir / "figures" / tag)
    res_rows = res["results"].set_index("stratum")
    rows: list[dict] = []
    for key, model in res["models"].items():
        g = work[work["stratum"] == key]
        rr = res_rows.loc[key]
        degenerate = bool(rr.get("degenerate", False))
        mix_wins_raw = rr.get("mixture_wins_aic", None)
        mix_wins = bool(mix_wins_raw) if mix_wins_raw is not None and not pd.isna(mix_wins_raw) else False
        # AIC (which penalises the extra mixture params) is the arbiter.  A decisive
        # mixture win is kept even when w1 > 0.75 trips the "degenerate" flag —
        # under heavy Big censoring the wear-out tail legitimately carries a small
        # weight, and forcing a single Weibull there badly under-fits the KM (e.g.
        # Vt_nonsour_Pooled: k2 b50≈376 matches KM 396 vs single-Weibull 574).
        if mix_wins:
            model_kind = "k2_high_w1" if degenerate else "k2"
        else:
            model_kind = "k1_degenerate" if degenerate else "k1_aic"

        if model_kind in ("k2", "k2_high_w1"):
            w1 = float(model.weight_1)
            b1 = float(model.component_1.beta)
            e1 = float(model.component_1.eta)
            b2 = float(model.component_2.beta)
            e2 = float(model.component_2.eta)
        else:
            sw = fit_basic_weibull(g["tte"].to_numpy(float), g["event"].to_numpy(int))
            b1 = b2 = float(sw["beta"])
            e1 = e2 = float(sw["eta"])
            w1 = 0.0

        rows.append({
            "stratum": key,
            "model_kind": model_kind,
            "w1": round(w1, 6),
            "beta1": round(b1, 6),
            "eta1": round(e1, 2),
            "beta2": round(b2, 6),
            "eta2": round(e2, 2),
            "b20": round(_quantile(0.20, w1, b1, e1, b2, e2), 1),
            "b50": round(_quantile(0.50, w1, b1, e1, b2, e2), 1),
            "b80": round(_quantile(0.80, w1, b1, e1, b2, e2), 1),
            "n_runs": int(len(g)),
            "n_failures": int(g["event"].sum()),
            "n_censored": int((1 - g["event"]).sum()),
            "n_big_censored": int((g["source"].eq("big_only") & g["event"].eq(0)).sum()),
            "delta_aic": rr.get("delta_aic"),
            "clock": "ttf_mix_svod_plus_big_nno",
        })
    return pd.DataFrame(rows).sort_values("stratum").reset_index(drop=True), res


def fit_registry(pop: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Fit Global + field_sour + field_sour_contractor strata (shipped schema)."""
    field_pop = pop[pop["field"].notna() & (pop["field"] != "None")].copy()
    field_pop["field_stratum"] = field_pop["field"] + "_" + field_pop["h2s_class"]
    field_pop["contractor_stratum"] = (
        field_pop["field"] + "_" + field_pop["h2s_class"] + "_" + field_pop["contractor_group"]
    )

    field_models, _ = _registry_rows(field_pop, "field_stratum", out_dir, "field")
    ctr_models, _ = _registry_rows(field_pop, "contractor_stratum", out_dir, "contractor")

    glob = pop.copy()
    glob["global_stratum"] = "Global"
    global_models, _ = _registry_rows(glob, "global_stratum", out_dir, "global")

    field_models["field"] = field_models["stratum"].str.rsplit("_", n=1).str[0]
    field_models["h2s_class"] = field_models["stratum"].str.rsplit("_", n=1).str[1]
    field_models["contractor_group"] = "Pooled"
    field_models["stratum"] = field_models["stratum"] + "_Pooled"

    ctr_parts = ctr_models["stratum"].str.rsplit("_", n=2, expand=True)
    ctr_models["field"] = ctr_parts[0]
    ctr_models["h2s_class"] = ctr_parts[1]
    ctr_models["contractor_group"] = ctr_parts[2]

    global_models["stratum"] = "Global_Pooled"
    global_models["field"] = "Global"
    global_models["h2s_class"] = "Pooled"
    global_models["contractor_group"] = "Pooled"

    refit = pd.concat([ctr_models, field_models, global_models], ignore_index=True)
    return refit


# ---------------------------------------------------------------------------
# A5 — Mc vintage variants on the corrected population
# ---------------------------------------------------------------------------

_MIN_AT_RISK = 10  # KM is uninformative once fewer than this many pumps remain at risk


def _reliable_horizon(kmf: KaplanMeierFitter, tte_max: float) -> float:
    """Largest time where the KM is still supported by ≥ _MIN_AT_RISK pumps.

    Beyond this the KM plateaus on censored observations, so a parametric tail that
    keeps descending shows a large but meaningless ΔS — heavy-censoring strata (Mc,
    76% censored) fail a naive [0, b80] gate purely from this artifact.
    """
    at = kmf.event_table["at_risk"]
    ok = at.index[at.to_numpy() >= _MIN_AT_RISK]
    return float(ok.max()) if len(ok) else float(tte_max)


def _km_stats(g: pd.DataFrame, model: TwoComponentLatentWeibullModel, b50: float, b80: float) -> dict:
    """KM-overlay acceptance stats for one stratum against its own population.

    Primary acceptance is max|ΔS| over the RELIABLE horizon [0, min(b80, t_reliable)];
    the full-[0, b80] value is reported alongside for transparency.
    """
    kmf = KaplanMeierFitter().fit(g["tte"].to_numpy(float), g["event"].to_numpy(int))
    t_grid = np.linspace(1.0, max(b80, 1.0), 400)
    model_s = np.asarray(latent_survival(t_grid, model), dtype=float)
    km_s = np.asarray(kmf.survival_function_at_times(t_grid).to_numpy(dtype=float))
    ds = np.abs(model_s - km_s)
    max_ds_full = float(np.nanmax(ds))
    horizon = min(b80, _reliable_horizon(kmf, float(g["tte"].max())))
    mask = t_grid <= horizon
    max_ds_rel = float(np.nanmax(ds[mask])) if mask.any() else max_ds_full
    med_ci = median_survival_times(kmf.confidence_interval_)
    lo, hi = float(med_ci.iloc[0, 0]), float(med_ci.iloc[0, 1])
    b50_in_ci = bool((np.isnan(lo) or b50 >= lo) and (np.isnan(hi) or b50 <= hi))
    return {
        "kmf": kmf,
        "km_median": float(kmf.median_survival_time_),
        "km_median_lo": lo,
        "km_median_hi": hi,
        "b50_in_km_ci": b50_in_ci,
        "reliable_horizon": round(horizon, 1),
        "max_abs_dS_reliable": round(max_ds_rel, 4),
        "max_abs_dS_on_0_b80": round(max_ds_full, 4),
        "accept_dS_lt_0p05": bool(max_ds_rel < 0.05),
    }


def _plot_overlay(kmf: KaplanMeierFitter, model, b50: float, b80: float, title: str, path: Path) -> None:
    t_grid = np.linspace(1.0, max(b80, 1.0), 400)
    model_s = np.asarray(latent_survival(t_grid, model), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["t", "s"]
    try:
        ci = kmf.confidence_interval_.reset_index()
        ci.columns = ["t", "lo", "hi"]
        ax.fill_between(ci["t"], ci["lo"], ci["hi"], alpha=0.15, color="steelblue")
    except Exception:
        pass
    ax.step(sf["t"], sf["s"], where="post", color="steelblue", lw=2, label="KM (corrected pop.)")
    ax.plot(t_grid, model_s, "k-", lw=2, label=f"Refit S(t), b50={b50:.0f}d")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.axvline(b50, color="crimson", ls="--", lw=1, label=f"model b50 (KM med={kmf.median_survival_time_:.0f})")
    ax.set_xlabel("Operating days (ttf_mix)")
    ax.set_ylabel("Survival")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mc_vintage_variants(pop: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Fit + KM-accept Mc on all-vintage / 2023+ / 2024+ install windows.

    Each variant is overlaid on the KM of ITS OWN install window (the population it
    represents), so A6 is a fair per-window test.  The chosen window is documented
    in the run summary; the deployed fleet is dominated by recent-vintage pumps.
    """
    mc = pop[(pop["field"] == "Mc") & (pop["h2s_class"] == "nonsour")].copy()
    fig_dir = out_dir / "figures"
    rows: list[dict] = []
    windows = {
        "all_vintage": pd.Timestamp.min,
        "install_2023plus": pd.Timestamp("2023-01-01"),
        "install_2024plus": pd.Timestamp("2024-01-01"),
    }
    for label, start in windows.items():
        sub = (mc if start == pd.Timestamp.min else mc[mc["install_date"] >= start]).copy()
        sub["stratum"] = "Mc_nonsour"
        if int(sub["event"].sum()) < 20:
            rows.append({"label": label, "n_runs": int(len(sub)),
                         "n_failures": int(sub["event"].sum()), "note": "too_few_failures"})
            continue
        reg, _ = _registry_rows(sub, "stratum", out_dir, f"mc_{label}")
        r = reg.iloc[0].to_dict()
        model = _model_from_row(pd.Series(r))
        stats = _km_stats(sub, model, float(r["b50"]), float(r["b80"]))
        _plot_overlay(stats["kmf"], model, float(r["b50"]), float(r["b80"]),
                      f"Mc {label}  max|ΔS|={stats['max_abs_dS_on_0_b80']}  b50∈CI={stats['b50_in_km_ci']}",
                      fig_dir / f"km_overlay_mc_{label}.png")
        r["label"] = label
        r["km_median"] = round(stats["km_median"], 1)
        r["km_median_lo"] = round(stats["km_median_lo"], 1) if np.isfinite(stats["km_median_lo"]) else None
        r["km_median_hi"] = round(stats["km_median_hi"], 1) if np.isfinite(stats["km_median_hi"]) else None
        r["b50_in_km_ci"] = stats["b50_in_km_ci"]
        r["reliable_horizon"] = stats["reliable_horizon"]
        r["max_abs_dS_reliable"] = stats["max_abs_dS_reliable"]
        r["max_abs_dS_on_0_b80"] = stats["max_abs_dS_on_0_b80"]
        r["accept_dS_lt_0p05"] = stats["accept_dS_lt_0p05"]
        r["note"] = ""
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A6 — KM-overlay acceptance + B50 shift vs shipped bundle
# ---------------------------------------------------------------------------

_FOCUS_STRATA = ["Mc_nonsour_Pooled", "Ya_nonsour_Pooled", "Vt_nonsour_Pooled", "Vt_sour_Pooled"]


def _model_from_row(row: pd.Series) -> TwoComponentLatentWeibullModel:
    return TwoComponentLatentWeibullModel(
        weight_1=float(row["w1"]),
        component_1=WeibullParameters(beta=float(row["beta1"]), eta=float(row["eta1"]), label="C1"),
        component_2=WeibullParameters(beta=float(row["beta2"]), eta=float(row["eta2"]), label="C2"),
    )


def km_acceptance(pop: pd.DataFrame, refit: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    reg = refit.set_index("stratum")
    for stratum in _FOCUS_STRATA:
        parts = stratum.split("_")
        field, h2s = parts[0], parts[1]
        g = pop[(pop["field"] == field) & (pop["h2s_class"] == h2s)]
        if stratum not in reg.index or len(g) < 10 or g["event"].sum() < 5:
            rows.append({"stratum": stratum, "status": "skipped_insufficient_data",
                         "n_runs": int(len(g)), "n_failures": int(g["event"].sum())})
            continue
        model = _model_from_row(reg.loc[stratum])
        b50 = float(reg.loc[stratum, "b50"])
        b80 = float(reg.loc[stratum, "b80"])
        stats = _km_stats(g, model, b50, b80)
        _plot_overlay(stats["kmf"], model, b50, b80,
                      f"{stratum}  max|ΔS|={stats['max_abs_dS_on_0_b80']}  b50∈CI={stats['b50_in_km_ci']}",
                      fig_dir / f"km_overlay_{stratum}.png")
        rows.append({
            "stratum": stratum,
            "status": "ok",
            "n_runs": int(len(g)),
            "n_failures": int(g["event"].sum()),
            "n_censored": int((g["event"] == 0).sum()),
            "model_b50": round(b50, 1),
            "km_median": round(stats["km_median"], 1),
            "km_median_lo": round(stats["km_median_lo"], 1) if np.isfinite(stats["km_median_lo"]) else None,
            "km_median_hi": round(stats["km_median_hi"], 1) if np.isfinite(stats["km_median_hi"]) else None,
            "b50_in_km_ci": stats["b50_in_km_ci"],
            "reliable_horizon": stats["reliable_horizon"],
            "max_abs_dS_reliable": stats["max_abs_dS_reliable"],
            "max_abs_dS_on_0_b80": stats["max_abs_dS_on_0_b80"],
            "accept_dS_lt_0p05": stats["accept_dS_lt_0p05"],
        })
    return pd.DataFrame(rows)


_MC_WINDOW_PREFERENCE = ["install_2024plus", "install_2023plus", "all_vintage"]


def _apply_mc_choice(refit: pd.DataFrame, mc: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Replace the naive all-vintage Mc_nonsour_Pooled row with the chosen vintage.

    Choice rule (A6-only; D1 fact/model is out of Workstream-A scope): among windows
    that PASS A6 (max|ΔS| < 0.05 AND b50 within the KM median CI) prefer the most
    recent (the deployed fleet is recent-vintage); if none passes, keep the tightest
    ΔS window and let the summary flag Mc as not-yet-accepted.
    """
    cand = mc[mc.get("note", "").eq("") & mc["b50"].notna()].copy() if "b50" in mc.columns else mc.iloc[0:0]
    if cand.empty:
        return refit, "none"
    passing = cand[cand.get("accept_dS_lt_0p05", False) & cand.get("b50_in_km_ci", False)]
    pool = passing if not passing.empty else cand
    order = {w: i for i, w in enumerate(_MC_WINDOW_PREFERENCE)}
    if not passing.empty:
        pool = pool.sort_values("label", key=lambda s: s.map(lambda x: order.get(x, 99)))
        chosen = pool.iloc[0]
    else:
        chosen = pool.sort_values("max_abs_dS_reliable").iloc[0]

    refit = refit.copy()
    mask = refit["stratum"].isin(["Mc_nonsour_Pooled", "Mc_nonsour_brt"])
    refit = refit[~mask].reset_index(drop=True)
    row = {
        "stratum": "Mc_nonsour_Pooled", "field": "Mc", "h2s_class": "nonsour",
        "contractor_group": "Pooled", "model_kind": chosen["model_kind"],
        "w1": chosen["w1"], "beta1": chosen["beta1"], "eta1": chosen["eta1"],
        "beta2": chosen["beta2"], "eta2": chosen["eta2"],
        "b20": chosen["b20"], "b50": chosen["b50"], "b80": chosen["b80"],
        "n_runs": chosen["n_runs"], "n_failures": chosen["n_failures"],
        "n_censored": chosen.get("n_censored"), "n_big_censored": chosen.get("n_big_censored"),
        "delta_aic": chosen.get("delta_aic"),
        "clock": "ttf_mix_svod_plus_big_nno", "mc_window": chosen["label"],
    }
    refit = pd.concat([refit, pd.DataFrame([row])], ignore_index=True)
    return refit, str(chosen["label"])


def _b50_shift(refit: pd.DataFrame, out_dir: Path) -> pd.DataFrame | None:
    old_path = PR_CONFIG.model_registry_path(PR_CONFIG.BUNDLE_DATE)
    if not old_path.exists():
        return None
    old = pd.read_csv(old_path, encoding="utf-8-sig")[["stratum", "model_kind", "b50", "n_runs", "n_failures"]]
    cmp = old.merge(refit[["stratum", "model_kind", "b50", "n_runs", "n_failures"]],
                    on="stratum", how="outer", suffixes=("_old", "_new"))
    cmp["b50_delta"] = pd.to_numeric(cmp["b50_new"], errors="coerce") - pd.to_numeric(cmp["b50_old"], errors="coerce")
    cmp["b50_ratio"] = pd.to_numeric(cmp["b50_new"], errors="coerce") / pd.to_numeric(cmp["b50_old"], errors="coerce")
    return cmp.sort_values("b50_ratio", ascending=False, na_position="last").reset_index(drop=True)


def _event_censor_by_stratum(pop: pd.DataFrame) -> pd.DataFrame:
    d = pop.assign(
        _censored=(pop["event"] == 0).astype(int),
        _flip=((pop["event_raw"] == 1) & (pop["event"] == 0)).astype(int),
        _big=(pop["source"] == "big_only").astype(int),
    )
    out = d.groupby("stratum").agg(
        n_runs=("event", "size"),
        events_new=("event", "sum"),
        events_old=("event_raw", "sum"),
        censored_new=("_censored", "sum"),
        flips_event_to_censored=("_flip", "sum"),
        n_big_only=("_big", "sum"),
    )
    return out.reset_index().sort_values("n_runs", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equipment-big", type=Path, default=None)
    ap.add_argument("--prediction-workbook", type=Path, default=None)
    ap.add_argument("--out-date", default=date.today().isoformat())
    ap.add_argument("--num-starts", type=int, default=8)
    args = ap.parse_args()

    phase2_mixture.NUM_STARTS = int(args.num_starts)
    out = results_dir("esp_survival_big_censored_refit", args.out_date)
    out.mkdir(parents=True, exist_ok=True)

    pop, big_only, audit = build_fit_population(
        prediction_workbook_path=args.prediction_workbook,
        equipment_big_path=args.equipment_big,
    )
    pop.to_csv(out / "fit_population.csv", index=False, encoding="utf-8-sig")
    big_only.to_csv(out / "big_only_runs.csv", index=False, encoding="utf-8-sig")

    # A0 vocabulary audit for user confirmation.
    big_all = load_equipment_big(args.equipment_big or resolve_equipment_big_path())
    esp_vocab_audit(big_all).to_csv(out / "esp_vocab_audit.csv", index=False, encoding="utf-8-sig")

    ec = _event_censor_by_stratum(pop)
    ec.to_csv(out / "event_censor_by_stratum.csv", index=False, encoding="utf-8-sig")

    refit = fit_registry(pop, out)

    # A5 — Mc vintage variants; wire the chosen window into the candidate Mc row.
    mc = mc_vintage_variants(pop, out)
    refit, mc_choice = _apply_mc_choice(refit, mc)
    refit.to_csv(out / "esp_models_big_censored_refit.csv", index=False, encoding="utf-8-sig")
    mc.to_csv(out / "mc_vintage_variants.csv", index=False, encoding="utf-8-sig")

    shift = _b50_shift(refit, out)
    if shift is not None:
        shift.to_csv(out / "esp_models_old_vs_refit_b50.csv", index=False, encoding="utf-8-sig")

    accept = km_acceptance(pop, refit, out)
    accept.to_csv(out / "km_acceptance.csv", index=False, encoding="utf-8-sig")

    pd.Series(audit).to_frame("value").to_csv(out / "source_audit.csv", encoding="utf-8-sig")

    print("\n=== source audit ===")
    print(pd.Series(audit).to_string())
    print("\n=== event/censor/flip by stratum ===")
    print(ec.to_string(index=False))
    print("\n=== Mc vintage variants ===")
    print(mc.to_string(index=False))
    print(f"\nMc chosen window for candidate Mc_nonsour_Pooled: {mc_choice}")
    print("\n=== KM acceptance (Mc/Ya/Vt) ===")
    print(accept.to_string(index=False))
    if shift is not None:
        print("\n=== B50 old→new (corrected) ===")
        print(shift[["stratum", "b50_old", "b50_new", "b50_ratio", "n_failures_old", "n_failures_new"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
