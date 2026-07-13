"""Refit the Mc/MR survival stratum from Свод + WellsArtificialLiftBig.

This is intentionally narrow: it only replaces the Mc nonsour registry rows in
the production-risk bundle, leaving all other strata unchanged.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data.equipment_big import load_equipment_big
from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    latent_life_quantile,
)
from analysis.models.survival.bootstrap_ci import bootstrap_stratum_ci
from analysis.models.survival.weibull_em import fit_latent_weibull_em
from analysis.models.survival.weibull_model import fit_basic_weibull
from analysis.paths import resolve_equipment_big_path, resolve_prediction_workbook_path, results_dir
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk

# Only the two Мирнинский pads (Mc, Mr) — NE/Непский is geologically distinct and
# is no longer force-mapped to Mc (see config.EXPLICIT_GLOBAL_FALLBACK).
MC_PREFIXES = {"MC", "MR"}
# New-vintage regime.  KM on Big Mc+Mr runs shows install-2024 and install-2025+
# cohorts share a ~240-day median TTF, distinct from the ~480-day ≤2023 cohort;
# the current fleet is dominated by that new vintage, so the shipped row is fit on
# installs from RECENCY_START.  The all-history fit is still reported for contrast.
# 2024+ (not 2025+) is used deliberately: the two new-vintage cohorts fit to the
# same b50 (~216 vs ~224 op-days), so 2024+ is preferred for its larger sample
# (160 vs 95 failures) and tighter b50 CI.  The residual Мирнинский under-count is
# an idle/workover-month failure pattern, not a fit-window artifact — a faster
# window does not close it.
RECENCY_START = pd.Timestamp("2024-01-01")
DEFAULT_CUTOFF = pd.Timestamp("2026-06-30")
SOURCE_BUNDLE_DATE = "2026-07-08"
TARGET_BUNDLE_DATE = "2026-07-13"
# Both old Mc rows are removed; only Mc_nonsour_Pooled is re-emitted.  Contractor
# variants (brt/slb/oth) then cascade to the pooled fit via StrataModel.resolve —
# Mc data is too thin to justify a separate per-contractor Weibull.
TARGET_STRATA = ("Mc_nonsour_Pooled", "Mc_nonsour_brt")


@dataclass(frozen=True)
class RefitResult:
    bundle_dir: Path
    report_dir: Path
    dataset: pd.DataFrame
    fit_rows: pd.DataFrame


def _prefix(code: object) -> str:
    s = str(code or "").strip().upper()
    return s.split("_", 1)[0] if "_" in s else s


def _positive_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) and out > 0 else None


def _duration_days(start, end, nno_days, uptime_factor: float) -> float | None:
    nno = _positive_float(nno_days)
    if nno is not None:
        return nno
    if pd.isna(start) or pd.isna(end):
        return None
    days = max(0.0, float((pd.Timestamp(end) - pd.Timestamp(start)).days))
    return days * uptime_factor if days > 0 else None


def _bundle_uptime_factor(source_bundle_date: str) -> float:
    models = pd.read_csv(C.model_registry_path(source_bundle_date), encoding="utf-8-sig")
    focus = models[models["stratum"] == "Mc_nonsour_Pooled"]
    if focus.empty:
        return 0.745
    return float(focus["uptime_factor"].iloc[0])


def build_dataset(
    *,
    prediction_workbook_path: Path | None = None,
    equipment_big_path: Path | None = None,
    source_bundle_date: str = SOURCE_BUNDLE_DATE,
    cutoff: pd.Timestamp = DEFAULT_CUTOFF,
) -> pd.DataFrame:
    """Build censored Mc/MR run frame, preferring Big intervals over duplicate Свод rows."""
    uptime = _bundle_uptime_factor(source_bundle_date)
    rows: list[dict] = []

    big = load_equipment_big(equipment_big_path or resolve_equipment_big_path())
    if not big.empty:
        focus = big[
            big.get("is_esp", pd.Series(False, index=big.index)).fillna(False)
            & big["well_key"].notna()
            & big["install_date"].notna()
        ].copy()
        focus = focus[
            focus["well_key"].map(_prefix).isin(MC_PREFIXES)
            | focus.get("field_raw", pd.Series("", index=focus.index)).astype(str).str.contains("Мачч|Нелб", case=False, regex=True)
        ]
        for _, row in focus.iterrows():
            well = crosswalk.norm_well(row.get("well_key"))
            start = pd.to_datetime(row.get("install_date"), errors="coerce")
            if well is None or pd.isna(start) or start >= cutoff:
                continue
            fail = pd.to_datetime(row.get("fail_date"), errors="coerce")
            pull = pd.to_datetime(row.get("pull_date"), errors="coerce")
            event = bool(pd.notna(fail) and fail <= cutoff)
            if event:
                end = fail
            else:
                candidates = [d for d in (pull, cutoff) if pd.notna(d) and d > start]
                end = min(candidates) if candidates else cutoff
            duration = _duration_days(start, end, row.get("nno_days"), uptime)
            if duration is None or duration <= 0:
                continue
            rows.append(
                {
                    "well_code": well,
                    "install_date": start,
                    "end_date": pd.Timestamp(end),
                    "tte": float(duration),
                    "event": int(event),
                    "source": "Big",
                    "contractor_group": "brt",
                }
            )

    src = crosswalk.load_esp_source(
        bundle_date=source_bundle_date,
        prediction_workbook_path=prediction_workbook_path or resolve_prediction_workbook_path(),
    )
    for well, runs in src.runs_by_well.items():
        if _prefix(well) not in MC_PREFIXES and not any(str(r.field_raw or "").strip().lower() in {"mc", "mr"} for r in runs):
            continue
        for run in runs:
            start = pd.Timestamp(run.mount) if run.mount is not None else pd.NaT
            if pd.isna(start) or start >= cutoff:
                continue
            event = bool(run.failure_flag == 1 and run.stop is not None and pd.Timestamp(run.stop) <= cutoff)
            if event:
                end = pd.Timestamp(run.stop)
            else:
                candidates = [pd.Timestamp(d) for d in (run.demo, run.stop, cutoff) if d is not None and pd.Timestamp(d) > start]
                end = min(candidates) if candidates else cutoff
            duration = _duration_days(start, end, run.age_op, uptime)
            if duration is None or duration <= 0:
                continue
            rows.append(
                {
                    "well_code": well,
                    "install_date": start,
                    "end_date": end,
                    "tte": float(duration),
                    "event": int(event),
                    "source": "Свод",
                    "contractor_group": crosswalk.contractor_group(run.ctr_raw),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Mc/MR refit dataset is empty.")

    # Prefer Big when a Свод row for the same well has a mount date within ±7 days.
    df["source_rank"] = df["source"].map({"Big": 0, "Свод": 1}).fillna(9)
    df = df.sort_values(["well_code", "install_date", "source_rank"]).reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    for well, idxs in df.groupby("well_code").groups.items():
        kept_big_dates: list[pd.Timestamp] = []
        for idx in idxs:
            row = df.loc[idx]
            start = pd.Timestamp(row["install_date"])
            if row["source"] == "Big":
                kept_big_dates.append(start)
                continue
            if any(abs((start - b).days) <= 7 for b in kept_big_dates):
                keep[idx] = False
    df = df[keep].drop(columns=["source_rank"]).reset_index(drop=True)
    return df


def _fit_frame(df: pd.DataFrame, label: str) -> tuple[dict, TwoComponentLatentWeibullModel]:
    durations = df["tte"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=int)
    result = fit_latent_weibull_em(durations, events, num_starts=24, min_eta_ratio=2.0)
    model = result.model
    nll_k2 = float(result.objective_value)
    aic_k2 = 2.0 * 5 + 2.0 * nll_k2
    sw = fit_basic_weibull(durations, events)
    nll_k1 = float(sw["nll"])
    aic_k1 = float(sw["aic"])
    delta_aic = aic_k1 - aic_k2
    degenerate = "DEGENERATE" in result.message
    if degenerate:
        model_kind = "k1_degenerate"
    elif delta_aic <= 0:
        model_kind = "k1_aic"
    else:
        model_kind = "k2"

    if model_kind == "k2":
        w1 = float(model.weight_1)
        b1 = float(model.component_1.beta)
        e1 = float(model.component_1.eta)
        b2 = float(model.component_2.beta)
        e2 = float(model.component_2.eta)
    else:
        b = float(sw["beta"])
        e = float(sw["eta"])
        w1, b1, e1, b2, e2 = 0.0, b, e, b, e
        model = TwoComponentLatentWeibullModel(
            weight_1=0.0,
            component_1=WeibullParameters(beta=b, eta=e, label="C1"),
            component_2=WeibullParameters(beta=b, eta=e, label="C2"),
        )

    def q(p: float) -> float:
        return float(latent_life_quantile(p, model))

    row = {
        "label": label,
        "model_kind": model_kind,
        "n_runs": int(len(df)),
        "n_failures": int(events.sum()),
        "w1": w1,
        "beta1": b1,
        "eta1": e1,
        "beta2": b2,
        "eta2": e2,
        "b20": q(0.20),
        "b50": q(0.50),
        "b80": q(0.80),
        "nll_k1": nll_k1,
        "nll_k2": nll_k2,
        "aic_k1": aic_k1,
        "aic_k2": aic_k2,
        "delta_aic": delta_aic,
        "message": result.message,
    }
    return row, model


def _registry_rows(fit_row: dict, source_rows: pd.DataFrame, fit_date: str) -> list[dict]:
    """Build the single replacement row (Mc_nonsour_Pooled).

    Only the pooled row is emitted; brt/slb/oth Mc wells resolve to it via the
    StrataModel cascade.  Emitting a per-contractor copy would duplicate the same
    numbers under a misleading ``contractor_group`` and ``n_runs`` label.
    """
    row = source_rows[source_rows["stratum"] == "Mc_nonsour_Pooled"].iloc[0].to_dict()
    row.update(
        {
            "stratum": "Mc_nonsour_Pooled",
            "field": "Mc",
            "h2s_class": "nonsour",
            "contractor_group": "Pooled",
            "model_kind": fit_row["model_kind"],
            "w1": round(float(fit_row["w1"]), 6),
            "beta1": round(float(fit_row["beta1"]), 6),
            "eta1": round(float(fit_row["eta1"]), 2),
            "beta2": round(float(fit_row["beta2"]), 6),
            "eta2": round(float(fit_row["eta2"]), 2),
            "b20": round(float(fit_row["b20"]), 1),
            "b50": round(float(fit_row["b50"]), 1),
            "b80": round(float(fit_row["b80"]), 1),
            "b50_lo": fit_row.get("b50_lo", ""),
            "b50_hi": fit_row.get("b50_hi", ""),
            "uptime_factor": 0.7450,
            "n_runs": int(fit_row["n_runs"]),
            "n_failures": int(fit_row["n_failures"]),
            "pct_mixed_clock": "",
            "clock": "ttf_mix",
            "fit_date": fit_date,
        }
    )
    return [row]


def run(
    *,
    prediction_workbook_path: Path | None = None,
    equipment_big_path: Path | None = None,
    source_bundle_date: str = SOURCE_BUNDLE_DATE,
    target_bundle_date: str = TARGET_BUNDLE_DATE,
    cutoff: pd.Timestamp = DEFAULT_CUTOFF,
) -> RefitResult:
    report_dir = results_dir("esp_survival_mc_refit")
    fit_date = target_bundle_date
    df = build_dataset(
        prediction_workbook_path=prediction_workbook_path,
        equipment_big_path=equipment_big_path,
        source_bundle_date=source_bundle_date,
        cutoff=cutoff,
    )
    all_row, _ = _fit_frame(df, "all_history")
    recent = df[df["install_date"] >= RECENCY_START].copy()
    if int(recent["event"].sum()) < 50:
        raise RuntimeError(f"Mc/MR recency window has too few failures: {int(recent['event'].sum())}")
    recent_row, recent_model = _fit_frame(recent, f"install_{RECENCY_START.year}plus")
    if not (120.0 <= float(recent_row["b50"]) <= 320.0):
        raise RuntimeError(f"Mc/MR refit b50 sanity failed: {recent_row['b50']:.1f} op-days")
    ci = bootstrap_stratum_ci(
        recent.rename(columns={"well_code": "well_key"}),
        recent_model,
        n_boot=40,
        seed=42,
        max_iter=100,
    )
    recent_row["b50_lo"] = ci.get("b50_lo")
    recent_row["b50_hi"] = ci.get("b50_hi")
    recent_row["b50_ci_reliable"] = ci.get("reliable")

    fit_rows = pd.DataFrame([all_row, recent_row])
    (report_dir / "tables" / "mc_refit_dataset.csv").write_text(
        df.to_csv(index=False), encoding="utf-8-sig"
    )
    fit_rows.to_csv(report_dir / "tables" / "mc_refit_summary.csv", index=False, encoding="utf-8-sig")

    src_dir = C.bundle_dir(source_bundle_date)
    dst_dir = C.BUNDLE_ROOT / target_bundle_date
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    models = pd.read_csv(src_dir / "esp_models.csv", encoding="utf-8-sig")
    replacement = pd.DataFrame(_registry_rows(recent_row, models, fit_date))
    models = models[~models["stratum"].isin(TARGET_STRATA)].copy()
    models = pd.concat([models, replacement], ignore_index=True)
    models.to_csv(dst_dir / "esp_models.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Mc/MR Refit Summary",
        "",
        f"- Source bundle: `{source_bundle_date}`",
        f"- Target bundle: `{target_bundle_date}`",
        f"- Cutoff: `{cutoff.date().isoformat()}`",
        f"- Dataset rows: `{len(df)}`, failures: `{int(df['event'].sum())}`",
        f"- Recency rows: `{len(recent)}`, failures: `{int(recent['event'].sum())}`",
        "",
        "## Shipped Recency Fit",
        "",
        f"- model_kind: `{recent_row['model_kind']}`",
        f"- w1: `{recent_row['w1']:.4f}`",
        f"- beta1/eta1: `{recent_row['beta1']:.4f}` / `{recent_row['eta1']:.1f}`",
        f"- beta2/eta2: `{recent_row['beta2']:.4f}` / `{recent_row['eta2']:.1f}`",
        f"- b50: `{recent_row['b50']:.1f}` op-days",
        f"- b50 CI: `{recent_row.get('b50_lo')}`–`{recent_row.get('b50_hi')}` op-days "
        f"(bootstrap reliable: `{recent_row.get('b50_ci_reliable')}`)",
        f"- delta_aic: `{recent_row['delta_aic']:.1f}`",
        "",
        "Rows in `esp_models.csv`: `Mc_nonsour_Pooled` replaced; old `Mc_nonsour_brt` "
        "dropped (brt now cascades to the pooled fit).",
        "",
    ]
    (report_dir / "reports" / "mc_refit_summary.md").write_text("\n".join(report), encoding="utf-8")
    return RefitResult(bundle_dir=dst_dir, report_dir=report_dir, dataset=df, fit_rows=fit_rows)
