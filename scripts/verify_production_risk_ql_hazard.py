"""Verification harness for the production-risk Ql hazard layer.

This script implements the cheap/output-level checks from
docs/notes/production_risk_ql_hazard_verification_plan.md.  It runs the workflow
once in memory, writes audit tables under results/production_risk_ql_hazard_verification,
and exits non-zero only for contract violations that are code-level invariants.
Model-review findings are reported as WARN so the reviewer can make the explicit
shipping decisions called out in the plan.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for item in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import config as C  # noqa: E402
from analysis.workflows.production_risk.run import run  # noqa: E402


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _status(passed: bool, *, warn: bool = False) -> str:
    if passed:
        return "PASS"
    return "WARN" if warn else "FAIL"


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _theta_checks(production: dict[str, pd.DataFrame], projection: pd.DataFrame) -> tuple[list[Check], pd.DataFrame]:
    checks: list[Check] = []
    base = projection[projection["scenario"].astype(str).str.startswith("base")].copy()
    if base.empty:
        checks.append(Check("base_q_layer_neutral", "FAIL", "No base scenario rows found."))
    else:
        neutral = (
            np.allclose(base["theta_ql"].to_numpy(dtype=float), 1.0, atol=1e-12)
            and np.allclose(base["theta_qw"].to_numpy(dtype=float), 1.0, atol=1e-12)
            and np.allclose(
                base["theta"].to_numpy(dtype=float),
                base["theta_static"].to_numpy(dtype=float),
                atol=1e-12,
            )
        )
        checks.append(
            Check(
                "base_q_layer_neutral",
                _status(bool(neutral)),
                "Base rows have theta_ql=1, theta_qw=1, theta=theta_static."
                if neutral
                else "Base scenario has non-neutral Ql/Qw theta values.",
            )
        )

    by_well = production["by_well"]
    needed = {"theta", "theta_static", "theta_ql", "theta_qw"}
    missing = sorted(needed - set(by_well.columns))
    if missing:
        checks.append(Check("by_well_theta_columns", "FAIL", f"Missing columns: {', '.join(missing)}"))
        theta_audit = pd.DataFrame()
    else:
        stress = by_well[by_well["scenario"] == C.STRESS_SCENARIO_ID].copy()
        identity = np.allclose(
            stress["theta"].to_numpy(dtype=float),
            stress["theta_static"].to_numpy(dtype=float) * stress["theta_ql"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
        )
        qw_neutral = np.allclose(stress["theta_qw"].to_numpy(dtype=float), 1.0, atol=1e-12)
        checks.append(Check("by_well_theta_columns", _status(True), "theta_static/theta_ql/theta_qw are present."))
        checks.append(
            Check(
                "by_well_theta_identity",
                _status(bool(identity and qw_neutral)),
                "Stress by_well theta identity holds and theta_qw is neutral."
                if identity and qw_neutral
                else "Stress by_well theta != theta_static * theta_ql or theta_qw != 1.",
            )
        )
        theta_audit = (
            stress.assign(age_band=pd.cut(stress["age_mean"], bins=[-1, 180, 533, 1000, 2000, np.inf]))
            .groupby(["model_field", "age_band"], observed=False)["theta_ql"]
            .quantile([0.05, 0.5, 0.95])
            .unstack()
            .reset_index()
            .rename(columns={0.05: "q05", 0.5: "q50", 0.95: "q95"})
        )
    return checks, theta_audit


def _stress_base_envelope(projection: pd.DataFrame) -> tuple[Check, pd.DataFrame]:
    base = projection[projection["scenario"] == "base_p75"].copy()
    stress = projection[projection["scenario"] == C.STRESS_SCENARIO_ID].copy()
    key = ["wid", "month"]
    merged = stress.merge(
        base[key + ["expected_failures"]].rename(columns={"expected_failures": "base_expected_failures"}),
        on=key,
        how="inner",
        suffixes=("", "_stress"),
    )
    merged = merged[merged["base_expected_failures"] > 1e-12].copy()
    if merged.empty:
        return Check("stress_base_envelope", "FAIL", "No comparable base/stress rows."), merged
    merged["stress_base_ratio"] = merged["expected_failures"] / merged["base_expected_failures"]
    ql_cap = math.exp(
        C.QL_HAZARD_CAP_LOG_RATIO
        * max(
            abs(C.QL_HAZARD_BETA + C.QL_HAZARD_GAMMA * math.log(1.0)),
            abs(C.QL_HAZARD_BETA + C.QL_HAZARD_GAMMA * math.log(2500.0)),
        )
    )
    theta_static = merged["theta_static"].astype(float)
    merged["envelope_upper"] = np.maximum(theta_static, 1.0 / theta_static) * ql_cap * 1.05
    merged["envelope_lower"] = np.minimum(theta_static, 1.0 / theta_static) / ql_cap / 1.05
    violations = merged[
        (merged["stress_base_ratio"] > merged["envelope_upper"])
        | (merged["stress_base_ratio"] < merged["envelope_lower"])
    ].copy()
    status = _status(violations.empty)
    detail = (
        f"Compared {len(merged)} well-months; ql_cap={ql_cap:.3g}; violations={len(violations)}; "
        f"max_ratio={merged['stress_base_ratio'].max():.3g}."
    )
    return Check("stress_base_envelope", status, detail), violations


def _failure_rate_fit(fr_base, fr_stress) -> tuple[list[Check], pd.DataFrame, pd.DataFrame]:
    rows = []
    ratios = []
    for label, result in [("base", fr_base), ("stress", fr_stress)]:
        df = result.monthly.copy()
        fact = df[(df["month"] >= "2024-01") & (df["month"] <= "2026-06")].copy()
        fact = fact[fact["observed_failures"].notna() & fact["observed_rate"].notna()]
        for field, grp in fact.groupby("field"):
            err = grp["predicted_failures"].astype(float) - grp["observed_failures"].astype(float)
            rows.append(
                {
                    "scenario": label,
                    "field": field,
                    "months": int(len(grp)),
                    "fact_failures": float(grp["observed_failures"].sum()),
                    "model_failures": float(grp["predicted_failures"].sum()),
                    "mae_failures": float(err.abs().mean()),
                    "mean_bias_failures": float(err.mean()),
                }
            )
            if label == "base" and "2024-01" <= str(grp["month"].min()) <= "2026-06":
                model_sum = float(grp["predicted_failures"].sum())
                fact_sum = float(grp["observed_failures"].sum())
                ratios.append(
                    {
                        "field": field,
                        "fact_failures": fact_sum,
                        "model_failures": model_sum,
                        "fact_model_ratio": fact_sum / model_sum if model_sum > 0 else np.nan,
                    }
                )
    fit = pd.DataFrame(rows).sort_values(["scenario", "field"])
    ratio_df = pd.DataFrame(ratios).sort_values("field")
    material = ratio_df[ratio_df["fact_failures"] >= 20].copy()
    material["abs_ratio_error"] = (material["fact_model_ratio"] - 1.0).abs()
    bad = material[material["abs_ratio_error"] > 0.10]
    checks = [
        Check(
            "reporting_field_calibration_ratio",
            _status(bad.empty, warn=True),
            f"Material fields checked={len(material)}; outside ±10%={len(bad)}."
            + (" Review calibration decisions." if not bad.empty else ""),
        )
    ]
    pivot = fit.pivot_table(index="field", columns="scenario", values="mae_failures", aggfunc="first")
    if {"base", "stress"} <= set(pivot.columns):
        worse = pivot[pivot["stress"] > pivot["base"]]
        checks.append(
            Check(
                "stress_fact_window_mae",
                _status(worse.empty, warn=True),
                f"Stress MAE worse than base in {len(worse)} fields over 2024-01..2026-06.",
            )
        )
    return checks, fit, ratio_df


def _ql_coverage_check(fr_stress) -> tuple[Check, pd.DataFrame]:
    audit = fr_stress.coverage.get("ql_hazard_firing_audit", {})
    status_counts = audit.get("status_counts", {}) if isinstance(audit, dict) else {}
    plan_counts = audit.get("plan_month_status_counts", {}) if isinstance(audit, dict) else {}
    plan_failures = sum(v for k, v in plan_counts.items() if k not in {"applied_daily", "applied_monthly"})
    total = int(audit.get("total_slices", 0)) if isinstance(audit, dict) else 0
    check = Check(
        "ql_hazard_firing_coverage",
        _status(plan_failures == 0),
        f"historical slices={total}; status_counts={status_counts}; plan_month_nonapplied={plan_failures}.",
    )
    sample = pd.DataFrame(audit.get("by_field_month_sample", [])) if isinstance(audit, dict) else pd.DataFrame()
    return check, sample


def build_report(checks: list[Check], out_dir: Path) -> None:
    lines = ["# Production Risk Ql Hazard Verification", ""]
    for check in checks:
        lines.append(f"- **{check.status}** `{check.name}` — {check.detail}")
    lines.append("")
    lines.append("Artifacts:")
    for name in [
        "checks.csv",
        "theta_ql_distribution.csv",
        "stress_base_envelope_violations.csv",
        "failure_rate_fit.csv",
        "calibration_ratios.csv",
        "ql_hazard_firing_sample.csv",
    ]:
        lines.append(f"- `{name}`")
    (out_dir / "verification_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-start", type=_parse_date, default=C.FORECAST_START)
    parser.add_argument("--horizon-end", type=_parse_date, default=C.HORIZON_END)
    parser.add_argument("--bundle-date", default=C.BUNDLE_DATE)
    args = parser.parse_args()

    cfg = C.RunConfig(
        forecast_start=args.forecast_start,
        horizon_end=args.horizon_end,
        bundle_date=args.bundle_date,
        write_excel=False,
        full_tables=False,
    )
    result = run(cfg)
    out_dir = results_dir("production_risk_ql_hazard_verification")

    checks: list[Check] = []
    theta_checks, theta_audit = _theta_checks(result["production"], result["projection"])
    checks.extend(theta_checks)
    envelope_check, envelope_violations = _stress_base_envelope(result["projection"])
    checks.append(envelope_check)
    fit_checks, fit, ratios = _failure_rate_fit(result["failure_rate"], result["failure_rate_stress"])
    checks.extend(fit_checks)
    ql_check, ql_sample = _ql_coverage_check(result["failure_rate_stress"])
    checks.append(ql_check)

    checks_df = pd.DataFrame([check.__dict__ for check in checks])
    _save(checks_df, out_dir / "tables" / "checks.csv")
    _save(theta_audit, out_dir / "tables" / "theta_ql_distribution.csv")
    _save(envelope_violations, out_dir / "tables" / "stress_base_envelope_violations.csv")
    _save(fit, out_dir / "tables" / "failure_rate_fit.csv")
    _save(ratios, out_dir / "tables" / "calibration_ratios.csv")
    _save(ql_sample, out_dir / "tables" / "ql_hazard_firing_sample.csv")
    (out_dir / "reports" / "coverage.json").write_text(
        json.dumps(result["failure_rate_stress"].coverage, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    build_report(checks, out_dir / "reports")

    print(f"Verification artifacts written to: {out_dir}")
    print(checks_df.to_string(index=False))
    return 1 if (checks_df["status"] == "FAIL").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
