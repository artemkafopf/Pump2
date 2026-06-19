from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.input_paths import resolve_v03_all_path
from analysis.sqlite_paths import resolve_techregime_db_path
from scripts.analyze_failure_horizon import load_runs
from scripts.data_utils import _numeric, load_daily_merged


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "ttf_true_analysis"
TR_DB_PATH = resolve_techregime_db_path()

TELEMETRY_IN_OPERATION_RULE = "qliq > 0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare calendar/runtime TTF against operational-day TTF from telemetry and techregime.")
    parser.add_argument("--input", default=str(resolve_v03_all_path()), help="Path to V03_all workbook.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def load_runs_with_pdk(path: Path) -> pd.DataFrame:
    runs = load_runs(path).copy()
    runs["calendar_days"] = (runs["Дата остановки"] - runs["Дата монтажа"]).dt.days.astype(float)
    if "Наработка (сут)" in runs.columns:
        runs["pdk_ttf_days"] = _numeric(runs["Наработка (сут)"])
    else:
        runs["pdk_ttf_days"] = np.nan
    runs["pdk_minus_calendar_days"] = runs["pdk_ttf_days"] - runs["calendar_days"]
    return runs


def load_techregime_status_daily(wells: list[str]) -> pd.DataFrame:
    if not wells:
        return pd.DataFrame(columns=["well_id", "dt", "status", "treg_in_operation"])
    frames: list[pd.DataFrame] = []
    with sqlite3.connect(TR_DB_PATH) as connection:
        for start in range(0, len(wells), 400):
            chunk = wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT col_0003 AS well_id, col_0010 AS dt, col_0004 AS status
                FROM techregime_records
                WHERE col_0003 IN ({placeholders})
            """
            frames.append(pd.read_sql_query(query, connection, params=chunk))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["well_id", "dt", "status"])
    df["well_id"] = df["well_id"].astype("string").str.strip()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce", dayfirst=True)
    df["status"] = df["status"].astype("string").str.strip()
    df = df.loc[df["well_id"].notna() & df["dt"].notna()].copy()
    df = (
        df.groupby(["well_id", "dt"], as_index=False)["status"]
        .agg(lambda s: next((str(v) for v in s.tolist() if pd.notna(v) and str(v) != "<NA>"), pd.NA))
        .sort_values(["well_id", "dt"])
        .reset_index(drop=True)
    )
    df["treg_in_operation"] = df["status"].astype("string").eq("В работе").fillna(False)
    return df


def telemetry_vs_techregime_candidate_report(wells: list[str]) -> pd.DataFrame:
    daily = load_daily_merged(wells)
    telemetry = daily.loc[daily["source"].eq("telemetry")].copy()
    treg = load_techregime_status_daily(wells)
    merged = telemetry.merge(treg[["well_id", "dt", "treg_in_operation"]], on=["well_id", "dt"], how="inner")
    for column in ["freq", "qliq", "load"]:
        merged[column] = _numeric(merged[column])
    y = merged["treg_in_operation"].fillna(False).astype(bool)
    candidates = {
        "qliq_gt_0": merged["qliq"] > 0,
        "freq_gt_0": merged["freq"] > 0,
        "load_gt_0": merged["load"] > 0,
        "freq_and_qliq": (merged["freq"] > 0) & (merged["qliq"] > 0),
        "freq_or_qliq": (merged["freq"] > 0) | (merged["qliq"] > 0),
        "any_positive": (merged["freq"] > 0) | (merged["qliq"] > 0) | (merged["load"] > 0),
    }
    rows: list[dict[str, object]] = []
    for name, pred in candidates.items():
        pred = pred.fillna(False).astype(bool)
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        tn = int((~pred & ~y).sum())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        specificity = float(tn / max(tn + fp, 1))
        f1 = float((2 * precision * recall) / max(precision + recall, 1e-9))
        accuracy = float((tp + tn) / max(len(y), 1))
        rows.append(
            {
                "candidate": name,
                "rows": int(len(y)),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)


def compare_ttf_sources(runs: pd.DataFrame) -> pd.DataFrame:
    wells = sorted(runs["Скв."].dropna().astype(str).str.strip().unique().tolist())
    daily = load_daily_merged(wells)
    treg = load_techregime_status_daily(wells)

    daily["tele_in_operation"] = (_numeric(daily.get("qliq", pd.Series(index=daily.index, dtype=float))) > 0).fillna(False)
    treg_indexed = treg.set_index(["well_id", "dt"]) if not treg.empty else treg

    rows: list[dict[str, object]] = []
    for run in runs.to_dict(orient="records"):
        well_id = str(run["Скв."]).strip()
        install = pd.Timestamp(run["Дата монтажа"])
        stop = pd.Timestamp(run["Дата остановки"])

        run_daily = daily.loc[
            (daily["well_id"].astype("string") == well_id)
            & (daily["dt"] >= install)
            & (daily["dt"] <= stop)
        ].copy()
        tele_days = int(run_daily["dt"].nunique())
        tele_operating_days = int(run_daily.loc[run_daily["tele_in_operation"], "dt"].nunique())

        treg_operating_days = 0
        treg_days = 0
        if not treg.empty:
            treg_run = treg.loc[
                (treg["well_id"].astype("string") == well_id)
                & (treg["dt"] >= install)
                & (treg["dt"] <= stop)
            ].copy()
            treg_days = int(treg_run["dt"].nunique())
            treg_operating_days = int(treg_run.loc[treg_run["treg_in_operation"], "dt"].nunique())

        if treg_operating_days > 0:
            ttf_true_best = float(treg_operating_days)
            ttf_true_source = "techregime_status"
        elif tele_operating_days > 0:
            ttf_true_best = float(tele_operating_days)
            ttf_true_source = "telemetry_qliq_gt_0"
        else:
            ttf_true_best = np.nan
            ttf_true_source = "missing"

        rows.append(
            {
                "row_id": int(run["row_id"]),
                "well_id": well_id,
                "field": run.get("Месторождение"),
                "contractor": run.get("Принадлежность"),
                "pad": run.get("Куст"),
                "install_date": install,
                "stop_date": stop,
                "event": int(run["event"]),
                "calendar_days": float(run["calendar_days"]),
                "pdk_ttf_days": float(run["pdk_ttf_days"]) if pd.notna(run["pdk_ttf_days"]) else np.nan,
                "ttf_tele_days": float(tele_operating_days) if tele_days > 0 else np.nan,
                "ttf_treg_days": float(treg_operating_days) if treg_days > 0 else np.nan,
                "ttf_true_best_days": ttf_true_best,
                "ttf_true_source": ttf_true_source,
                "telemetry_days_in_interval": tele_days,
                "techregime_days_in_interval": treg_days,
            }
        )
    result = pd.DataFrame(rows)
    result["pdk_minus_calendar_days"] = result["pdk_ttf_days"] - result["calendar_days"]
    result["pdk_minus_tele_days"] = result["pdk_ttf_days"] - result["ttf_tele_days"]
    result["pdk_minus_treg_days"] = result["pdk_ttf_days"] - result["ttf_treg_days"]
    result["pdk_minus_true_best_days"] = result["pdk_ttf_days"] - result["ttf_true_best_days"]
    return result


def build_summary(run_cmp: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": int(len(run_cmp)),
        "rows_with_pdk_ttf": int(run_cmp["pdk_ttf_days"].notna().sum()),
        "rows_with_ttf_tele": int(run_cmp["ttf_tele_days"].notna().sum()),
        "rows_with_ttf_treg": int(run_cmp["ttf_treg_days"].notna().sum()),
        "rows_with_ttf_true_best": int(run_cmp["ttf_true_best_days"].notna().sum()),
        "ttf_true_source_counts": {str(k): int(v) for k, v in run_cmp["ttf_true_source"].astype("string").value_counts().items()},
        "telemetry_in_operation_rule": TELEMETRY_IN_OPERATION_RULE,
        "telemetry_candidate_ranking": candidates.to_dict(orient="records"),
    }
    for left, right, diff in [
        ("pdk_ttf_days", "calendar_days", "pdk_minus_calendar_days"),
        ("pdk_ttf_days", "ttf_tele_days", "pdk_minus_tele_days"),
        ("pdk_ttf_days", "ttf_treg_days", "pdk_minus_treg_days"),
        ("pdk_ttf_days", "ttf_true_best_days", "pdk_minus_true_best_days"),
    ]:
        mask = run_cmp[left].notna() & run_cmp[right].notna()
        key = f"{left}_vs_{right}"
        if not mask.any():
            summary[key] = {"rows": 0}
            continue
        delta = run_cmp.loc[mask, diff]
        summary[key] = {
            "rows": int(mask.sum()),
            "mean_delta_days": float(delta.mean()),
            "median_delta_days": float(delta.median()),
            "p10_delta_days": float(delta.quantile(0.10)),
            "p90_delta_days": float(delta.quantile(0.90)),
            "corr": float(run_cmp.loc[mask, left].corr(run_cmp.loc[mask, right])),
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs_with_pdk(Path(args.input))
    wells = sorted(runs["Скв."].dropna().astype(str).str.strip().unique().tolist())
    candidates = telemetry_vs_techregime_candidate_report(wells)
    run_cmp = compare_ttf_sources(runs)
    summary = build_summary(run_cmp, candidates)

    run_cmp.to_csv(output_dir / "ttf_true_comparison_by_run.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output_dir / "telemetry_in_operation_candidates.csv", index=False, encoding="utf-8-sig")
    (output_dir / "ttf_true_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# TTF True Analysis",
        "",
        f"- Input workbook: `{args.input}`",
        f"- Telemetry in-operation rule selected from overlap analysis: `{TELEMETRY_IN_OPERATION_RULE}`",
        "",
        "## Coverage",
        "",
        f"- Runs: `{summary['rows']}`",
        f"- Runs with PDK TTF: `{summary['rows_with_pdk_ttf']}`",
        f"- Runs with telemetry TTF: `{summary['rows_with_ttf_tele']}`",
        f"- Runs with techregime TTF: `{summary['rows_with_ttf_treg']}`",
        f"- Runs with best-available TTF_true: `{summary['rows_with_ttf_true_best']}`",
        f"- TTF_true source counts: `{json.dumps(summary['ttf_true_source_counts'], ensure_ascii=False)}`",
        "",
        "## Best telemetry candidate",
        "",
        f"- Top candidate row: `{json.dumps(candidates.iloc[0].to_dict(), ensure_ascii=False) if not candidates.empty else '{}'} `",
        "",
        "## Comparison snapshots",
        "",
        f"- PDK vs calendar: `{json.dumps(summary['pdk_ttf_days_vs_calendar_days'], ensure_ascii=False)}`",
        f"- PDK vs telemetry-op days: `{json.dumps(summary['pdk_ttf_days_vs_ttf_tele_days'], ensure_ascii=False)}`",
        f"- PDK vs techregime-op days: `{json.dumps(summary['pdk_ttf_days_vs_ttf_treg_days'], ensure_ascii=False)}`",
        f"- PDK vs best-available TTF_true: `{json.dumps(summary['pdk_ttf_days_vs_ttf_true_best_days'], ensure_ascii=False)}`",
        "",
    ]
    (output_dir / "ttf_true_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Outputs written to: {output_dir}")
    if not candidates.empty:
        print("Top telemetry in-operation candidate:")
        print(candidates.head(5).to_string(index=False))
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
