from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for path_text in (str(REPO_ROOT), str(BACKEND_DIR)):
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from analysis.stress_transforms import apply_transform
from scripts.analyze_vt_60hz_scenario import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SCENARIO_DIR,
    LOAD_MEAN_RESPONSE_FEATURES,
    LOAD_STD_RESPONSE_FEATURES,
    PRESSURE_RESPONSE_FEATURES,
    _numeric,
    apply_quality_filters,
    fit_regressor,
    fit_weibull_with_fallback,
    fit_frequency_response_model,
    latest_portfolio_rows,
    load_rolling_dataset,
    load_vt_rolling_dataset,
    propagate_scenario_state,
    select_stress_terms,
    weibull_group_lookup,
    weibull_row_summary,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_scenario_phase1_decomposition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose Vt 60 Hz Weibull scenario drivers well by well.")
    parser.add_argument("--field", default="Vt", help="Field code.")
    parser.add_argument("--scenario-frequency", type=float, default=60.0, help="Scenario frequency in Hz.")
    parser.add_argument("--portfolio-cutoff", default="2025-01-01", help="Latest portfolio cutoff date.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def row_term_contributions(result: object, row: pd.Series) -> list[dict[str, float | str]]:
    items: list[dict[str, float | str]] = []
    for term in result.stress_terms:
        coefficient, reference, reference_multiplier = result.resolve_term_parameters(term.name, row=row)
        value = float(row[term.column])
        transformed = float(apply_transform(term.transform, [value], [reference], scale=term.scale)[0])
        weighted = float(coefficient * transformed)
        items.append(
            {
                "term_name": term.name,
                "column": term.column,
                "transform": term.transform,
                "coefficient": float(coefficient),
                "reference": float(reference),
                "reference_multiplier": float(reference_multiplier),
                "scale": float(term.scale),
                "value": value,
                "transformed": transformed,
                "weighted_stress": weighted,
            }
        )
    return items


def summarize_direction(series: pd.Series) -> dict[str, float]:
    numeric = _numeric(series)
    valid = numeric.dropna()
    if valid.empty:
        return {"mean": float("nan"), "median": float("nan"), "positive_share": float("nan"), "negative_share": float("nan")}
    return {
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "positive_share": float((valid > 0).mean()),
        "negative_share": float((valid < 0).mean()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff_date = pd.Timestamp(args.portfolio_cutoff)

    _, _, rolling_all, _ = load_rolling_dataset(None)
    filtered_all, _ = apply_quality_filters(rolling_all)
    _, _, rolling, _ = load_vt_rolling_dataset(args.field)
    filtered, _ = apply_quality_filters(rolling)
    portfolio = latest_portfolio_rows(filtered, cutoff_date)
    if portfolio.empty:
        raise RuntimeError("No portfolio rows remain after cutoff filter.")

    qliq_model = fit_frequency_response_model(filtered)
    load_mean_model = fit_regressor(filtered, target_column="load_30d_mean", feature_columns=LOAD_MEAN_RESPONSE_FEATURES)
    load_std_model = fit_regressor(filtered, target_column="load_30d_std", feature_columns=LOAD_STD_RESPONSE_FEATURES)
    pressure_model = fit_regressor(filtered, target_column="pressure_ratio_bhp_30d", feature_columns=PRESSURE_RESPONSE_FEATURES)

    selected_stress_terms, stress_ranking = select_stress_terms(
        filtered,
        duration_column="days_to_stop",
        event_column="event",
        group_columns=["Принадлежность", "pump_family"],
        min_group_size=20,
    )
    weibull_result, weibull_errors = fit_weibull_with_fallback(filtered, selected_stress_terms)

    baseline_portfolio = portfolio.copy()
    scenario_portfolio = propagate_scenario_state(
        baseline_portfolio,
        scenario_frequency_hz=float(args.scenario_frequency),
        qliq_model=qliq_model,
        load_mean_model=load_mean_model,
        load_std_model=load_std_model,
        pressure_model=pressure_model,
    )

    group_lookup = weibull_group_lookup(weibull_result)

    well_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    for row_index, baseline_row in baseline_portfolio.iterrows():
        analysis_row_id = int(baseline_row["analysis_row_id"])
        group_key = group_lookup.get(analysis_row_id)
        if group_key is None:
            continue
        scenario_row = scenario_portfolio.loc[row_index]

        base_summary = weibull_row_summary(weibull_result, group_key, baseline_row)
        scen_summary = weibull_row_summary(weibull_result, group_key, scenario_row)

        base_terms = {item["term_name"]: item for item in row_term_contributions(weibull_result, baseline_row)}
        scen_terms = {item["term_name"]: item for item in row_term_contributions(weibull_result, scenario_row)}

        base_total = float(sum(item["weighted_stress"] for item in base_terms.values()))
        scen_total = float(sum(item["weighted_stress"] for item in scen_terms.values()))
        delta_by_term = {
            term_name: float(scen_terms[term_name]["weighted_stress"] - base_terms[term_name]["weighted_stress"])
            for term_name in base_terms
        }
        dominant_term = min(delta_by_term, key=delta_by_term.get) if delta_by_term else None
        opposing_term = max(delta_by_term, key=delta_by_term.get) if delta_by_term else None

        well_rows.append(
            {
                "analysis_row_id": analysis_row_id,
                "well_id": baseline_row["well_id"],
                "anchor_date": str(pd.Timestamp(baseline_row["anchor_date"]).date()),
                "contractor": baseline_row.get("Принадлежность"),
                "pump_family": baseline_row.get("pump_family"),
                "group_key": group_key,
                "baseline_freq_30d_mean": float(baseline_row["freq_30d_mean"]),
                "scenario_freq_30d_mean": float(scenario_row["freq_30d_mean"]),
                "baseline_Kpod_30d": float(baseline_row["Kpod_30d"]),
                "scenario_Kpod_30d": float(scenario_row["Kpod_30d"]),
                "delta_Kpod_30d": float(scenario_row["Kpod_30d"] - baseline_row["Kpod_30d"]),
                "baseline_Kpod_freq_30d": float(baseline_row["Kpod_freq_30d"]),
                "scenario_Kpod_freq_30d": float(scenario_row["Kpod_freq_30d"]),
                "delta_Kpod_freq_30d": float(scenario_row["Kpod_freq_30d"] - baseline_row["Kpod_freq_30d"]),
                "baseline_freq_ratio_30d": float(baseline_row["frequency_to_reference_ratio_30d"]),
                "scenario_freq_ratio_30d": float(scenario_row["frequency_to_reference_ratio_30d"]),
                "delta_freq_ratio_30d": float(scenario_row["frequency_to_reference_ratio_30d"] - baseline_row["frequency_to_reference_ratio_30d"]),
                "baseline_pressure_ratio_bhp_30d": float(baseline_row["pressure_ratio_bhp_30d"]),
                "scenario_pressure_ratio_bhp_30d": float(scenario_row["pressure_ratio_bhp_30d"]),
                "delta_pressure_ratio_bhp_30d": float(scenario_row["pressure_ratio_bhp_30d"] - baseline_row["pressure_ratio_bhp_30d"]),
                "baseline_load_30d_std": float(baseline_row["load_30d_std"]),
                "scenario_load_30d_std": float(scenario_row["load_30d_std"]),
                "delta_load_30d_std": float(scenario_row["load_30d_std"] - baseline_row["load_30d_std"]),
                "baseline_weighted_stress_total": base_total,
                "scenario_weighted_stress_total": scen_total,
                "delta_weighted_stress_total": float(scen_total - base_total),
                "baseline_weibull_eta": float(base_summary["eta"]),
                "scenario_weibull_eta": float(scen_summary["eta"]),
                "delta_weibull_eta": float(scen_summary["eta"] - base_summary["eta"]),
                "baseline_weibull_median_ttf_days": float(base_summary["median_ttf_days"]),
                "scenario_weibull_median_ttf_days": float(scen_summary["median_ttf_days"]),
                "delta_weibull_median_ttf_days": float(scen_summary["median_ttf_days"] - base_summary["median_ttf_days"]),
                "baseline_weibull_fail_prob_30d": float(base_summary["fail_prob_30d"]),
                "scenario_weibull_fail_prob_30d": float(scen_summary["fail_prob_30d"]),
                "delta_weibull_fail_prob_30d": float(scen_summary["fail_prob_30d"] - base_summary["fail_prob_30d"]),
                "dominant_improving_term": dominant_term,
                "dominant_worsening_term": opposing_term,
            }
        )

        for term_name, base_term in base_terms.items():
            scen_term = scen_terms[term_name]
            term_rows.append(
                {
                    "analysis_row_id": analysis_row_id,
                    "well_id": baseline_row["well_id"],
                    "anchor_date": str(pd.Timestamp(baseline_row["anchor_date"]).date()),
                    "term_name": term_name,
                    "column": base_term["column"],
                    "transform": base_term["transform"],
                    "coefficient": float(base_term["coefficient"]),
                    "reference": float(base_term["reference"]),
                    "baseline_value": float(base_term["value"]),
                    "scenario_value": float(scen_term["value"]),
                    "delta_value": float(scen_term["value"] - base_term["value"]),
                    "baseline_transformed": float(base_term["transformed"]),
                    "scenario_transformed": float(scen_term["transformed"]),
                    "delta_transformed": float(scen_term["transformed"] - base_term["transformed"]),
                    "baseline_weighted_stress": float(base_term["weighted_stress"]),
                    "scenario_weighted_stress": float(scen_term["weighted_stress"]),
                    "delta_weighted_stress": float(scen_term["weighted_stress"] - base_term["weighted_stress"]),
                }
            )

    well_df = pd.DataFrame(well_rows)
    term_df = pd.DataFrame(term_rows)
    well_df.to_csv(output_dir / "vt_60hz_weibull_decomposition_by_well.csv", index=False, encoding="utf-8-sig")
    term_df.to_csv(output_dir / "vt_60hz_weibull_decomposition_by_term.csv", index=False, encoding="utf-8-sig")
    stress_ranking.to_csv(output_dir / "vt_60hz_weibull_stress_ranking.csv", index=False, encoding="utf-8-sig")

    term_summary = (
        term_df.groupby(["term_name", "column", "transform"], as_index=False)
        .agg(
            mean_delta_weighted_stress=("delta_weighted_stress", "mean"),
            median_delta_weighted_stress=("delta_weighted_stress", "median"),
            wells_improved=("delta_weighted_stress", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
            wells_worsened=("delta_weighted_stress", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            mean_baseline_value=("baseline_value", "mean"),
            mean_scenario_value=("scenario_value", "mean"),
        )
        .sort_values("mean_delta_weighted_stress")
        .reset_index(drop=True)
    )
    term_summary.to_csv(output_dir / "vt_60hz_weibull_term_summary.csv", index=False, encoding="utf-8-sig")

    dominant_counts = (
        well_df["dominant_improving_term"].astype("string").value_counts(dropna=True).rename_axis("term_name").reset_index(name="n_wells")
    )
    dominant_counts.to_csv(output_dir / "vt_60hz_dominant_improving_term_counts.csv", index=False, encoding="utf-8-sig")

    summary = {
        "portfolio_rows": int(len(well_df)),
        "baseline_Kpod_30d": summarize_direction(well_df["baseline_Kpod_30d"]),
        "scenario_Kpod_30d": summarize_direction(well_df["scenario_Kpod_30d"]),
        "delta_Kpod_30d": summarize_direction(well_df["delta_Kpod_30d"]),
        "baseline_Kpod_freq_30d": summarize_direction(well_df["baseline_Kpod_freq_30d"]),
        "scenario_Kpod_freq_30d": summarize_direction(well_df["scenario_Kpod_freq_30d"]),
        "delta_Kpod_freq_30d": summarize_direction(well_df["delta_Kpod_freq_30d"]),
        "delta_weighted_stress_total": summarize_direction(well_df["delta_weighted_stress_total"]),
        "delta_weibull_median_ttf_days": summarize_direction(well_df["delta_weibull_median_ttf_days"]),
        "delta_weibull_fail_prob_30d": summarize_direction(well_df["delta_weibull_fail_prob_30d"]),
        "dominant_improving_term_counts": dominant_counts.to_dict(orient="records"),
        "selected_stress_terms": selected_stress_terms,
        "weibull_group_columns": list(weibull_result.group_columns),
        "weibull_stage_summaries": weibull_result.to_dict()["stage_summaries"],
        "weibull_errors": weibull_errors,
    }
    (output_dir / "vt_60hz_weibull_decomposition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Vt 60 Hz Weibull Decomposition",
        "",
        f"- Portfolio rows analyzed: `{len(well_df)}`",
        f"- Weibull group columns: `{weibull_result.group_columns}`",
        "",
        "## Kpod Shift",
        "",
        f"- Mean baseline `Kpod_30d`: `{summary['baseline_Kpod_30d']['mean']:.4f}`",
        f"- Mean scenario `Kpod_30d`: `{summary['scenario_Kpod_30d']['mean']:.4f}`",
        f"- Mean delta `Kpod_30d`: `{summary['delta_Kpod_30d']['mean']:+.4f}`",
        "",
        "## Kpod_freq Shift",
        "",
        f"- Mean baseline `Kpod_freq_30d`: `{summary['baseline_Kpod_freq_30d']['mean']:.4f}`",
        f"- Mean scenario `Kpod_freq_30d`: `{summary['scenario_Kpod_freq_30d']['mean']:.4f}`",
        f"- Mean delta `Kpod_freq_30d`: `{summary['delta_Kpod_freq_30d']['mean']:+.4f}`",
        "",
        "## Weibull Stress Delta",
        "",
        f"- Mean total weighted stress delta: `{summary['delta_weighted_stress_total']['mean']:+.4f}`",
        f"- Share of wells with lower total stress: `{summary['delta_weighted_stress_total']['negative_share']:.3f}`",
        f"- Mean Weibull median TTF delta: `{summary['delta_weibull_median_ttf_days']['mean']:+.2f} d`",
        f"- Mean Weibull 30d failure-probability delta: `{summary['delta_weibull_fail_prob_30d']['mean']:+.4f}`",
        "",
        "## Term Summary",
        "",
    ]
    for row in term_summary.itertuples(index=False):
        report_lines.append(
            f"- `{row.term_name}` ({row.column}, {row.transform}): "
            f"mean stress delta `{row.mean_delta_weighted_stress:+.4f}`, "
            f"median `{row.median_delta_weighted_stress:+.4f}`, "
            f"improved wells `{row.wells_improved}`, worsened wells `{row.wells_worsened}`"
        )
    if not dominant_counts.empty:
        report_lines.extend(["", "## Dominant Improving Term Counts", ""])
        for row in dominant_counts.itertuples(index=False):
            report_lines.append(f"- `{row.term_name}`: `{row.n_wells}` wells")
    (output_dir / "vt_60hz_weibull_decomposition_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote decomposition outputs to: {output_dir}")
    print(f"Portfolio rows analyzed: {len(well_df)}")


if __name__ == "__main__":
    main()
