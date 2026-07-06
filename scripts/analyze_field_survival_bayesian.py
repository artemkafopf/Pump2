from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from analysis.paths import results_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for candidate in (str(REPO_ROOT), str(BACKEND_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from analysis import BayesianLatentWeibullConfig, fit_bayesian_latent_weibull
from scripts.analyze_kpod_window_thresholds import ALL_PATH, load_runs


_SLUG = "bayesian_field_survival"
DEFAULT_HORIZONS = (90.0, 180.0, 365.0, 730.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Global, Ya, and Vt survival using the Bayesian latent Weibull mixture.")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs.")
    parser.add_argument("--n-iter", type=int, default=1000, help="Total MCMC iterations.")
    parser.add_argument("--burn-in", type=int, default=400, help="Burn-in iterations.")
    parser.add_argument("--thin", type=int, default=4, help="Thinning interval.")
    parser.add_argument("--chains", type=int, default=1, help="Number of chains.")
    parser.add_argument("--k-max", type=int, default=4, help="Maximum K for unknown-K fitting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _prepare_base_runs() -> pd.DataFrame:
    runs = load_runs(ALL_PATH).copy()
    base = runs[["row_id", "Месторождение", "Наработка (сут)", "Failure Flag"]].copy()
    base = base.rename(
        columns={
            "row_id": "id",
            "Месторождение": "field",
            "Наработка (сут)": "duration",
            "Failure Flag": "event",
        }
    )
    return base


def _fit_population(name: str, df: pd.DataFrame, config: BayesianLatentWeibullConfig, output_dir: Path) -> dict[str, object]:
    result = fit_bayesian_latent_weibull(
        df,
        duration_column="duration",
        event_column="event",
        id_column="id",
        config=config,
    )
    result.export_bundle(output_dir)

    summary = result.posterior_summary_frame()
    summary.to_csv(output_dir / "posterior_summary_with_population.csv", index=False)
    k_frame = result.posterior_k_frame()
    k_frame.to_csv(output_dir / "posterior_k_with_population.csv", index=False)
    survival = result.posterior_survival_frame(max_time=1000, num_points=201)
    survival.to_csv(output_dir / "posterior_survival_with_population.csv", index=False)
    life_quantiles = result.posterior_life_quantiles_frame()
    life_quantiles.to_csv(output_dir / "posterior_life_quantiles.csv", index=False)

    payload: dict[str, object] = {
        "population": name,
        "rows": int(len(df)),
        "failures": int(df["event"].sum()),
        "censored": int((df["event"] == 0).sum()),
        "mean_K": float(np.mean(result.k_draws)),
        "mode_K": int(pd.Series(result.k_draws).value_counts().idxmax()),
        "p_K_eq_1": float(np.mean(result.k_draws == 1)),
        "p_K_eq_2": float(np.mean(result.k_draws == 2)),
        "p_K_ge_3": float(np.mean(result.k_draws >= 3)),
    }

    for component_index in range(1, result.max_components + 1):
        component_name = f"component_{component_index}"
        for parameter in ("weight", "eta", "beta", "median_life"):
            chunk = summary.loc[
                (summary["component"] == component_name)
                & (summary["parameter"] == parameter)
            ]
            if chunk.empty:
                continue
            payload[f"{component_name}_{parameter}_mean"] = float(chunk["mean"].iloc[0])
            payload[f"{component_name}_{parameter}_q50"] = float(chunk["q50"].iloc[0])
            payload[f"{component_name}_{parameter}_presence"] = float(chunk["presence_probability"].iloc[0])

    for label in ("B10", "B25", "B50", "B90"):
        chunk = life_quantiles.loc[life_quantiles["label"] == label]
        if chunk.empty:
            continue
        payload[f"{label}_mean"] = float(chunk["mean"].iloc[0])
        payload[f"{label}_q50"] = float(chunk["q50"].iloc[0])

    active = result.active_unit_predictions()
    if not active.empty:
        payload["active_next30_mean_avg"] = float(active["next_30_day_failure_mean"].mean())
        payload["active_next90_mean_avg"] = float(active["next_90_day_failure_mean"].mean())
        valid_remaining = active["median_remaining_life"].replace([np.inf, -np.inf], np.nan).dropna()
        payload["active_remaining_life_median_avg"] = float(valid_remaining.mean()) if not valid_remaining.empty else float("nan")

    return {
        "result": result,
        "summary_row": payload,
        "k_frame": k_frame.assign(population=name),
        "survival_frame": survival.assign(population=name),
    }


def build_markdown_report(summary_df: pd.DataFrame, pairwise: pd.DataFrame) -> str:
    summary_by_pop = {row["population"]: row for row in summary_df.to_dict(orient="records")}

    lines = [
        "# Bayesian Field Survival Comparison",
        "",
        "## Populations",
        "",
    ]
    for population in ("Global", "Ya", "Vt"):
        row = summary_by_pop[population]
        lines.append(
            f"- `{population}`: rows `{row['rows']}`, failures `{row['failures']}`, "
            f"mode K `{row['mode_K']}`, B50 `{row['B50_q50']:.1f}` d, "
            f"avg next-90d risk among active units `{row['active_next90_mean_avg']:.3f}`."
        )

    vt = summary_by_pop["Vt"]
    ya = summary_by_pop["Ya"]
    global_row = summary_by_pop["Global"]
    lines.extend(
        [
            "",
            "## Main Differences",
            "",
            f"- `Vt` is the shortest-lived population: B50 `{vt['B50_q50']:.1f}` d versus `{ya['B50_q50']:.1f}` d for `Ya` and `{global_row['B50_q50']:.1f}` d globally.",
            f"- `Vt` also has the highest early and active-unit risk: B10 `{vt['B10_q50']:.1f}` d and active next-90d risk `{vt['active_next90_mean_avg']:.3f}`.",
            f"- `Ya` is the longest-lived population, driven by a larger long-life component: component 2 eta `{ya['component_2_eta_mean']:.1f}` d versus `{vt['component_2_eta_mean']:.1f}` d for `Vt`.",
            f"- `Vt` has much more mass in the short-life component: component 1 weight `{vt['component_1_weight_mean']:.3f}` versus `{ya['component_1_weight_mean']:.3f}` in `Ya` and `{global_row['component_1_weight_mean']:.3f}` globally.",
            "",
            "## Horizon Snapshot",
            "",
        ]
    )

    for horizon in (90.0, 180.0, 365.0, 730.0):
        horizon_rows = pairwise.loc[(pairwise["time"] - horizon).abs() <= 5.0].copy()
        if horizon_rows.empty:
            continue
        row = horizon_rows.iloc[0]
        lines.append(
            f"- Around `{horizon:.0f}` d: `S_Vt - S_Ya = {row['Vt_minus_Ya']:.3f}`, "
            f"`S_Vt - S_Global = {row['Vt_minus_Global']:.3f}`, "
            f"`S_Ya - S_Global = {row['Ya_minus_Global']:.3f}`."
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else results_dir(_SLUG)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _prepare_base_runs()
    populations = {
        "Global": base.copy(),
        "Ya": base.loc[base["field"].astype("string") == "Ya"].copy(),
        "Vt": base.loc[base["field"].astype("string") == "Vt"].copy(),
    }
    config = BayesianLatentWeibullConfig(
        n_components=2,
        use_unknown_k=True,
        k_max=int(args.k_max),
        lambda_k=3.0,
        birth_death_probability=0.25,
        n_iter=int(args.n_iter),
        burn_in=int(args.burn_in),
        thin=int(args.thin),
        n_chains=int(args.chains),
        proposal_sd_log_eta=0.08,
        proposal_sd_log_beta=0.06,
        random_seed=int(args.seed),
    )

    fit_outputs: list[dict[str, object]] = []
    for population_name, subset in populations.items():
        print(f"Fitting {population_name}: rows={len(subset)}, failures={int(subset['event'].sum())}")
        population_output = output_dir / population_name.lower()
        population_output.mkdir(parents=True, exist_ok=True)
        fit_outputs.append(_fit_population(population_name, subset, config, population_output))

    summary_df = pd.DataFrame([item["summary_row"] for item in fit_outputs])
    summary_df.to_csv(output_dir / "field_survival_comparison_summary.csv", index=False)

    k_frame = pd.concat([item["k_frame"] for item in fit_outputs], ignore_index=True)
    k_frame.to_csv(output_dir / "field_survival_k_distribution.csv", index=False)

    curves = pd.concat([item["survival_frame"] for item in fit_outputs], ignore_index=True)
    curves.to_csv(output_dir / "field_survival_curves.csv", index=False)

    pairwise = curves.pivot(index="time", columns="population", values="survival_mean").reset_index()
    for left, right in (("Vt", "Ya"), ("Vt", "Global"), ("Ya", "Global")):
        if left in pairwise.columns and right in pairwise.columns:
            pairwise[f"{left}_minus_{right}"] = pairwise[left] - pairwise[right]
    pairwise.to_csv(output_dir / "field_survival_pairwise_differences.csv", index=False)

    report = {
        "settings": {
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "chains": args.chains,
            "k_max": args.k_max,
            "seed": args.seed,
        },
        "summary": summary_df.to_dict(orient="records"),
    }
    (output_dir / "field_survival_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "field_survival_report.md").write_text(build_markdown_report(summary_df, pairwise), encoding="utf-8")

    print(f"Outputs written to: {output_dir}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
