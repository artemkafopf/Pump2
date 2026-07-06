from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for candidate in (str(REPO_ROOT), str(BACKEND_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from analysis import predict_regularized_logistic_from_table
from scripts.analyze_field_mode_multivariate import (
    augment_model_features,
    load_feature_table,
    load_population_posteriors,
    merge_features_with_posteriors,
    resolve_feature_table,
    resolve_posterior_root,
    _numeric,
)


DEFAULT_POPULATION = "vt"
DEFAULT_MODEL_NAME = "failure_only_with_category_interactions"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Vt latent short-mode risk surfaces over mount_year and gypsum proxy."
    )
    parser.add_argument(
        "--posterior-root",
        default=None,
        help="Directory with posterior bundles. Defaults to latest bayesian_field_survival_compare* folder.",
    )
    parser.add_argument(
        "--feature-table",
        default=None,
        help="Path to analysis_dataset.csv. Defaults to latest vt_60hz_prompt_analysis_*/tables/analysis_dataset.csv.",
    )
    parser.add_argument(
        "--population",
        default=DEFAULT_POPULATION,
        help="Population to plot. Default: vt.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name from the multivariate fit outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <posterior-root>/<population>/multivariate_short_mode/<model-name>/surface.",
    )
    parser.add_argument(
        "--year-points",
        type=int,
        default=100,
        help="Number of grid points along the mount_year axis.",
    )
    parser.add_argument(
        "--gypsum-points",
        type=int,
        default=100,
        help="Number of grid points along the gypsum axis.",
    )
    parser.add_argument(
        "--gypsum-q-low",
        type=float,
        default=0.10,
        help="Lower quantile for positive gypsum proxy range.",
    )
    parser.add_argument(
        "--gypsum-q-high",
        type=float,
        default=0.95,
        help="Upper quantile for positive gypsum proxy range.",
    )
    return parser.parse_args()


def load_multivariate_coefficient_table(posterior_root: Path, population: str, model_name: str) -> pd.DataFrame:
    path = posterior_root / "field_mode_multivariate_coefficients.csv"
    frame = pd.read_csv(path)
    subset = frame.loc[(frame["population"] == population) & (frame["model_name"] == model_name)].copy()
    if subset.empty:
        raise ValueError(f"No coefficient rows found for population={population}, model={model_name}")
    return subset.reset_index(drop=True)


def select_model_frame(merged: pd.DataFrame, population: str, model_name: str) -> pd.DataFrame:
    frame = merged.loc[merged["population"] == population].copy()
    if "failure_only" in model_name:
        frame = frame.loc[_numeric(frame["event_model"]).eq(1)].copy()
    if frame.empty:
        raise ValueError(f"No rows available for population={population}, model={model_name}")
    return frame


def _mode_string(series: pd.Series) -> str:
    clean = series.fillna("<missing>").astype(str)
    if clean.empty:
        return "<missing>"
    return str(clean.mode(dropna=False).iloc[0])


def build_baseline_scenarios(model_frame: pd.DataFrame) -> pd.DataFrame:
    contractor = _mode_string(model_frame["contractor"]) if "contractor" in model_frame.columns else "<missing>"
    pump_family = _mode_string(model_frame["pump_family"]) if "pump_family" in model_frame.columns else "<missing>"
    failure_category = _mode_string(model_frame["failure_category_raw"]) if "failure_category_raw" in model_frame.columns else "<missing>"

    mount_year = float(_numeric(model_frame["mount_year"]).median()) if "mount_year" in model_frame.columns else np.nan
    avg_kpod = float(_numeric(model_frame["avg_kpod"]).median()) if "avg_kpod" in model_frame.columns else np.nan
    freq_above = float(_numeric(model_frame["freq_above_55hz_pct"]).median()) if "freq_above_55hz_pct" in model_frame.columns else np.nan
    freq_below = float(_numeric(model_frame["freq_below_45hz_pct"]).median()) if "freq_below_45hz_pct" in model_frame.columns else np.nan
    freq_signed = float(_numeric(model_frame["freq_signed_exposure"]).median()) if "freq_signed_exposure" in model_frame.columns else np.nan
    freq_w_mean = float(_numeric(model_frame["freq_w_mean"]).median()) if "freq_w_mean" in model_frame.columns else np.nan
    tlf_per_day = float(_numeric(model_frame["tlf_per_day"]).median()) if "tlf_per_day" in model_frame.columns else np.nan
    calcium = float(_numeric(model_frame["calcium_load_per_day"]).median()) if "calcium_load_per_day" in model_frame.columns else np.nan
    chloride = float(_numeric(model_frame["chloride_load_per_day"]).median()) if "chloride_load_per_day" in model_frame.columns else np.nan
    sulfate = float(_numeric(model_frame["sulfate_load_per_day"]).median()) if "sulfate_load_per_day" in model_frame.columns else np.nan
    gypsum = float(_numeric(model_frame["gypsum_proxy_per_day"]).median()) if "gypsum_proxy_per_day" in model_frame.columns else np.nan

    low_mask = _numeric(model_frame.get("high_h2s_flag_numeric", model_frame.get("high_h2s_flag"))).fillna(0.0) < 0.5
    high_mask = _numeric(model_frame.get("high_h2s_flag_numeric", model_frame.get("high_h2s_flag"))).fillna(0.0) >= 0.5
    low_h2s_value = float(_numeric(model_frame.loc[low_mask, "h2s_effective_mg_l"]).median()) if low_mask.any() else 0.0
    high_h2s_value = float(_numeric(model_frame.loc[high_mask, "h2s_effective_mg_l"]).median()) if high_mask.any() else max(low_h2s_value, 3.0)
    if not np.isfinite(low_h2s_value):
        low_h2s_value = 0.0
    if not np.isfinite(high_h2s_value):
        high_h2s_value = max(low_h2s_value, 3.0)

    return pd.DataFrame(
        [
            {
                "scenario": "low_h2s",
                "high_h2s_flag": False,
                "h2s_effective_mg_l": low_h2s_value,
                "mount_year": mount_year,
                "avg_kpod": avg_kpod,
                "freq_above_55hz_pct": freq_above,
                "freq_below_45hz_pct": freq_below,
                "freq_signed_exposure": freq_signed,
                "freq_w_mean": freq_w_mean,
                "tlf_per_day": tlf_per_day,
                "calcium_load_per_day": calcium,
                "chloride_load_per_day": chloride,
                "sulfate_load_per_day": sulfate,
                "gypsum_proxy_per_day": gypsum,
                "contractor": contractor,
                "pump_family": pump_family,
                "failure_category_raw": failure_category,
            },
            {
                "scenario": "high_h2s",
                "high_h2s_flag": True,
                "h2s_effective_mg_l": high_h2s_value,
                "mount_year": mount_year,
                "avg_kpod": avg_kpod,
                "freq_above_55hz_pct": freq_above,
                "freq_below_45hz_pct": freq_below,
                "freq_signed_exposure": freq_signed,
                "freq_w_mean": freq_w_mean,
                "tlf_per_day": tlf_per_day,
                "calcium_load_per_day": calcium,
                "chloride_load_per_day": chloride,
                "sulfate_load_per_day": sulfate,
                "gypsum_proxy_per_day": gypsum,
                "contractor": contractor,
                "pump_family": pump_family,
                "failure_category_raw": failure_category,
            },
        ]
    )


def build_surface_grid(
    baseline_scenarios: pd.DataFrame,
    model_frame: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    year_points: int,
    gypsum_points: int,
    gypsum_q_low: float,
    gypsum_q_high: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    year_series = _numeric(model_frame["mount_year"]).dropna()
    gypsum_series = _numeric(model_frame["gypsum_proxy_per_day"]).dropna()
    gypsum_positive = gypsum_series.loc[gypsum_series > 0.0]
    if year_series.empty or gypsum_positive.empty:
        raise ValueError("Not enough mount_year or positive gypsum data to build the surface.")

    year_grid = np.linspace(float(year_series.min()), float(year_series.max()), int(year_points))
    gypsum_low = float(gypsum_positive.quantile(float(gypsum_q_low)))
    gypsum_high = float(gypsum_positive.quantile(float(gypsum_q_high)))
    gypsum_low = max(gypsum_low, float(gypsum_positive.min()))
    gypsum_high = max(gypsum_high, gypsum_low * 1.01)
    gypsum_grid = np.geomspace(gypsum_low, gypsum_high, int(gypsum_points))

    rows: list[dict[str, object]] = []
    for scenario_row in baseline_scenarios.to_dict(orient="records"):
        for mount_year in year_grid:
            for gypsum_proxy in gypsum_grid:
                row = dict(scenario_row)
                row["mount_year"] = float(mount_year)
                row["gypsum_proxy_per_day"] = float(gypsum_proxy)
                rows.append(row)

    surface = pd.DataFrame(rows)
    scored = predict_regularized_logistic_from_table(
        augment_model_features(surface.copy()),
        coefficient_table,
    )
    surface["predicted_probability"] = scored["predicted_probability"].to_numpy(dtype=float)
    surface["linear_predictor"] = scored["linear_predictor"].to_numpy(dtype=float)
    return surface, baseline_scenarios


def build_threshold_frame(surface: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario, mount_year), subset in surface.groupby(["scenario", "mount_year"], dropna=False):
        ordered = subset.sort_values("gypsum_proxy_per_day")
        crossed = ordered.loc[ordered["predicted_probability"] >= threshold]
        rows.append(
            {
                "scenario": scenario,
                "mount_year": float(mount_year),
                "threshold_probability": threshold,
                "gypsum_proxy_at_threshold": float(crossed["gypsum_proxy_per_day"].iloc[0]) if not crossed.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_surface(surface: pd.DataFrame, output_path: Path, title: str) -> None:
    scenarios = ["low_h2s", "high_h2s"]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5), constrained_layout=True, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    levels = np.linspace(0.0, 1.0, 21)
    contour = None
    for axis, scenario in zip(axes, scenarios, strict=False):
        subset = surface.loc[surface["scenario"] == scenario].copy()
        pivot = subset.pivot(index="gypsum_proxy_per_day", columns="mount_year", values="predicted_probability").sort_index()
        X, Y = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
        Z = pivot.to_numpy(dtype=float)
        contour = axis.contourf(X, Y, Z, levels=levels, cmap="viridis", vmin=0.0, vmax=1.0)
        lines = axis.contour(X, Y, Z, levels=[0.25, 0.5, 0.75], colors="white", linewidths=0.9)
        axis.clabel(lines, fmt="%.2f", fontsize=8)
        axis.set_yscale("log")
        axis.set_xlabel("Mount year")
        axis.set_title(scenario.replace("_", " ").title())
        axis.grid(alpha=0.15)

    axes[0].set_ylabel("Gypsum proxy per day")
    fig.suptitle(title)
    colorbar = fig.colorbar(contour, ax=axes, shrink=0.95)
    colorbar.set_label("Predicted short-mode probability")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    posterior_root = resolve_posterior_root(args.posterior_root)
    feature_table_path = resolve_feature_table(args.feature_table)
    population = str(args.population).lower()
    model_name = str(args.model_name)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else posterior_root / population / "multivariate_short_mode" / model_name / "surface"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_feature_table(feature_table_path)
    posteriors = load_population_posteriors(posterior_root)
    merged = augment_model_features(merge_features_with_posteriors(features, posteriors))
    model_frame = select_model_frame(merged, population=population, model_name=model_name)
    coefficient_table = load_multivariate_coefficient_table(posterior_root, population=population, model_name=model_name)
    baseline_scenarios = build_baseline_scenarios(model_frame)
    surface, baseline_scenarios = build_surface_grid(
        baseline_scenarios=baseline_scenarios,
        model_frame=model_frame,
        coefficient_table=coefficient_table,
        year_points=int(args.year_points),
        gypsum_points=int(args.gypsum_points),
        gypsum_q_low=float(args.gypsum_q_low),
        gypsum_q_high=float(args.gypsum_q_high),
    )
    thresholds = build_threshold_frame(surface, threshold=0.5)

    surface.to_csv(output_dir / "vt_mount_year_gypsum_surface.csv", index=False)
    baseline_scenarios.to_csv(output_dir / "vt_mount_year_gypsum_surface_baselines.csv", index=False)
    thresholds.to_csv(output_dir / "vt_mount_year_gypsum_surface_thresholds.csv", index=False)
    plot_surface(
        surface,
        output_path=output_dir / "vt_mount_year_gypsum_surface.png",
        title=f"{population.upper()} short-mode risk surface: {model_name}",
    )

    print(f"Posterior root: {posterior_root}")
    print(f"Feature table: {feature_table_path}")
    print(f"Outputs written to: {output_dir}")
    print("Baseline scenarios:")
    print(baseline_scenarios.to_string(index=False))
    print("\nThreshold summary (first rows):")
    print(thresholds.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
