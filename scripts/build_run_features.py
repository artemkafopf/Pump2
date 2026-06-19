from __future__ import annotations

import argparse
import json
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
from analysis.modeling_config import (
    DEFAULT_GLF_THRESHOLD,
    DEFAULT_INFANT_MORTALITY_DAYS,
    DEFAULT_MATURE_WINDOW_DAYS,
    DEFAULT_MIN_MATURE_OBS,
    assert_no_explanatory_features,
)
from scripts.analyze_failure_horizon import ALL_PATH, load_runs
from scripts.data_utils import _numeric, add_dynamic_salt_proxies, load_daily_merged, source_coverage_report


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "run_features_phase1"


def _normalize_pump_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    replacements = {
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return "".join(character for character in text if character.isalnum())


def pump_family(value: object) -> str:
    normalized = _normalize_pump_text(value)
    if not normalized:
        return "<missing>"
    for prefix in ("MT5A", "5A", "GN", "SN", "DN"):
        if normalized.startswith(prefix):
            return prefix
    return normalized[:8] if normalized else "<missing>"


def season_of_install(value: pd.Timestamp | object) -> str | None:
    timestamp = pd.Timestamp(value) if not pd.isna(value) else pd.NaT
    if pd.isna(timestamp):
        return None
    month = int(timestamp.month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def split_infant_vs_mature_runs(
    runs: pd.DataFrame,
    infant_mortality_days: int = DEFAULT_INFANT_MORTALITY_DAYS,
    mature_window_days: int = DEFAULT_MATURE_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    working = runs.copy()
    duration = _numeric(working["run_days"])
    working["pump_family"] = working.get("Тип УЭЦН", pd.Series(index=working.index)).map(pump_family)

    infant_mask = duration < infant_mortality_days
    gap_mask = duration.ge(infant_mortality_days) & duration.lt(infant_mortality_days + mature_window_days)
    mature_mask = duration.ge(infant_mortality_days + mature_window_days)

    infant_runs = working.loc[infant_mask].copy().reset_index(drop=True)
    gap_runs = working.loc[gap_mask].copy().reset_index(drop=True)
    mature_runs = working.loc[mature_mask].copy().reset_index(drop=True)

    def counts_by(column: str, frame: pd.DataFrame) -> dict[str, int]:
        if column not in frame.columns or frame.empty:
            return {}
        return {str(key): int(value) for key, value in frame[column].astype("string").fillna("<missing>").value_counts().items()}

    summary = {
        "total_runs": int(len(working)),
        "infant_mortality_runs": int(len(infant_runs)),
        "infant_mortality_rate": float(len(infant_runs) / max(len(working), 1)),
        "gap_population_runs": int(len(gap_runs)),
        "gap_population_rate": float(len(gap_runs) / max(len(working), 1)),
        "mature_runs": int(len(mature_runs)),
        "mature_rate": float(len(mature_runs) / max(len(working), 1)),
        "infant_mortality_by_field": counts_by("Месторождение", infant_runs),
        "infant_mortality_by_contractor": counts_by("Принадлежность", infant_runs),
        "infant_mortality_by_pump_family": counts_by("pump_family", infant_runs),
        "gap_by_field": counts_by("Месторождение", gap_runs),
        "median_infant_duration_days": None if infant_runs.empty else float(_numeric(infant_runs["run_days"]).median()),
        "median_gap_duration_days": None if gap_runs.empty else float(_numeric(gap_runs["run_days"]).median()),
    }
    return mature_runs, infant_runs, gap_runs, summary


def _run_number_at_well(runs: pd.DataFrame) -> pd.Series:
    ordered = runs.sort_values(["Скв.", "Дата монтажа", "Дата остановки", "row_id"]).copy()
    ordered["run_number_at_well"] = ordered.groupby("Скв.").cumcount() + 1
    return ordered.set_index("row_id")["run_number_at_well"]


def _h2s_proxy_frame(runs: pd.DataFrame) -> pd.DataFrame:
    working = runs.copy()
    working["well_id"] = working["Скв."].astype("string").str.strip()
    working["field"] = working.get("Месторождение", pd.Series(index=working.index, dtype="object")).astype("string").str.strip()
    working["pad"] = working.get("Куст", pd.Series(index=working.index, dtype="object")).astype("string").str.strip()
    working["h2s_numeric"] = _numeric(working.get("Массовая доля сероводорода, мг/дм³"))

    well_lookup = (
        working.loc[working["well_id"].notna()]
        .groupby("well_id", dropna=False)["h2s_numeric"]
        .median()
        .rename("h2s_well_median")
        .reset_index()
    )
    pad_lookup = (
        working.loc[working["field"].notna() & working["pad"].notna()]
        .groupby(["field", "pad"], dropna=False)["h2s_numeric"]
        .median()
        .rename("h2s_pad_median")
        .reset_index()
    )

    merged = (
        working[["row_id", "well_id", "field", "pad"]]
        .drop_duplicates(subset=["row_id"])
        .merge(well_lookup, on="well_id", how="left")
        .merge(pad_lookup, on=["field", "pad"], how="left")
    )
    merged["h2s_proxy_mg_l"] = merged["h2s_well_median"].combine_first(merged["h2s_pad_median"])
    merged["h2s_proxy_source"] = np.where(
        merged["h2s_well_median"].notna(),
        "well_median",
        np.where(merged["h2s_pad_median"].notna(), "pad_median", "missing"),
    )
    return merged[["row_id", "pad", "h2s_well_median", "h2s_pad_median", "h2s_proxy_mg_l", "h2s_proxy_source"]]


def _static_feature_frame(runs: pd.DataFrame) -> pd.DataFrame:
    working = runs.copy()
    working["well_id"] = working["Скв."].astype("string").str.strip()
    working["duration_days"] = _numeric(working["run_days"])
    working["pump_family"] = working["Тип УЭЦН"].map(pump_family)
    run_numbers = _run_number_at_well(working)
    working["run_number_at_well"] = working["row_id"].map(run_numbers)
    working["season_of_install"] = working["Дата монтажа"].map(season_of_install)
    h2s_proxy = _h2s_proxy_frame(working).set_index("row_id")
    static = pd.DataFrame(
        {
            "row_id": working["row_id"],
            "well_id": working["well_id"],
            "event": working["event"],
            "duration_days": working["duration_days"],
            "install_date": working["Дата монтажа"],
            "stop_date": working["Дата остановки"],
            "field": working.get("Месторождение"),
            "pad": working.get("Куст"),
            "contractor": working.get("Принадлежность"),
            "pump_type": working.get("Тип УЭЦН"),
            "pump_family": working["pump_family"],
            "nominal_qliq": _numeric(working.get("Ном. Произв. м₃/сут")),
            "nominal_head_50hz": _numeric(working.get("Ном.напор (50Гц)")),
            "nominal_freq": _numeric(working.get("Номинальная частота, Гц")),
            "motor_power_kw": _numeric(working.get("Мощность, кВт")),
            "stages": _numeric(working.get("Кол.ступеней")),
            "submergence_depth_m": _numeric(working.get("Глубина спуска УЭЦН, по НКТ")),
            "pbubble": _numeric(working.get("Дав. Нас")),
            "curvature_flag": working.get("Работа в кривизне"),
            "h2s_well_median_mg_l": working["row_id"].map(h2s_proxy["h2s_well_median"]),
            "h2s_pad_median_mg_l": working["row_id"].map(h2s_proxy["h2s_pad_median"]),
            "h2s_proxy_mg_l": working["row_id"].map(h2s_proxy["h2s_proxy_mg_l"]),
            "h2s_proxy_source": working["row_id"].map(h2s_proxy["h2s_proxy_source"]),
            "run_number_at_well": _numeric(working["run_number_at_well"]),
            "season_of_install": working["season_of_install"],
        }
    )
    return static


def _daily_run_frame(daily_df: pd.DataFrame, run_row: dict[str, object]) -> pd.DataFrame:
    interval = daily_df.loc[
        (daily_df["well_id"].astype("string") == str(run_row["Скв."]).strip())
        & (daily_df["dt"] >= run_row["Дата монтажа"])
        & (daily_df["dt"] <= run_row["Дата остановки"])
    ].copy()
    if interval.empty:
        return interval

    nominal_qliq = _numeric(pd.Series([run_row.get("Ном. Произв. м₃/сут")])).iloc[0]
    nominal_freq = _numeric(pd.Series([run_row.get("Номинальная частота, Гц")])).iloc[0]
    pbubble = _numeric(pd.Series([run_row.get("Дав. Нас")])).iloc[0]

    interval["kpod_daily"] = np.nan
    if pd.notna(nominal_qliq) and abs(float(nominal_qliq)) > 1e-12:
        interval["kpod_daily"] = _numeric(interval["qliq"]) / float(nominal_qliq)

    interval["kpod_freq_daily"] = np.nan
    if pd.notna(nominal_freq):
        freq = _numeric(interval["freq"])
        valid = interval["kpod_daily"].notna() & freq.notna() & (freq.abs() > 1e-12)
        interval.loc[valid, "kpod_freq_daily"] = interval.loc[valid, "kpod_daily"] * (float(nominal_freq) / freq.loc[valid])

    interval["pressure_ratio_daily"] = np.nan
    if pd.notna(pbubble) and abs(float(pbubble)) > 1e-12:
        interval["pressure_ratio_daily"] = _numeric(interval["rzab"]) / float(pbubble)

    interval["neg_excess_kpod_daily"] = np.maximum(0.8 - _numeric(interval["kpod_daily"]), 0.0)
    interval["glf_excess_daily"] = np.maximum(_numeric(interval["gas_factor"]) - DEFAULT_GLF_THRESHOLD, 0.0)
    return interval


def _aggregate_window_features(source_df: pd.DataFrame, prefix: str, glf_threshold: float) -> dict[str, float]:
    result: dict[str, float] = {}
    if source_df.empty:
        return result
    kpod = _numeric(source_df.get("kpod_daily", pd.Series(dtype=float)))
    kpod_freq = _numeric(source_df.get("kpod_freq_daily", pd.Series(dtype=float)))
    pressure_ratio = _numeric(source_df.get("pressure_ratio_daily", pd.Series(dtype=float)))
    gas_factor = _numeric(source_df.get("gas_factor", pd.Series(dtype=float)))
    load = _numeric(source_df.get("load", pd.Series(dtype=float)))
    freq = _numeric(source_df.get("freq", pd.Series(dtype=float)))
    neg_excess = _numeric(source_df.get("neg_excess_kpod_daily", pd.Series(dtype=float)))
    glf_excess = _numeric(source_df.get("glf_excess_daily", pd.Series(dtype=float)))
    salt_load = _numeric(source_df.get("daily_salt_load_kg", pd.Series(dtype=float)))
    gypsum_proxy = _numeric(source_df.get("daily_gypsum_scale_proxy", pd.Series(dtype=float)))

    if kpod.notna().any():
        result[f"kpod_{prefix}_mean"] = float(kpod.mean())
        result[f"frac_kpod_below_0p7_{prefix}"] = float((kpod < 0.7).mean())
    if kpod_freq.notna().any():
        result[f"kpod_freq_{prefix}_mean"] = float(kpod_freq.mean())
    if pressure_ratio.notna().any():
        result[f"pzab_over_pbubble_{prefix}_mean"] = float(pressure_ratio.mean())
        result[f"frac_pzab_below_1_{prefix}"] = float((pressure_ratio < 1.0).mean())
    if gas_factor.notna().any():
        result[f"glf_{prefix}_mean"] = float(gas_factor.mean())
        result[f"frac_glf_above_thr_{prefix}"] = float((gas_factor > glf_threshold).mean())
    if load.notna().any():
        result[f"load_{prefix}_mean"] = float(load.mean())
        result[f"load_{prefix}_std"] = float(load.std(ddof=0)) if len(load.dropna()) > 1 else 0.0
    if freq.notna().any():
        result[f"freq_{prefix}_mean"] = float(freq.mean())
    if neg_excess.notna().any():
        result[f"mean_neg_excess_kpod_{prefix}"] = float(neg_excess.mean())
    if salt_load.notna().any():
        result[f"salt_proxy_{prefix}_mean"] = float(salt_load.mean())
        result[f"integrated_salt_proxy_{prefix}"] = float(salt_load.sum())
    if gypsum_proxy.notna().any():
        result[f"gypsum_proxy_{prefix}_mean"] = float(gypsum_proxy.mean())
    if prefix == "w" and glf_excess.notna().any():
        result["integrated_glf_excess_w"] = float(glf_excess.sum())
    return result


def build_run_features(
    runs: pd.DataFrame,
    glf_threshold: float = DEFAULT_GLF_THRESHOLD,
    infant_mortality_days: int = DEFAULT_INFANT_MORTALITY_DAYS,
    mature_window_days: int = DEFAULT_MATURE_WINDOW_DAYS,
    min_mature_obs: int = DEFAULT_MIN_MATURE_OBS,
    include_whole_life: bool = True,
    predictive: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    mature_runs, infant_runs, gap_runs, summary = split_infant_vs_mature_runs(
        runs,
        infant_mortality_days=infant_mortality_days,
        mature_window_days=mature_window_days,
    )
    all_wells = runs["Скв."].dropna().astype(str).str.strip().unique().tolist()
    daily_merged = load_daily_merged(all_wells)
    daily_merged = add_dynamic_salt_proxies(daily_merged, qliq_column="qliq", fill_mode="step", extend_backward=True)
    coverage = source_coverage_report(daily_merged)

    mature_static = _static_feature_frame(mature_runs)
    rows: list[dict[str, object]] = []
    missing_mature_window_rows = 0
    missing_whole_life_rows = 0
    for run_row in mature_runs.to_dict(orient="records"):
        interval = _daily_run_frame(daily_merged, run_row)
        post_infant_start = run_row["Дата монтажа"] + pd.Timedelta(days=infant_mortality_days)
        mature_end = post_infant_start + pd.Timedelta(days=mature_window_days)
        mature_window = interval.loc[(interval["dt"] >= post_infant_start) & (interval["dt"] <= mature_end)].copy()
        whole_life = interval.loc[interval["dt"] >= post_infant_start].copy()

        feature_row = mature_static.loc[mature_static["row_id"].eq(int(run_row["row_id"]))].iloc[0].to_dict()
        feature_row["mature_obs_days"] = int(mature_window["dt"].nunique())
        feature_row["whole_life_obs_days"] = int(whole_life["dt"].nunique())

        if int(mature_window["dt"].nunique()) >= int(min_mature_obs):
            feature_row.update(_aggregate_window_features(mature_window, "m", glf_threshold))
        else:
            missing_mature_window_rows += 1

        if include_whole_life and int(whole_life["dt"].nunique()) >= int(min_mature_obs):
            feature_row.update(_aggregate_window_features(whole_life, "w", glf_threshold))
        elif include_whole_life:
            missing_whole_life_rows += 1

        rows.append(feature_row)

    mature_features = pd.DataFrame(rows)
    if predictive and not mature_features.empty:
        assert_no_explanatory_features(mature_features.columns.tolist(), predictive=True)

    summary["mature_feature_rows"] = int(len(mature_features))
    summary["mature_rows_missing_m_window"] = int(missing_mature_window_rows)
    summary["mature_rows_missing_w_window"] = int(missing_whole_life_rows)
    summary["mean_telemetry_fraction"] = None if coverage.empty else float(coverage["telemetry_fraction"].mean())
    summary["mature_wells"] = int(mature_features["well_id"].nunique()) if not mature_features.empty else 0
    summary["mature_rows_with_h2s_proxy"] = int(mature_features["h2s_proxy_mg_l"].notna().sum()) if "h2s_proxy_mg_l" in mature_features.columns else 0
    summary["mature_rows_h2s_from_well"] = int(mature_features["h2s_proxy_source"].astype("string").eq("well_median").sum()) if "h2s_proxy_source" in mature_features.columns else 0
    summary["mature_rows_h2s_from_pad"] = int(mature_features["h2s_proxy_source"].astype("string").eq("pad_median").sum()) if "h2s_proxy_source" in mature_features.columns else 0
    summary["mature_rows_with_salt_proxy_m"] = int(mature_features["salt_proxy_m_mean"].notna().sum()) if "salt_proxy_m_mean" in mature_features.columns else 0
    summary["mature_rows_with_salt_proxy_w"] = int(mature_features["salt_proxy_w_mean"].notna().sum()) if "salt_proxy_w_mean" in mature_features.columns else 0
    return mature_features, infant_runs, gap_runs, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 1 mature-life run features.")
    parser.add_argument("--input", default=str(resolve_v03_all_path()), help="Path to the run-level workbook.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/JSON outputs.")
    parser.add_argument("--predictive", action="store_true", help="Build predictive-safe feature table (reject _w_ features).")
    parser.add_argument("--no-whole-life", action="store_true", help="Do not include explanatory whole-life features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(Path(args.input))
    mature_features, infant_runs, gap_runs, summary = build_run_features(
        runs,
        include_whole_life=not args.no_whole_life,
        predictive=bool(args.predictive),
    )
    daily_coverage = source_coverage_report(load_daily_merged(runs["Скв."].dropna().astype(str).tolist()))

    mature_features.to_csv(output_dir / "mature_run_features.csv", index=False, encoding="utf-8-sig")
    infant_runs.to_csv(output_dir / "infant_runs.csv", index=False, encoding="utf-8-sig")
    gap_runs.to_csv(output_dir / "gap_runs.csv", index=False, encoding="utf-8-sig")
    daily_coverage.to_csv(output_dir / "source_coverage.csv", index=False, encoding="utf-8-sig")
    (output_dir / "infant_mortality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    coverage_rows = pd.DataFrame(
        [
            {"metric": "mature_feature_rows", "value": int(len(mature_features))},
            {"metric": "infant_runs", "value": int(len(infant_runs))},
            {"metric": "gap_runs", "value": int(len(gap_runs))},
            {"metric": "mature_rows_missing_m_window", "value": int(summary["mature_rows_missing_m_window"])},
            {"metric": "mature_rows_missing_w_window", "value": int(summary["mature_rows_missing_w_window"])},
        ]
    )
    coverage_rows.to_csv(output_dir / "mature_feature_coverage.csv", index=False, encoding="utf-8-sig")

    print(f"Run features written to: {output_dir}")
    print(f"Mature feature rows: {len(mature_features)}")
    print(f"Infant runs: {len(infant_runs)}")
    print(f"Gap runs: {len(gap_runs)}")


if __name__ == "__main__":
    main()
