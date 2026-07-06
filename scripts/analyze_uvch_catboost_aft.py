"""Train a CatBoost AFT model for UVCh wells and export scenario outputs."""

from __future__ import annotations

import ast
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in __import__("sys").path:
        __import__("sys").path.insert(0, _p)

from analysis.paths import results_dir

_SLUG = "uvch_catboost_aft"
_LEGACY_DOCS = REPO_ROOT / "archive" / "uvch_claude"  # moved from docs/ during cleanup
DB_PATH = REPO_ROOT / "data" / "warehouse" / "pump2.db"
ANALYSIS_PATH = _LEGACY_DOCS / "analysis.py"

OUT_DIR = results_dir(_SLUG)
OUT_CSV = OUT_DIR / "catboost_aft_uvch.csv"
OUT_MODEL = OUT_DIR / "catboost_aft_model.cbm"
OUT_PARAMS = OUT_DIR / "catboost_aft_params.json"
OUT_REPORT = OUT_DIR / "catboost_aft_report.md"
OUT_CALIBRATION = OUT_DIR / "catboost_aft_calibration.png"

TRAIN_FIELDS = ("Vt", "Za", "Az", "Ic")
CAT_FEATURES = ["field", "contractor", "pump_type"]
FEATURES = [
    "freq_w_mean",
    "freq_above_55hz_pct",
    "glf_m_mean",
    "load_m_mean",
    "kpod_m_mean",
    "frac_kpod_below_0p7_m",
    "h2s_proxy_mg_l",
    "gypsum_proxy_m_mean",
    "salt_proxy_m_mean",
    "nominal_flow_m3d",
    "motor_power_kw",
    "pbubble_atm",
    "run_number",
    "field",
    "contractor",
    "pump_type",
]
RANDOM_SEED = 42
TEST_FRACTION = 0.2
MISSING_CATEGORY = "__missing__"
BETA_BASE = 1.316
ETA_BASE = 240.3
BETA_VCH = 1.295
ETA_VCH = 232.5
M_T3 = 1.31


def _weibull_h(t: float, eta: float, beta: float) -> float:
    return (t / eta) ** beta


def h_base(t: float) -> float:
    return _weibull_h(t, ETA_BASE, BETA_BASE)


def h_vch(t: float) -> float:
    return _weibull_h(t, ETA_VCH, BETA_VCH)


def target_band(f_after: float, f_cur: float) -> str:
    if abs(f_after - f_cur) < 0.01:
        return "T0"
    if f_after < 55:
        return "T1"
    if f_after < 58:
        return "T2"
    return "T3"


def weibull_rul_q(q: float, age: float, h_fn, m: float = 1.0, max_horizon: int = 6000) -> float:
    lo, hi = 0.0, float(max_horizon)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        survival = math.exp(-m * (h_fn(age + mid) - h_fn(age)))
        if survival > q:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 1)


def band_model(band: str):
    if band in ("T0", "T1"):
        return h_base, 1.0
    if band == "T2":
        return h_vch, 1.0
    return h_base, M_T3


def load_uvch_raw() -> list[tuple[Any, ...]]:
    text = ANALYSIS_PATH.read_text(encoding="utf-8")
    start = text.index("RAW = [")
    end = text.index("]\n\n# ── Helper functions") + 1
    raw_literal = text[start + len("RAW = "):end]
    return ast.literal_eval(raw_literal)


def load_positive_mart() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM mart__vt_freq55 WHERE ttf_true_best_days > 0",
            conn,
            parse_dates=["install_date", "stop_date"],
        )
    finally:
        conn.close()

    df["kip"] = df["ttf_true_best_days"] / df["run_days"]
    df = df.sort_values(["well", "install_date", "stop_date", "row_id"], na_position="last").reset_index(drop=True)
    df["run_number"] = df.groupby("well").cumcount() + 1
    return df


def make_aft_labels(ttf: np.ndarray, event: np.ndarray) -> list[tuple[float, float]]:
    lower = ttf.astype(float)
    upper = np.where(event.astype(int) == 1, lower, -1.0)
    return list(zip(lower, upper))


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(index=df.index)
    for feature in FEATURES:
        if feature in CAT_FEATURES:
            prepared[feature] = df[feature].astype("string").fillna(MISSING_CATEGORY)
        else:
            prepared[feature] = pd.to_numeric(df[feature], errors="coerce")
    return prepared


def split_by_well(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wells = np.array(sorted(df["well"].dropna().astype(str).unique().tolist()))
    if len(wells) < 2:
        raise RuntimeError("Not enough distinct wells to create a hold-out split.")

    rng = np.random.default_rng(RANDOM_SEED)
    test_size = max(1, int(len(wells) * TEST_FRACTION))
    for _ in range(100):
        wells_test = set(rng.choice(wells, size=test_size, replace=False).tolist())
        df_train = df[~df["well"].isin(wells_test)].copy()
        df_test = df[df["well"].isin(wells_test)].copy()
        if not df_train.empty and not df_test.empty and int(df_train["event"].sum()) > 0 and int(df_test["event"].sum()) > 0:
            return df_train, df_test

    raise RuntimeError("Failed to find a well-level hold-out split with events in both train and test.")


def build_pool(df: pd.DataFrame) -> tuple[Pool, pd.DataFrame]:
    features = prepare_feature_frame(df)
    cat_indices = [features.columns.get_loc(column) for column in CAT_FEATURES]
    pool = Pool(
        data=features,
        label=make_aft_labels(df["ttf_true_best_days"].to_numpy(dtype=float), df["event"].to_numpy(dtype=int)),
        cat_features=cat_indices,
    )
    return pool, features


def make_model(*, iterations: int = 1000, verbose: int | bool = 100) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="SurvivalAft:dist=Extreme",
        eval_metric="SurvivalAft:dist=Extreme",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        random_seed=RANDOM_SEED,
        verbose=verbose,
        allow_writing_files=False,
    )


def estimate_sigma_beta(model: CatBoostRegressor, df: pd.DataFrame) -> tuple[float, float]:
    train_events = df[df["event"] == 1].copy()
    if train_events.empty:
        return float("nan"), float("nan")
    mu = model.predict(prepare_feature_frame(train_events))
    residuals = np.log(train_events["ttf_true_best_days"].to_numpy(dtype=float)) - mu
    sigma_hat = float(np.std(residuals) * math.sqrt(6.0) / math.pi)
    beta_weibull = float(1.0 / sigma_hat) if sigma_hat > 0 else float("nan")
    return sigma_hat, beta_weibull


def conditional_rul(q_surv: float, age: float, eta_i: float, beta: float) -> float:
    if not np.isfinite(eta_i) or eta_i <= 0 or not np.isfinite(beta) or beta <= 0:
        return float("nan")
    h_age = (age / eta_i) ** beta
    delta_h = -math.log(q_surv)
    t_star = eta_i * (h_age + delta_h) ** (1.0 / beta)
    return max(t_star - age, 0.0)


def build_last_history_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty:
        return {}
    latest = (
        df.sort_values(["well", "install_date", "stop_date", "row_id"], na_position="last")
        .groupby("well", as_index=False)
        .tail(1)
    )
    return {str(row["well"]): row for _, row in latest.iterrows()}


def scenario_freq_above(history_pct: float, band: str) -> float:
    if band == "T0":
        return history_pct
    if band == "T1":
        return 0.10
    return 0.90


def safe_round(value: float | None, digits: int = 1) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def calibration_plot(model: CatBoostRegressor, df_test: pd.DataFrame) -> dict[str, Any]:
    vt = df_test[df_test["field"] == "Vt"].copy()
    summary: dict[str, Any] = {"vt_holdout_rows": int(len(vt)), "quintiles": []}

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(vt) < 10 or vt["event"].sum() == 0:
        ax.text(0.5, 0.5, "Not enough Vt hold-out data for calibration plot", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUT_CALIBRATION, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return summary

    vt["mu_pred"] = model.predict(prepare_feature_frame(vt))
    vt["eta_pred"] = np.exp(vt["mu_pred"])
    n_bins = min(5, int(vt["eta_pred"].nunique(dropna=True)))
    if n_bins < 2:
        ax.text(0.5, 0.5, "Predicted eta has too few unique values", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUT_CALIBRATION, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return summary

    vt["eta_bin"] = pd.qcut(vt["eta_pred"], q=n_bins, labels=False, duplicates="drop")
    kmf = KaplanMeierFitter()
    for bucket in sorted(vt["eta_bin"].dropna().unique()):
        group = vt[vt["eta_bin"] == bucket].copy()
        label = f"Q{int(bucket) + 1} (n={len(group)})"
        kmf.fit(group["ttf_true_best_days"], event_observed=group["event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False)
        median_eta = float(group["eta_pred"].median())
        ax.scatter([median_eta], [0.5], s=25, marker="o")
        summary["quintiles"].append(
            {
                "bin": int(bucket) + 1,
                "rows": int(len(group)),
                "median_pred_eta": median_eta,
                "km_median": float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None,
            }
        )

    ax.set_title("CatBoost AFT calibration by predicted eta quintile (Vt hold-out)")
    ax.set_xlabel("Time, days")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_CALIBRATION, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def build_feature_row(base_row: pd.Series, f_target: float, freq_above_55_target: float) -> pd.DataFrame:
    row = base_row.copy()
    row["freq_w_mean"] = f_target
    row["freq_above_55hz_pct"] = freq_above_55_target
    return prepare_feature_frame(pd.DataFrame([row]))


def score_uvch_wells(
    *,
    uvch_raw: list[tuple[Any, ...]],
    model: CatBoostRegressor,
    beta_weibull: float,
    train_lookup: dict[str, pd.Series],
    positive_lookup: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, nno, _q_liq, _q_oil, _glf, _gas_bef, _gas_aft, f_cur, _ml_cur, f_after, *_rest in uvch_raw:
        band = target_band(f_after, f_cur)
        h_aft, m_aft = band_model(band)
        weibull_rul_p50_c = weibull_rul_q(0.50, nno, h_base)
        weibull_rul_p50_a = weibull_rul_q(0.50, nno, h_aft, m_aft)

        if well in train_lookup:
            history = train_lookup[well]
            history_source = "train_filtered"
            fallback_reason = ""
        elif well in positive_lookup:
            history = positive_lookup[well]
            history_source = "mart_positive_ttf"
            fallback_reason = "well_missing_after_train_filters"
        else:
            history = None
            history_source = "missing"
            fallback_reason = "no_positive_ttf_history_in_mart"

        base: dict[str, Any] = {
            "well": well,
            "nno": int(nno),
            "f_cur": float(f_cur),
            "f_after": float(f_after),
            "band": band,
            "catboost_available": 0,
            "history_source": history_source,
            "fallback_reason": fallback_reason,
            "eta_cur": None,
            "rul_p10_c": None,
            "rul_p50_c": None,
            "rul_p90_c": None,
            "nno_forecast_c": None,
            "eta_aft": None,
            "rul_p10_a": None,
            "rul_p50_a": None,
            "rul_p90_a": None,
            "nno_forecast_a": None,
            "delta_rul_pct": None,
            "delta_nno_pct": None,
            "weibull_rul_p50_c": weibull_rul_p50_c,
            "weibull_rul_p50_a": weibull_rul_p50_a,
        }

        if history is None:
            rows.append(base)
            continue

        current_pct = pd.to_numeric(pd.Series([history.get("freq_above_55hz_pct")]), errors="coerce").iloc[0]
        scenario_pct = scenario_freq_above(current_pct, band)

        current_features = build_feature_row(history, float(f_cur), current_pct)
        after_features = build_feature_row(history, float(f_after), scenario_pct)
        eta_cur = float(math.exp(model.predict(current_features)[0]))
        eta_aft = float(math.exp(model.predict(after_features)[0]))

        rul_p10_c = conditional_rul(0.90, nno, eta_cur, beta_weibull)
        rul_p50_c = conditional_rul(0.50, nno, eta_cur, beta_weibull)
        rul_p90_c = conditional_rul(0.10, nno, eta_cur, beta_weibull)
        rul_p10_a = conditional_rul(0.90, nno, eta_aft, beta_weibull)
        rul_p50_a = conditional_rul(0.50, nno, eta_aft, beta_weibull)
        rul_p90_a = conditional_rul(0.10, nno, eta_aft, beta_weibull)
        nno_forecast_c = float(nno + rul_p50_c) if np.isfinite(rul_p50_c) else float("nan")
        nno_forecast_a = float(nno + rul_p50_a) if np.isfinite(rul_p50_a) else float("nan")

        if np.isfinite(rul_p50_c) and rul_p50_c != 0:
            delta_rul_pct = (rul_p50_a - rul_p50_c) / rul_p50_c * 100.0
        else:
            delta_rul_pct = float("nan")
        if np.isfinite(nno_forecast_c) and nno_forecast_c != 0:
            delta_nno_pct = (nno_forecast_a - nno_forecast_c) / nno_forecast_c * 100.0
        else:
            delta_nno_pct = float("nan")

        base.update(
            {
                "catboost_available": 1,
                "eta_cur": safe_round(eta_cur),
                "rul_p10_c": safe_round(rul_p10_c, 0),
                "rul_p50_c": safe_round(rul_p50_c, 0),
                "rul_p90_c": safe_round(rul_p90_c, 0),
                "nno_forecast_c": safe_round(nno_forecast_c, 0),
                "eta_aft": safe_round(eta_aft),
                "rul_p10_a": safe_round(rul_p10_a, 0),
                "rul_p50_a": safe_round(rul_p50_a, 0),
                "rul_p90_a": safe_round(rul_p90_a, 0),
                "nno_forecast_a": safe_round(nno_forecast_a, 0),
                "delta_rul_pct": safe_round(delta_rul_pct),
                "delta_nno_pct": safe_round(delta_nno_pct),
            }
        )
        rows.append(base)

    return pd.DataFrame(rows)


def mean_cohort_rul_check(model: CatBoostRegressor, beta_weibull: float, df: pd.DataFrame) -> dict[str, Any]:
    cohort = df[df["freq_w_mean"].between(50.0, 55.0, inclusive="both")].copy()
    if cohort.empty:
        return {"rows": 0, "mean_catboost_rul_p50_nno100": None}
    etas = np.exp(model.predict(prepare_feature_frame(cohort)))
    p50_values = [conditional_rul(0.50, 100.0, float(eta), beta_weibull) for eta in etas]
    valid = [value for value in p50_values if np.isfinite(value)]
    return {
        "rows": int(len(cohort)),
        "mean_catboost_rul_p50_nno100": float(np.mean(valid)) if valid else None,
        "cohort_weibull_reference": 141.0,
    }


def build_report(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    c_index: float | None,
    sigma_hat: float,
    beta_weibull: float,
    eval_best_iteration: int,
    final_iterations: int,
    calibration_summary: dict[str, Any],
    mean_check: dict[str, Any],
    uvch_results: pd.DataFrame,
) -> str:
    available = uvch_results[uvch_results["catboost_available"] == 1].copy()
    if available.empty:
        top_gap = available
    else:
        available["gap_current"] = (available["rul_p50_c"] - available["weibull_rul_p50_c"]).abs()
        available["gap_after"] = (available["rul_p50_a"] - available["weibull_rul_p50_a"]).abs()
        available["max_gap"] = available[["gap_current", "gap_after"]].max(axis=1)
        top_gap = available.sort_values("max_gap", ascending=False).head(10)

    lines = [
        "# CatBoost AFT UVCh Report",
        "",
        "## Training set",
        "",
        f"- Rows after filters: `{len(train_df) + len(test_df)}`",
        f"- Train rows: `{len(train_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Train wells: `{train_df['well'].nunique()}`",
        f"- Test wells: `{test_df['well'].nunique()}`",
        f"- Events in train: `{int(train_df['event'].sum())}`",
        f"- Events in test: `{int(test_df['event'].sum())}`",
        "",
        "## Model metrics",
        "",
        f"- Hold-out C-index: `{c_index:.3f}`" if c_index is not None and np.isfinite(c_index) else "- Hold-out C-index: `n/a`",
        f"- Post-hoc sigma_hat: `{sigma_hat:.4f}`" if np.isfinite(sigma_hat) else "- Post-hoc sigma_hat: `n/a`",
        f"- Post-hoc beta_weibull: `{beta_weibull:.4f}`" if np.isfinite(beta_weibull) else "- Post-hoc beta_weibull: `n/a`",
        f"- Cohort beta reference M0_base: `1.316`",
        f"- Cohort beta reference M0_vch: `1.295`",
        f"- Eval best iteration: `{eval_best_iteration}`",
        f"- Final model iterations: `{final_iterations}`",
        "",
        "## Cohort sanity check",
        "",
        f"- Rows in 50-55 Hz cohort: `{mean_check.get('rows', 0)}`",
        f"- Mean CatBoost RUL P50 at nno=100: `{mean_check['mean_catboost_rul_p50_nno100']:.1f}` days" if mean_check.get("mean_catboost_rul_p50_nno100") is not None else "- Mean CatBoost RUL P50 at nno=100: `n/a`",
        f"- Weibull reference at nno=100: `{mean_check.get('cohort_weibull_reference', 141.0):.1f}` days" if mean_check.get("cohort_weibull_reference") is not None else "- Weibull reference at nno=100: `n/a`",
        "",
        "## UVCh coverage",
        "",
        f"- Total UVCh wells: `{len(uvch_results)}`",
        f"- CatBoost available: `{int((uvch_results['catboost_available'] == 1).sum())}`",
        f"- Fallback to Weibull-only: `{int((uvch_results['catboost_available'] == 0).sum())}`",
        "",
        "## Calibration",
        "",
        f"- Vt hold-out rows used: `{calibration_summary.get('vt_holdout_rows', 0)}`",
        f"- Calibration plot: `{OUT_CALIBRATION.name}`",
        "",
        "## Largest CatBoost vs Weibull gaps",
        "",
    ]

    if top_gap.empty:
        lines.append("No CatBoost-scored UVCh wells were available.")
    else:
        lines.append("| well | band | rul_p50_c | weibull_p50_c | rul_p50_a | weibull_p50_a | max_gap |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for _, row in top_gap.iterrows():
            lines.append(
                f"| {row['well']} | {row['band']} | {row['rul_p50_c']} | {row['weibull_rul_p50_c']} | "
                f"{row['rul_p50_a']} | {row['weibull_rul_p50_a']} | {row['max_gap']:.1f} |"
            )

    missing = uvch_results[uvch_results["history_source"] == "missing"]["well"].tolist()
    if missing:
        lines.extend(
            [
                "",
                "## Missing in mart",
                "",
                ", ".join(missing),
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `beta_weibull` is a post-hoc approximation from training residuals, not a native CatBoost parameter.",
            "- Numeric missing values were left as `NaN`; categorical missing values were mapped to `__missing__`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    uvch_raw = load_uvch_raw()
    positive_df = load_positive_mart()
    train_all = positive_df[(positive_df["kip"] > 0.5) & (positive_df["field"].isin(TRAIN_FIELDS))].copy()

    df_train, df_test = split_by_well(train_all)
    train_pool, _ = build_pool(df_train)
    test_pool, test_features = build_pool(df_test)

    eval_model = make_model(iterations=1000, verbose=100)
    eval_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
    best_iteration = int(eval_model.get_best_iteration())
    final_iterations = best_iteration if best_iteration > 0 else int(eval_model.tree_count_)
    final_iterations = max(final_iterations, 1)

    mu_test = eval_model.predict(test_features)
    if int(df_test["event"].sum()) > 0:
        c_index = float(
            concordance_index(
                df_test["ttf_true_best_days"].to_numpy(dtype=float),
                predicted_scores=mu_test,
                event_observed=df_test["event"].to_numpy(dtype=int),
            )
        )
    else:
        c_index = None

    final_pool, _ = build_pool(train_all)
    final_model = make_model(iterations=final_iterations, verbose=False)
    final_model.fit(final_pool)
    final_model.save_model(OUT_MODEL)

    sigma_hat, beta_weibull = estimate_sigma_beta(final_model, train_all)
    calibration_summary = calibration_plot(eval_model, df_test)
    train_lookup = build_last_history_lookup(train_all)
    positive_lookup = build_last_history_lookup(positive_df)
    uvch_results = score_uvch_wells(
        uvch_raw=uvch_raw,
        model=final_model,
        beta_weibull=beta_weibull,
        train_lookup=train_lookup,
        positive_lookup=positive_lookup,
    )
    uvch_results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    mean_check = mean_cohort_rul_check(final_model, beta_weibull, train_all)
    params = {
        "sigma_hat": sigma_hat,
        "beta_weibull": beta_weibull,
        "c_index": c_index,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "train_wells": int(df_train["well"].nunique()),
        "test_wells": int(df_test["well"].nunique()),
        "train_rows_filtered_total": int(len(train_all)),
        "train_fields": list(TRAIN_FIELDS),
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "eval_best_iteration": final_iterations,
    }
    OUT_PARAMS.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(
        train_df=df_train,
        test_df=df_test,
        c_index=c_index,
        sigma_hat=sigma_hat,
        beta_weibull=beta_weibull,
        eval_best_iteration=best_iteration,
        final_iterations=final_iterations,
        calibration_summary=calibration_summary,
        mean_check=mean_check,
        uvch_results=uvch_results,
    )
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_MODEL}")
    print(f"Saved {OUT_PARAMS}")
    print(f"Saved {OUT_REPORT}")
    print(f"Saved {OUT_CALIBRATION}")
    print(f"C-index: {c_index:.3f}" if c_index is not None else "C-index: n/a")
    print(f"sigma_hat: {sigma_hat:.4f}")
    print(f"beta_weibull: {beta_weibull:.4f}")


if __name__ == "__main__":
    main()
