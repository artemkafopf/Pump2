"""CatBoost AFT sensitivity grid: fields × features × kip_threshold × distribution."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analysis.paths import results_dir
from catboost import CatBoostRegressor, Pool
from lifelines.utils import concordance_index
from scipy.optimize import minimize_scalar
from scipy.stats import logistic as sp_logistic
from scipy.stats import norm as sp_norm

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_uvch_catboost_aft import (  # noqa: E402
    MISSING_CATEGORY,
    RANDOM_SEED,
    band_model,
    build_last_history_lookup,
    h_base,
    load_positive_mart,
    load_uvch_raw,
    make_aft_labels,
    safe_round,
    scenario_freq_above,
    split_by_well,
    target_band,
    weibull_rul_q,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_SLUG = "uvch_catboost_aft_sensitivity"
OUT_DIR = results_dir(_SLUG)
WEIBULL_REF = 141.0
SHORT_THRESHOLD = 40.0

CAT_FEATURES = ["field", "contractor", "pump_type"]

_BASE = [
    "freq_w_mean", "freq_above_55hz_pct", "glf_m_mean", "load_m_mean",
    "kpod_m_mean", "frac_kpod_below_0p7_m", "h2s_proxy_mg_l",
    "gypsum_proxy_m_mean", "salt_proxy_m_mean", "nominal_flow_m3d",
    "motor_power_kw", "pbubble_atm", "run_number",
    "field", "contractor", "pump_type",
]

FEATURE_VARIANTS: dict[str, list[str]] = {
    "freq_only": [
        "freq_w_mean", "freq_above_55hz_pct", "run_number",
        "field", "contractor", "pump_type",
    ],
    "baseline_current": _BASE,
    "no_chemistry": [f for f in _BASE if f not in {"h2s_proxy_mg_l", "gypsum_proxy_m_mean", "salt_proxy_m_mean"}],
    "no_run_number": [f for f in _BASE if f != "run_number"],
    "freq_plus_mech": [
        "freq_w_mean", "freq_above_55hz_pct", "load_m_mean",
        "kpod_m_mean", "frac_kpod_below_0p7_m", "run_number",
        "field", "contractor", "pump_type",
    ],
}

FIELDS_VARIANTS: dict[str, tuple[str, ...]] = {
    "vt_only": ("Vt",),
    "vt_za_az_ic": ("Vt", "Za", "Az", "Ic"),
    "vt_za_az_ic_ya": ("Vt", "Za", "Az", "Ic", "Ya"),
}

KIP_THRESHOLDS = [0.4, 0.5, 0.6]

# Ось: тип дистрибуции AFT вместо depth/lr
# Extreme = Weibull (текущий baseline), Normal = LogNormal, Logistic = Log-Logistic
DIST_VARIANTS = ["Extreme", "Normal", "Logistic"]
MODEL_DEPTH = 6
MODEL_LR = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for f in features:
        if f in CAT_FEATURES:
            col = df[f].astype("string") if f in df.columns else pd.Series(MISSING_CATEGORY, index=df.index)
            result[f] = col.fillna(MISSING_CATEGORY)
        else:
            result[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else float("nan")
    return result


def _pool(df: pd.DataFrame, features: list[str], dist: str) -> tuple[Pool, pd.DataFrame]:
    feat = _prepare(df, features)
    cat_idx = [feat.columns.get_loc(c) for c in CAT_FEATURES if c in feat.columns]
    pool = Pool(
        data=feat,
        label=make_aft_labels(
            df["ttf_true_best_days"].to_numpy(dtype=float),
            df["event"].to_numpy(dtype=int),
        ),
        cat_features=cat_idx,
    )
    return pool, feat


def _cbm(dist: str, iters: int = 500) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function=f"SurvivalAft:dist={dist}",
        eval_metric=f"SurvivalAft:dist={dist}",
        iterations=iters,
        depth=MODEL_DEPTH,
        learning_rate=MODEL_LR,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )


# ---------------------------------------------------------------------------
# Sigma MLE — правильная оценка по полному набору (events + censored)
# для каждой дистрибуции
# ---------------------------------------------------------------------------

def _sigma_mle(model: CatBoostRegressor, df: pd.DataFrame, features: list[str], dist: str) -> float:
    """MLE for AFT sigma using both observed failures and right-censored observations.

    log(T) = mu(x) + sigma * eps
      Extreme  -> eps ~ Gumbel(0,1)  -> T ~ Weibull, sigma = 1/beta
      Normal   -> eps ~ N(0,1)       -> T ~ LogNormal
      Logistic -> eps ~ Logistic(0,1)-> T ~ Log-Logistic
    """
    log_t = np.log(df["ttf_true_best_days"].to_numpy(dtype=float))
    event = df["event"].to_numpy(dtype=int)
    mu = model.predict(_prepare(df, features))

    if dist == "Extreme":
        def neg_ll(s: float) -> float:
            z = (log_t - mu) / s
            return -float(np.sum(event * (-np.log(s) + z - np.exp(z)) + (1 - event) * (-np.exp(z))))

    elif dist == "Normal":
        def neg_ll(s: float) -> float:
            z = (log_t - mu) / s
            return -float(np.sum(event * (sp_norm.logpdf(z) - np.log(s)) + (1 - event) * sp_norm.logsf(z)))

    elif dist == "Logistic":
        def neg_ll(s: float) -> float:
            z = (log_t - mu) / s
            return -float(np.sum(event * (sp_logistic.logpdf(z) - np.log(s)) + (1 - event) * sp_logistic.logsf(z)))

    else:
        raise ValueError(f"Unknown dist: {dist}")

    result = minimize_scalar(neg_ll, bounds=(0.05, 5.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Conditional RUL — закрытая формула для каждой дистрибуции
# ---------------------------------------------------------------------------

def _rul(q_surv: float, age: float, mu: float, sigma: float, dist: str) -> float:
    """Conditional q_surv-quantile of remaining life given survival to age.

    Solves: S(age + u) / S(age) = q_surv  for u >= 0.
    """
    if age <= 0:
        age = 1.0
    log_age = math.log(age)

    if dist == "Extreme":
        # Weibull: S(t) = exp(-exp((log(t)-mu)/sigma))
        # beta = 1/sigma, eta = exp(mu)
        eta = math.exp(mu)
        beta = 1.0 / sigma
        from analyze_uvch_catboost_aft import conditional_rul
        return conditional_rul(q_surv, age, eta, beta)

    elif dist == "Normal":
        # LogNormal: S(t) = 1 - Phi((log(t) - mu) / sigma)
        z_age = (log_age - mu) / sigma
        s_age = float(sp_norm.sf(z_age))
        if s_age < 1e-12:
            return 0.0
        target_s = q_surv * s_age
        if target_s < 1e-12:
            return 0.0
        z_star = float(sp_norm.isf(target_s))  # Phi^{-1}(1 - target_s)
        return max(math.exp(mu + sigma * z_star) - age, 0.0)

    elif dist == "Logistic":
        # Log-Logistic: S(t) = 1 / (1 + exp((log(t) - mu) / sigma))
        z_age = (log_age - mu) / sigma
        s_age = float(sp_logistic.sf(z_age))
        if s_age < 1e-12:
            return 0.0
        target_s = q_surv * s_age
        if target_s < 1e-12:
            return 0.0
        z_star = float(sp_logistic.isf(target_s))
        return max(math.exp(mu + sigma * z_star) - age, 0.0)

    else:
        raise ValueError(f"Unknown dist: {dist}")


# ---------------------------------------------------------------------------
# Cohort sanity check (50-55 Hz, nno=100)
# ---------------------------------------------------------------------------

def _cohort_rul(model: CatBoostRegressor, sigma: float, df: pd.DataFrame,
                features: list[str], dist: str) -> float | None:
    sub = df[df["freq_w_mean"].between(50.0, 55.0)].copy()
    if sub.empty or not np.isfinite(sigma) or sigma <= 0:
        return None
    mus = model.predict(_prepare(sub, features))
    vals = [_rul(0.50, 100.0, float(m), sigma, dist) for m in mus]
    valid = [v for v in vals if np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


# ---------------------------------------------------------------------------
# UVCh scoring
# ---------------------------------------------------------------------------

def _score_uvch(
    uvch_raw: list,
    model: CatBoostRegressor,
    sigma: float,
    train_lkp: dict,
    pos_lkp: dict,
    features: list[str],
    dist: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, nno, _ql, _qo, _glf, _gb, _ga, f_cur, _ml, f_after, *_ in uvch_raw:
        band = target_band(f_after, f_cur)
        h_fn, m = band_model(band)
        row: dict[str, Any] = {
            "well": well,
            "catboost_available": 0,
            "rul_p50_c": None,
            "rul_p50_a": None,
            "weibull_rul_p50_c": weibull_rul_q(0.50, nno, h_base),
            "weibull_rul_p50_a": weibull_rul_q(0.50, nno, h_fn, m),
        }
        history = train_lkp.get(well)
        if history is None:
            history = pos_lkp.get(well)

        if history is not None and np.isfinite(sigma) and sigma > 0:
            cur_pct = pd.to_numeric(
                pd.Series([history.get("freq_above_55hz_pct")]), errors="coerce"
            ).iloc[0]
            sc_pct = scenario_freq_above(cur_pct, band)

            def _predict(f_target: float, pct: float) -> float | None:
                r = history.copy()
                r["freq_w_mean"] = f_target
                r["freq_above_55hz_pct"] = pct
                mu = float(model.predict(_prepare(pd.DataFrame([r]), features))[0])
                rul = _rul(0.50, nno, mu, sigma, dist)
                return safe_round(rul, 0) if np.isfinite(rul) else None

            row.update({
                "catboost_available": 1,
                "rul_p50_c": _predict(float(f_cur), cur_pct),
                "rul_p50_a": _predict(float(f_after), sc_pct),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single config
# ---------------------------------------------------------------------------

def run_one(
    config_id: int,
    fields_variant: str,
    feature_variant: str,
    kip_threshold: float,
    dist: str,
    positive_df: pd.DataFrame,
    uvch_raw: list,
) -> dict[str, Any]:
    features = FEATURE_VARIANTS[feature_variant]
    fields = FIELDS_VARIANTS[fields_variant]
    data = positive_df[(positive_df["kip"] > kip_threshold) & (positive_df["field"].isin(fields))].copy()

    out: dict[str, Any] = {
        "config_id": config_id,
        "fields_variant": fields_variant,
        "feature_variant": feature_variant,
        "kip_threshold": kip_threshold,
        "dist": dist,
        "rows_total": len(data),
        "train_rows": None, "test_rows": None,
        "train_wells": None, "test_wells": None,
        "c_index": None, "sigma_hat": None,
        "beta_weibull": None,  # 1/sigma только для Extreme, для Normal/Logistic - не интерпретируется
        "cohort_mean_rul_p50_nno100": None, "cohort_weibull_gap_days": None,
        "uvch_available_wells": None,
        "uvch_mean_gap_current_days": None, "uvch_mean_gap_after_days": None,
        "uvch_median_gap_current_days": None, "uvch_extreme_short_count": None,
        "error": None,
    }

    if len(data) < 30:
        out["error"] = "too_few_rows"
        return out

    try:
        df_train, df_test = split_by_well(data)
    except RuntimeError as exc:
        out["error"] = str(exc)[:80]
        return out

    out["train_rows"] = len(df_train)
    out["test_rows"] = len(df_test)
    out["train_wells"] = int(df_train["well"].nunique())
    out["test_wells"] = int(df_test["well"].nunique())

    try:
        train_pool, _ = _pool(df_train, features, dist)
        test_pool, test_feat = _pool(df_test, features, dist)

        eval_model = _cbm(dist, iters=500)
        eval_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)
        best_iter = max(int(eval_model.get_best_iteration() or eval_model.tree_count_), 1)

        mu_test = eval_model.predict(test_feat)
        if int(df_test["event"].sum()) > 0:
            out["c_index"] = round(float(concordance_index(
                df_test["ttf_true_best_days"].to_numpy(dtype=float),
                predicted_scores=mu_test,
                event_observed=df_test["event"].to_numpy(dtype=int),
            )), 4)

        full_pool, _ = _pool(data, features, dist)
        final = _cbm(dist, iters=best_iter)
        final.fit(full_pool)

        sigma = _sigma_mle(final, data, features, dist)
        out["sigma_hat"] = round(sigma, 4) if np.isfinite(sigma) else None
        if dist == "Extreme":
            out["beta_weibull"] = round(1.0 / sigma, 4) if sigma > 0 else None

        cohort = _cohort_rul(final, sigma, data, features, dist)
        out["cohort_mean_rul_p50_nno100"] = round(cohort, 1) if cohort is not None else None
        out["cohort_weibull_gap_days"] = round(cohort - WEIBULL_REF, 1) if cohort is not None else None

        train_lkp = build_last_history_lookup(data)
        pos_lkp = build_last_history_lookup(positive_df)
        uvch = _score_uvch(uvch_raw, final, sigma, train_lkp, pos_lkp, features, dist)

        avail = uvch[uvch["catboost_available"] == 1].copy()
        out["uvch_available_wells"] = int(len(avail))
        if not avail.empty:
            gap_c = avail["rul_p50_c"].astype(float) - avail["weibull_rul_p50_c"].astype(float)
            gap_a = avail["rul_p50_a"].astype(float) - avail["weibull_rul_p50_a"].astype(float)
            out["uvch_mean_gap_current_days"] = round(float(gap_c.mean()), 1)
            out["uvch_mean_gap_after_days"] = round(float(gap_a.mean()), 1)
            out["uvch_median_gap_current_days"] = round(float(gap_c.median()), 1)
            out["uvch_extreme_short_count"] = int((avail["rul_p50_c"].astype(float) < SHORT_THRESHOLD).sum())

    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:80]}"

    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: object, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{float(v):.{d}f}"


def build_report(summary: pd.DataFrame, top: pd.DataFrame) -> str:
    valid = summary.dropna(subset=["c_index", "sigma_hat"])
    n_err = int(summary["error"].notna().sum())

    baseline_mask = (
        (summary["fields_variant"] == "vt_za_az_ic") &
        (summary["feature_variant"] == "baseline_current") &
        (summary["kip_threshold"] == 0.5) &
        (summary["dist"] == "Extreme")
    )
    brow = summary[baseline_mask].iloc[0] if baseline_mask.any() else None

    lines = [
        "# CatBoost AFT — Sensitivity по дистрибуции",
        "",
        f"Grid: **{len(summary)}** конфигураций — {len(valid)} успешных, {n_err} ошибок.",
        "",
        "Три дистрибуции AFT:",
        "- `Extreme` — Weibull/Gumbel (текущий baseline), монотонно растущий hazard",
        "- `Normal`  — LogNormal, одногорбый hazard (растёт затем убывает)",
        "- `Logistic`— Log-Logistic, одногорбый hazard, тяжёлые хвосты",
        "",
        "Sigma: оценка MLE по полному датасету (events + censored).",
        "Для Extreme: beta = 1/sigma (форма Weibull). Для Normal/Logistic — sigma напрямую.",
        "",
        "## Baseline (vt_za_az_ic / baseline_current / kip>0.5 / Extreme)",
        "",
    ]
    if brow is not None:
        lines += [
            f"- C-index: `{_fmt(brow['c_index'])}`",
            f"- sigma_hat: `{_fmt(brow['sigma_hat'])}` → beta = `{_fmt(brow.get('beta_weibull'))}`",
            f"- cohort RUL P50 при nno=100: `{_fmt(brow['cohort_mean_rul_p50_nno100'], 1)}` дней"
            f"  (gap: `{_fmt(brow['cohort_weibull_gap_days'], 1)}`)",
            f"- uvch_extreme_short_count: `{_fmt(brow.get('uvch_extreme_short_count'), 0)}`",
            f"- uvch_mean_gap_current_days: `{_fmt(brow.get('uvch_mean_gap_current_days'), 1)}`",
        ]

    lines += ["", "## Sigma и C-index по дистрибуции и feature_variant", ""]

    for dist in DIST_VARIANTS:
        sub = valid[valid["dist"] == dist]
        if sub.empty:
            continue
        grp = sub.groupby("feature_variant").agg(
            c_mean=("c_index", "mean"),
            sigma_mean=("sigma_hat", "mean"),
            sigma_min=("sigma_hat", "min"),
            sigma_max=("sigma_hat", "max"),
            short_mean=("uvch_extreme_short_count", "mean"),
            gap_mean=("uvch_mean_gap_current_days", "mean"),
        ).round(3)
        lines.append(f"### dist={dist}")
        lines.append("")
        lines.append("| feature_variant | C-idx mean | sigma mean | sigma min | sigma max | extreme_short | gap_cur |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, row in grp.iterrows():
            lines.append(
                f"| {k} | {row['c_mean']:.3f} | {row['sigma_mean']:.3f}"
                f" | {row['sigma_min']:.3f} | {row['sigma_max']:.3f}"
                f" | {row['short_mean']:.1f} | {row['gap_mean']:.1f} |"
            )
        lines.append("")

    lines += [
        "## Сравнение дистрибуций по baseline_current",
        "",
        "| dist | kip | C-idx | sigma | beta (Ext) | cohort_gap | extreme_short | gap_cur |",
        "|---|---|---|---|---|---|---|---|",
    ]
    bc = valid[valid["feature_variant"] == "baseline_current"].sort_values(["dist", "kip_threshold"])
    for _, r in bc.iterrows():
        beta_str = _fmt(r.get("beta_weibull")) if r["dist"] == "Extreme" else "—"
        lines.append(
            f"| {r['dist']} | {r['kip_threshold']} | {_fmt(r['c_index'])}"
            f" | {_fmt(r['sigma_hat'])} | {beta_str}"
            f" | {_fmt(r.get('cohort_weibull_gap_days'), 1)}"
            f" | {_fmt(r.get('uvch_extreme_short_count'), 0)}"
            f" | {_fmt(r.get('uvch_mean_gap_current_days'), 1)} |"
        )

    lines += [
        "",
        "## Топ-10 конфигураций по компромиссному скору",
        "",
        "| # | fields | features | kip | dist | C-idx | sigma | cohort_gap | short | gap_cur |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in top.iterrows():
        short = _fmt(r.get("uvch_extreme_short_count"), 0)
        lines.append(
            f"| {int(r['config_id'])} | {r['fields_variant']} | {r['feature_variant']}"
            f" | {r['kip_threshold']} | {r['dist']}"
            f" | {_fmt(r['c_index'])} | {_fmt(r['sigma_hat'])}"
            f" | {_fmt(r.get('cohort_weibull_gap_days'), 1)} | {short}"
            f" | {_fmt(r.get('uvch_mean_gap_current_days'), 1)} |"
        )

    lines += [
        "",
        "## Ключевые выводы",
        "",
    ]
    if not valid.empty:
        for dist in DIST_VARIANTS:
            sub = valid[valid["dist"] == dist]
            if sub.empty:
                continue
            sc = sub["uvch_extreme_short_count"].mean()
            gap = sub["uvch_mean_gap_current_days"].mean()
            ci = sub["c_index"].mean()
            sg = sub["sigma_hat"].mean()
            lines.append(
                f"- **{dist}**: C-idx={ci:.3f}, sigma={sg:.3f},"
                f" extreme_short={sc:.1f}, gap_cur={gap:.1f}"
            )

    lines += [
        "",
        "Полная таблица: `catboost_aft_sensitivity_summary.csv`",
        "Топ-конфигурации: `catboost_aft_sensitivity_top_configs.csv`",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    uvch_raw = load_uvch_raw()
    positive_df = load_positive_mart()

    configs: list[dict[str, Any]] = []
    idx = 1
    for fv in FIELDS_VARIANTS:
        for fvar in FEATURE_VARIANTS:
            for kip in KIP_THRESHOLDS:
                for dist in DIST_VARIANTS:
                    configs.append({
                        "config_id": idx,
                        "fields_variant": fv,
                        "feature_variant": fvar,
                        "kip_threshold": kip,
                        "dist": dist,
                    })
                    idx += 1

    print(f"Running {len(configs)} configurations (3 dist x 3 fields x 5 features x 3 kip)...")
    results: list[dict[str, Any]] = []
    for cfg in configs:
        tag = (
            f"[{cfg['config_id']:03d}/{len(configs)}]"
            f" {cfg['dist']:8s}"
            f" {cfg['fields_variant']:20s}"
            f" {cfg['feature_variant']:20s}"
            f" kip>{cfg['kip_threshold']}"
        )
        print(f"  {tag}", end=" ... ", flush=True)
        row = run_one(
            config_id=cfg["config_id"],
            fields_variant=cfg["fields_variant"],
            feature_variant=cfg["feature_variant"],
            kip_threshold=cfg["kip_threshold"],
            dist=cfg["dist"],
            positive_df=positive_df,
            uvch_raw=uvch_raw,
        )
        results.append(row)
        if row.get("error"):
            print(f"ERROR: {row['error']}")
        else:
            ci = row.get("c_index")
            sg = row.get("sigma_hat")
            print(f"c={ci:.3f} s={sg:.3f}" if ci is not None and sg is not None else "ok")

    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "catboost_aft_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    valid = summary.dropna(subset=["c_index", "sigma_hat", "uvch_extreme_short_count"])

    # Baseline C-index: Extreme / baseline_current / kip=0.5
    mask = (
        (valid["dist"] == "Extreme") &
        (valid["fields_variant"] == "vt_za_az_ic") &
        (valid["feature_variant"] == "baseline_current") &
        (valid["kip_threshold"] == 0.5)
    )
    baseline_ci = float(valid.loc[mask, "c_index"].iloc[0]) if mask.any() else 0.0

    # Скор: минимизировать extreme_short и |cohort_gap|, C-index не хуже baseline-0.03
    cands = valid[valid["c_index"] >= baseline_ci - 0.03].copy()
    if cands.empty:
        cands = valid.copy()
    cands["_score"] = (
        -cands["uvch_extreme_short_count"].fillna(99) * 3.0
        - cands["cohort_weibull_gap_days"].fillna(0.0).abs() * 0.5
    )
    top10 = cands.sort_values("_score", ascending=False).head(10).drop(columns=["_score"])
    top10.to_csv(OUT_DIR / "catboost_aft_sensitivity_top_configs.csv", index=False, encoding="utf-8-sig")

    report = build_report(summary, top10)
    (OUT_DIR / "catboost_aft_sensitivity_report.md").write_text(report, encoding="utf-8")

    print(f"\nDone. Files saved to {OUT_DIR}:")
    print(f"  catboost_aft_sensitivity_summary.csv  ({len(summary)} rows)")
    print(f"  catboost_aft_sensitivity_top_configs.csv  ({len(top10)} rows)")
    print(f"  catboost_aft_sensitivity_report.md")


if __name__ == "__main__":
    main()
