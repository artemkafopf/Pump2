"""Phase 8d — Extended Cox on K=2 latent Weibull, Tier 1/2/3 covariates (Vt).

For each continuous covariate:
  1. Split Vt runs into low (≤P25) and high (≥P75) groups (middle 50% dropped for contrast).
  2. Fit K=2 Weibull mixture to each group's KM curve (same init as phase8c).
  3. Extended Cox log-log OLS on fine mixture grid → γ, β*, R².
  4. Component-specific Extended Cox (analytical): γ_k = Δβ_k, β*_k for k=1 (infant), k=2 (wear-out).

Output tables/ :
    p8d_summary.csv          — one row per covariate: γ, β*, R², component params
    p8d_mixture_params.csv   — mixture params per covariate × group
    p8d_component_cox.csv    — component-specific (γ_k, β*_k, HR_k(1d)) per covariate

Output figures/ :
    p8d_<col>_hr.png         — time-varying HR: overall mixture + C1/C2 components
    p8d_<col>_fit.png        — KM + mixture overlay for low vs high group
    p8d_forest_gamma.png     — forest plot of γ across all covariates

Run via:
    python backend/analysis/workflows/vt_failure/run.py 8d
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    fit_latent_weibull_to_km_frame,
    latent_survival,
    latent_hazard,
    weibull_survival,
)
from analysis.models.survival.extended_cox import (
    ExtendedCoxResult,
    fit_extended_cox_on_survival_grid,
)


# ── Constants ─────────────────────────────────────────────────────────────────
MIN_FAILURES = 15       # minimum failures per group to attempt mixture + Cox
SPLIT_LO_PCT = 25       # reference group: ≤P25
SPLIT_HI_PCT = 75       # treatment group: ≥P75
N_GRID = 300            # time grid points for mixture survival evaluation
GRID_TRIM_Q = 0.97      # trim time grid at this quantile of observed durations
COX_MIN_TIME = 5.0      # exclude time < 5 days from OLS regression

# ── Initial guess (from Bayesian Vt posteriors — same as phase8c) ─────────────
_INIT_MODEL = TwoComponentLatentWeibullModel(
    weight_1=0.40,
    component_1=WeibullParameters(beta=0.85, eta=100.0, label="infant"),
    component_2=WeibullParameters(beta=1.30, eta=280.0, label="wear-out"),
)


# ── Covariate catalogue ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class CovConfig:
    label: str
    unit: str = ""
    tier: int = 1
    log_transform: bool = False
    lo_label: str = "Low (≤P25)"
    hi_label: str = "High (≥P75)"


COVARIATES: dict[str, CovConfig] = {
    # Tier 1 — operational
    "freq_w_mean": CovConfig(
        "VFD frequency", "Hz", tier=1,
        lo_label="Low freq (≤P25 Hz)", hi_label="High freq (≥P75 Hz)",
    ),
    "load_m_mean": CovConfig(
        "Motor load", "frac.", tier=1,
        lo_label="Low load (≤P25)", hi_label="High load (≥P75)",
    ),
    "kpod_m_mean": CovConfig(
        "Delivery ratio Q/Qnom (КПОД)", "", tier=1,
        lo_label="Under-loaded ≤P25", hi_label="Over-loaded ≥P75",
    ),
    "glf_m_mean": CovConfig(
        "Gas-liquid fraction", "", tier=1,
        lo_label="Low GLF (≤P25)", hi_label="High GLF (≥P75)",
    ),
    "pbubble_atm": CovConfig(
        "Bubble point pressure", "atm", tier=1,
        lo_label="Low Pbubble (≤P25)", hi_label="High Pbubble (≥P75)",
    ),
    # Tier 1 — derived
    "delta_bep": CovConfig(
        "BEP deviation (Q/Qnom − 1)", "", tier=1,
        lo_label="Under-loaded δBEP ≤P25", hi_label="Over-loaded δBEP ≥P75",
    ),
    # Tier 2 — chemistry proxies
    "h2s_proxy_mg_l": CovConfig(
        "H₂S concentration", "mg/L", tier=2, log_transform=True,
        lo_label="Low H₂S (≤P25)", hi_label="High H₂S (≥P75)",
    ),
    "salt_proxy_m_mean": CovConfig(
        "Salt proxy (Cl⁻)", "", tier=2, log_transform=True,
        lo_label="Low salt (≤P25)", hi_label="High salt (≥P75)",
    ),
    "gypsum_proxy_m_mean": CovConfig(
        "Gypsum scale proxy", "", tier=2, log_transform=True,
        lo_label="Low gypsum (≤P25)", hi_label="High gypsum (≥P75)",
    ),
    # Tier 3 — equipment
    "stages": CovConfig(
        "Pump stage count", "", tier=3,
        lo_label="Few stages (≤P25)", hi_label="Many stages (≥P75)",
    ),
    "n_stages_ratio": CovConfig(
        "Stages / Qnom (n_stages_ratio)", "", tier=3,
        lo_label="Low ratio (≤P25)", hi_label="High ratio (≥P75)",
    ),
}

# colour palette: low = blue, high = red
_C_LO = "#2980b9"
_C_HI = "#c0392b"
_C_C1 = "#E07B39"   # infant component
_C_C2 = "#7B2D8B"   # wear-out component


# ── Data preparation ──────────────────────────────────────────────────────────

def _prepare_vt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Vt runs with duration > 0 and add derived covariate columns."""
    vt = df[df["is_vt"]].copy()
    vt = vt[vt["duration"] > 0].reset_index(drop=True)

    # delta_bep = KPOD − 1  (deviation from BEP; negative = under-loaded)
    if "kpod_m_mean" in vt.columns:
        vt["delta_bep"] = vt["kpod_m_mean"] - 1.0

    # n_stages_ratio = stages / nominal_flow_m3d
    if "stages" in vt.columns and "nominal_flow_m3d" in vt.columns:
        safe_flow = vt["nominal_flow_m3d"].replace(0, np.nan)
        vt["n_stages_ratio"] = vt["stages"] / safe_flow

    # use avg_glf as fallback for glf_m_mean
    if "glf_m_mean" not in vt.columns and "avg_glf" in vt.columns:
        vt["glf_m_mean"] = vt["avg_glf"]

    return vt


# ── KM helper ────────────────────────────────────────────────────────────────

def _fit_km(sub: pd.DataFrame) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(sub["duration"], sub["event"])
    return kmf


def _kmf_to_frame(kmf: KaplanMeierFitter) -> pd.DataFrame:
    sf = kmf.survival_function_.copy()
    sf.columns = ["survival"]
    sf.index.name = "time"
    sf = sf.reset_index()
    et = kmf.event_table
    sf = sf.merge(
        et[["at_risk"]].rename(columns={"at_risk": "n_risk"})
        .reset_index().rename(columns={"event_at": "time"}),
        on="time", how="left",
    )
    sf["n_risk"] = sf["n_risk"].ffill().fillna(1.0)
    return sf


# ── Mixture fitting ───────────────────────────────────────────────────────────

def _fit_mixture(kmf: KaplanMeierFitter, label: str) -> TwoComponentLatentWeibullModel | None:
    km_frame = _kmf_to_frame(kmf)
    try:
        result = fit_latent_weibull_to_km_frame(
            km_frame,
            initial_model=_INIT_MODEL,
            min_n_risk=5,
            num_starts=8,
            max_iter=600,
        )
        m = result.model
        print(
            f"    [{label}] β₁={m.component_1.beta:.3f}  η₁={m.component_1.eta:.1f}d  "
            f"w₁={m.weight_1:.3f}  |  "
            f"β₂={m.component_2.beta:.3f}  η₂={m.component_2.eta:.1f}d  "
            f"w₂={m.weight_2:.3f}  |  RMSE={result.rmse:.4f}"
        )
        return m
    except Exception as exc:
        print(f"    [{label}] mixture fit failed: {exc}")
        return None


# ── Component-specific Extended Cox (analytical) ──────────────────────────────

def _component_cox_params(
    m_lo: TwoComponentLatentWeibullModel,
    m_hi: TwoComponentLatentWeibullModel,
) -> dict[int, dict]:
    """
    For each Weibull component k, recover (β*_k, γ_k) by comparing parameters
    between the high and low groups.

        γ_k  = β_k^hi − β_k^lo          (= Δβ_k, shape difference)
        β*_k = log(β_k^hi) − β_k^hi·log(η_k^hi)
             − log(β_k^lo) + β_k^lo·log(η_k^lo)
        HR_k(t) = exp(β*_k) · t^γ_k
    """
    results = {}
    for k, (c_lo, c_hi) in enumerate(
        [(m_lo.component_1, m_hi.component_1), (m_lo.component_2, m_hi.component_2)], start=1
    ):
        gamma_k = c_hi.beta - c_lo.beta
        beta_star_k = (
            np.log(c_hi.beta) - c_hi.beta * np.log(c_hi.eta)
            - np.log(c_lo.beta) + c_lo.beta * np.log(c_lo.eta)
        )
        results[k] = {
            "mode": c_lo.label,
            "beta_lo": c_lo.beta, "eta_lo": c_lo.eta,
            "beta_hi": c_hi.beta, "eta_hi": c_hi.eta,
            "gamma_k": gamma_k,
            "beta_star_k": beta_star_k,
            "hr_1d": float(np.exp(beta_star_k)),
        }
    return results


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_covariate_hr(
    col: str,
    cfg: CovConfig,
    res: ExtendedCoxResult,
    comp: dict,
    m_lo: TwoComponentLatentWeibullModel,
    m_hi: TwoComponentLatentWeibullModel,
    t_grid: np.ndarray,
    out: Path,
) -> None:
    """Two-panel: left = HR(t) overall + components; right = log HR vs log t."""
    h_lo = np.asarray(latent_hazard(t_grid, m_lo), dtype=float)
    h_hi = np.asarray(latent_hazard(t_grid, m_hi), dtype=float)
    hr_mix = h_hi / np.clip(h_lo, 1e-12, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    hr_tv = np.exp(res.beta_star + res.gamma * np.log(np.clip(t_grid, 1e-9, None)))
    hr_lo_ci, hr_hi_ci = res.hr_ci(t_grid)
    ax.plot(t_grid, hr_mix, color="steelblue", linewidth=2.2, alpha=0.85,
            label="Mixture HR (hazard ratio)")
    ax.plot(t_grid, hr_tv, color="navy", linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"Extended Cox  γ={res.gamma:+.3f}  p={res.p_gamma:.3f}")
    ax.fill_between(t_grid, hr_lo_ci, hr_hi_ci, alpha=0.12, color="navy", label="95% CI")

    for k, p in comp.items():
        colour = _C_C1 if k == 1 else _C_C2
        hr_k = np.exp(p["beta_star_k"] + p["gamma_k"] * np.log(np.clip(t_grid, 1e-9, None)))
        ax.plot(t_grid, hr_k, color=colour, linewidth=1.6, linestyle=":",
                label=f"C{k} ({p['mode']})  γ={p['gamma_k']:+.3f}  HR(1d)={p['hr_1d']:.2f}")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Time (days)", fontsize=10)
    ax.set_ylabel("Hazard Ratio (high / low group)", fontsize=10)
    ax.set_title(
        f"{cfg.label}  —  HR(t)  [Vt, P75 vs P25]\n"
        f"β*={res.beta_star:+.3f}  γ={res.gamma:+.3f}  R²={res.r_squared:.3f}",
        fontsize=10,
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.22)

    ax2 = axes[1]
    log_t = np.log(np.clip(t_grid, 1.0, None))
    ax2.plot(log_t, np.log(np.clip(hr_mix, 1e-12, None)),
             color="steelblue", linewidth=2.0, alpha=0.85, label="Mixture log HR")
    ax2.plot(log_t, res.beta_star + res.gamma * log_t,
             color="navy", linewidth=1.5, linestyle="--", alpha=0.7,
             label=f"OLS  slope γ={res.gamma:+.3f}")
    for k, p in comp.items():
        colour = _C_C1 if k == 1 else _C_C2
        ax2.plot(log_t, p["beta_star_k"] + p["gamma_k"] * log_t, color=colour,
                 linewidth=1.5, linestyle=":", label=f"C{k}  γ={p['gamma_k']:+.3f}")

    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.set_xlabel("log(t)  [t in days]", fontsize=10)
    ax2.set_ylabel("log HR", fontsize=10)
    ax2.set_title("log HR vs log(t) — linearity check\n(slope = γ)", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.22)

    fig.suptitle(
        f"Phase 8d — Extended Cox on latent Weibull  ·  {cfg.label}  (Vt field)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / f"p8d_{col}_hr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mixture_fit_pair(
    col: str,
    cfg: CovConfig,
    km_lo: KaplanMeierFitter,
    km_hi: KaplanMeierFitter,
    m_lo: TwoComponentLatentWeibullModel,
    m_hi: TwoComponentLatentWeibullModel,
    t_grid: np.ndarray,
    out: Path,
) -> None:
    """KM + mixture overlay for low vs high group (2 panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, kmf, model, label, colour in [
        (axes[0], km_lo, m_lo, cfg.lo_label, _C_LO),
        (axes[1], km_hi, m_hi, cfg.hi_label, _C_HI),
    ]:
        ax.step(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0],
                where="post", color=colour, alpha=0.6, linewidth=1.5, label="KM")
        try:
            ci = kmf.confidence_interval_survival_function_
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                            step="post", alpha=0.10, color=colour)
        except Exception:
            pass

        s_mix = np.asarray(latent_survival(t_grid, model), dtype=float)
        ax.plot(t_grid, s_mix, color=colour, linewidth=2.2, label="Mixture S(t)")

        s1 = np.asarray(weibull_survival(t_grid, model.component_1), dtype=float)
        s2 = np.asarray(weibull_survival(t_grid, model.component_2), dtype=float)
        ax.plot(t_grid, s1, color=colour, linewidth=1.0, linestyle="--", alpha=0.65,
                label=f"C1: β={model.component_1.beta:.2f} η={model.component_1.eta:.0f}d")
        ax.plot(t_grid, s2, color=colour, linewidth=1.0, linestyle=":", alpha=0.65,
                label=f"C2: β={model.component_2.beta:.2f} η={model.component_2.eta:.0f}d")

        ax.set_title(f"{label}\nw₁={model.weight_1:.2f}  w₂={model.weight_2:.2f}", fontsize=9)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("S(t)")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        f"Phase 8d — Latent Weibull fit  ·  {cfg.label}  (Vt)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / f"p8d_{col}_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_forest(summary_df: pd.DataFrame, out: Path) -> None:
    """Forest plot of γ across all covariates, coloured by tier."""
    df = summary_df[summary_df["gamma"].notna()].copy()
    if df.empty:
        return

    df = df.sort_values("gamma", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))

    tier_colours = {1: "#2c7bb6", 2: "#d7191c", 3: "#1a9641"}
    colours = [tier_colours.get(int(t), "#555") for t in df["tier"]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))

    ax.barh(y_pos, df["gamma"],
            xerr=1.96 * df["se_gamma"],
            height=0.55,
            color=colours,
            alpha=0.75,
            capsize=3)

    for yi, row in zip(y_pos, df.itertuples()):
        p_str = f"p={row.p_gamma:.3f}" if row.p_gamma >= 0.001 else "p<0.001"
        ax.text(float(df["gamma"].max()) + 0.05, yi,
                f"γ={row.gamma:+.3f}  {p_str}",
                va="center", fontsize=7.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=8.5)
    ax.set_xlabel("γ  (time-varying Extended Cox coefficient)", fontsize=10)
    ax.set_title(
        "Phase 8d — Extended Cox γ per covariate (Vt, P75 vs P25)\n"
        "γ > 0: HR increases over time  ·  γ < 0: HR decreases over time",
        fontsize=10,
    )

    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=f"Tier {t}") for t, c in tier_colours.items()]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "p8d_forest_gamma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, out: Path) -> dict:
    tbl_out = out
    fig_out = out.parent.parent / "figures" / "phase8d_latent_cox_cov"
    tbl_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 65}")
    print("PHASE 8d — EXTENDED COX ON LATENT WEIBULL  (Tier 1/2/3, Vt)")
    print(f"{'=' * 65}")

    vt = _prepare_vt_df(df)
    print(f"  Vt base dataset: {len(vt)} runs  |  {int(vt['event'].sum())} failures")

    t_max = float(np.quantile(vt["duration"].values, GRID_TRIM_Q))
    t_grid = np.linspace(1.0, t_max, N_GRID)

    summary_rows: list[dict] = []
    mixture_rows: list[dict] = []
    comp_rows: list[dict] = []

    for col, cfg in COVARIATES.items():
        if col not in vt.columns:
            print(f"\n  [{col}] column not in dataset — skipping")
            continue

        raw = pd.to_numeric(vt[col], errors="coerce")
        if raw.notna().sum() < 2 * MIN_FAILURES:
            print(f"\n  [{col}] insufficient non-missing rows — skipping")
            continue

        # optional log transform (for right-skewed chemistry proxies)
        vals = np.log1p(raw) if cfg.log_transform else raw

        p_lo = float(np.nanpercentile(vals, SPLIT_LO_PCT))
        p_hi = float(np.nanpercentile(vals, SPLIT_HI_PCT))

        mask_lo = vals <= p_lo
        mask_hi = vals >= p_hi

        sub_lo = vt[mask_lo & raw.notna()].copy()
        sub_hi = vt[mask_hi & raw.notna()].copy()

        fail_lo = int(sub_lo["event"].sum())
        fail_hi = int(sub_hi["event"].sum())

        print(
            f"\n  [{col}]  Tier {cfg.tier}  |  "
            f"lo: n={len(sub_lo)} fail={fail_lo}  |  "
            f"hi: n={len(sub_hi)} fail={fail_hi}"
        )

        if fail_lo < MIN_FAILURES or fail_hi < MIN_FAILURES:
            summary_rows.append({
                "covariate": col, "tier": cfg.tier, "label": cfg.label,
                "n_lo": len(sub_lo), "n_hi": len(sub_hi),
                "fail_lo": fail_lo, "fail_hi": fail_hi,
                "note": f"skip: <{MIN_FAILURES} failures per group",
            })
            continue

        km_lo = _fit_km(sub_lo)
        km_hi = _fit_km(sub_hi)

        print(f"    Fitting K=2 mixtures …")
        m_lo = _fit_mixture(km_lo, cfg.lo_label)
        m_hi = _fit_mixture(km_hi, cfg.hi_label)

        if m_lo is None or m_hi is None:
            summary_rows.append({
                "covariate": col, "tier": cfg.tier, "label": cfg.label,
                "n_lo": len(sub_lo), "n_hi": len(sub_hi),
                "fail_lo": fail_lo, "fail_hi": fail_hi,
                "note": "skip: mixture fit failed",
            })
            continue

        # Evaluate mixture survival on grid
        s_lo_grid = np.asarray(latent_survival(t_grid, m_lo), dtype=float)
        s_hi_grid = np.asarray(latent_survival(t_grid, m_hi), dtype=float)

        # Extended Cox log-log OLS
        try:
            cox_res = fit_extended_cox_on_survival_grid(
                t_grid, s_lo_grid, s_hi_grid,
                min_time=COX_MIN_TIME, trim_quantile=GRID_TRIM_Q,
            )
        except Exception as exc:
            print(f"    Extended Cox OLS failed: {exc}")
            summary_rows.append({
                "covariate": col, "tier": cfg.tier, "label": cfg.label,
                "n_lo": len(sub_lo), "n_hi": len(sub_hi),
                "fail_lo": fail_lo, "fail_hi": fail_hi,
                "note": f"skip: Cox OLS failed — {exc}",
            })
            continue

        # Component-specific Extended Cox (analytical)
        comp = _component_cox_params(m_lo, m_hi)

        print(
            f"    γ={cox_res.gamma:+.4f} (p={cox_res.p_gamma:.4f})  "
            f"β*={cox_res.beta_star:+.4f}  R²={cox_res.r_squared:.3f}"
        )
        print(
            f"    C1: γ={comp[1]['gamma_k']:+.3f}  β*={comp[1]['beta_star_k']:+.3f}  "
            f"HR(1d)={comp[1]['hr_1d']:.2f}"
        )
        print(
            f"    C2: γ={comp[2]['gamma_k']:+.3f}  β*={comp[2]['beta_star_k']:+.3f}  "
            f"HR(1d)={comp[2]['hr_1d']:.2f}"
        )

        # Mixture weight shift
        w1_shift = m_hi.weight_1 - m_lo.weight_1
        print(f"    Weight shift: w₁ {m_lo.weight_1:.3f} → {m_hi.weight_1:.3f}  "
              f"(Δw₁={w1_shift:+.3f})")

        # Save mixture params
        for grp, m, n, f in [
            (cfg.lo_label, m_lo, len(sub_lo), fail_lo),
            (cfg.hi_label, m_hi, len(sub_hi), fail_hi),
        ]:
            mixture_rows.append({
                "covariate": col, "group": grp,
                "n": n, "n_fail": f,
                "w1": round(m.weight_1, 4),
                "beta1": round(m.component_1.beta, 4),
                "eta1_days": round(m.component_1.eta, 1),
                "w2": round(m.weight_2, 4),
                "beta2": round(m.component_2.beta, 4),
                "eta2_days": round(m.component_2.eta, 1),
            })

        # Save component Cox params
        for k, p in comp.items():
            comp_rows.append({
                "covariate": col, "label": cfg.label, "tier": cfg.tier,
                "component": k, "mode": p["mode"],
                "beta_lo": round(p["beta_lo"], 4), "eta_lo": round(p["eta_lo"], 1),
                "beta_hi": round(p["beta_hi"], 4), "eta_hi": round(p["eta_hi"], 1),
                "gamma_k": round(p["gamma_k"], 4),
                "beta_star_k": round(p["beta_star_k"], 4),
                "hr_1d": round(p["hr_1d"], 4),
            })

        # Summary row
        summary_rows.append({
            "covariate": col, "tier": cfg.tier, "label": cfg.label,
            "n_lo": len(sub_lo), "n_hi": len(sub_hi),
            "fail_lo": fail_lo, "fail_hi": fail_hi,
            # overall Extended Cox
            "gamma": round(cox_res.gamma, 4),
            "se_gamma": round(cox_res.se_gamma, 4),
            "p_gamma": round(cox_res.p_gamma, 4),
            "beta_star": round(cox_res.beta_star, 4),
            "se_beta_star": round(cox_res.se_beta_star, 4),
            "p_beta_star": round(cox_res.p_beta_star, 4),
            "r_squared": round(cox_res.r_squared, 4),
            # component 1 (infant)
            "gamma_c1": round(comp[1]["gamma_k"], 4),
            "beta_star_c1": round(comp[1]["beta_star_k"], 4),
            "hr_1d_c1": round(comp[1]["hr_1d"], 4),
            # component 2 (wear-out)
            "gamma_c2": round(comp[2]["gamma_k"], 4),
            "beta_star_c2": round(comp[2]["beta_star_k"], 4),
            "hr_1d_c2": round(comp[2]["hr_1d"], 4),
            # mixture weight shift
            "delta_w1": round(w1_shift, 4),
            "note": "",
        })

        # Figures
        try:
            _plot_covariate_hr(col, cfg, cox_res, comp, m_lo, m_hi, t_grid, fig_out)
            _plot_mixture_fit_pair(col, cfg, km_lo, km_hi, m_lo, m_hi, t_grid, fig_out)
        except Exception as exc:
            print(f"    Figure failed: {exc}")

    # ── Save tables ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tbl_out / "p8d_summary.csv", index=False, encoding="utf-8-sig")

    if mixture_rows:
        pd.DataFrame(mixture_rows).to_csv(
            tbl_out / "p8d_mixture_params.csv", index=False, encoding="utf-8-sig"
        )

    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            tbl_out / "p8d_component_cox.csv", index=False, encoding="utf-8-sig"
        )

    # ── Forest plot ───────────────────────────────────────────────────────────
    try:
        _plot_forest(summary_df, fig_out)
    except Exception as exc:
        print(f"  Forest plot failed: {exc}")

    # ── Print summary table ───────────────────────────────────────────────────
    sig_df = summary_df[summary_df.get("p_gamma", pd.Series(dtype=float)).notna()].copy() \
        if "p_gamma" in summary_df.columns else pd.DataFrame()

    if not sig_df.empty:
        sig_df = sig_df.sort_values("p_gamma")
        print(f"\n  {'Covariate':<22}  {'γ':>8}  {'p(γ)':>8}  {'R²':>7}  {'γ_C1':>7}  {'γ_C2':>7}")
        print("  " + "-" * 68)
        for row in sig_df.itertuples():
            star = "*" if getattr(row, "p_gamma", 1) < 0.05 else " "
            print(
                f"  {row.covariate:<22}  "
                f"{getattr(row, 'gamma', float('nan')):>+8.4f}  "
                f"{getattr(row, 'p_gamma', float('nan')):>8.4f}{star} "
                f"{getattr(row, 'r_squared', float('nan')):>7.3f}  "
                f"{getattr(row, 'gamma_c1', float('nan')):>+7.3f}  "
                f"{getattr(row, 'gamma_c2', float('nan')):>+7.3f}"
            )

    print(f"\n  Outputs → {tbl_out}  &  {fig_out}")
    print(f"{'=' * 65}\n")

    return {
        "summary": summary_df,
        "mixture_params": pd.DataFrame(mixture_rows),
        "component_cox": pd.DataFrame(comp_rows),
    }
