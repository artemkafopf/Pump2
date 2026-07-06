"""Generate Vt model comparison figures for the dedicated comparison document.

Loads all fitted models touching Vt data:
  - Phase 1: single Weibull (Vt_sour, Vt_nonsour)
  - Phase 2 full: K=2 EM (Vt_sour, Vt_nonsour)
  - Phase 2 top-2: K=2 EM (Vt_sour, Vt_nonsour) — Борец + Шлюмберже only
  - Field full:  K=2 pooled Vt (sour+nonsour)
  - Field top-2: K=2 pooled Vt (Борец + Шлюмберже)

Outputs 3 panels:
  1. Vt_sour:    KM + single Weibull + full K=2 + top-2 K=2
  2. Vt_nonsour: KM + single Weibull + full K=2 + top-2 K=2
  3. Vt pooled:  combined KM + full pooled K=2 + top-2 pooled K=2
  4. RUL table figure for all non-degenerate models
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from analysis.paths import results_dir, RESULTS_ROOT
from analysis.workflows.esp_survival.data import load_failures_df
from analysis.workflows.esp_survival.phase2_mixture import load_models
from analysis.models.survival.latent_weibull_competing_risks import (
    latent_survival, latent_life_quantile,
)


def _load_cache(slug: str):
    root = RESULTS_ROOT / slug
    for d in sorted(root.glob("????-??-??"), reverse=True):
        p = d / "models" / "phase2_models.json"
        if p.exists():
            models, gb1, gb2, ts = load_models(p)
            return models
    return {}


def _weibull_sf(t, beta, eta):
    return np.exp(-(t / eta) ** beta)


def _km_plot(ax, t, e, color="steelblue", label="KM"):
    kmf = KaplanMeierFitter()
    kmf.fit(t, e)
    try:
        ci = kmf.confidence_interval_.reset_index()
        ci.columns = ["time", "lo", "hi"]
        ci = ci[ci["time"] > 0]
        ax.fill_between(ci["time"], ci["lo"], ci["hi"], alpha=0.12, color=color)
    except Exception:
        pass
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["time", "survival"]
    sf = sf[sf["time"] > 0]
    ax.step(sf["time"], sf["survival"], where="post", color=color, lw=2.0, label=label)
    return kmf


def _mix_curve(ax, t_grid, model, label, color, lw=1.8, ls="-"):
    s = np.asarray(latent_survival(t_grid, model), dtype=float)
    ax.plot(t_grid, s, color=color, lw=lw, ls=ls, label=label)


def main():
    df = load_failures_df()
    top2 = {"Борец", "Шлюмберже"}

    # ── load all model caches ─────────────────────────────────────────────────
    m_full   = _load_cache("esp_survival_phase2_mixture")   # Vt_sour, Vt_nonsour full
    m_top2   = _load_cache("esp_survival_phase2_top2")      # Vt_sour, Vt_nonsour top-2
    m_field  = _load_cache("esp_survival_field_mixture")    # Vt pooled full
    m_ftop2  = _load_cache("esp_survival_field_top2")       # Vt pooled top-2

    # ── Phase 1 single Weibull params ─────────────────────────────────────────
    ph1_csv = sorted(
        (RESULTS_ROOT / "esp_survival_phase1_weibull").glob("*/figures/phase1_single_weibull.csv"),
        reverse=True,
    )
    ph1 = pd.read_csv(ph1_csv[0]).set_index("stratum") if ph1_csv else pd.DataFrame()

    out = results_dir("esp_survival_vt_comparison")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_max_sour    = 400
    t_max_nonsour = 800
    t_max_pool    = 800

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 1 — Vt_sour
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 5))

    sour_all  = df[df["stratum"] == "Vt_sour"]
    sour_top2 = sour_all[sour_all["contractor"].isin(top2)]

    kmf_sour = _km_plot(ax, sour_all["tte"].to_numpy(), sour_all["event"].to_numpy(),
                        color="steelblue", label="KM — all contractors (n=89)")

    t_grid = np.linspace(0.5, t_max_sour, 500)

    # Single Weibull
    if "Vt_sour" in ph1.index:
        r = ph1.loc["Vt_sour"]
        sw = _weibull_sf(t_grid, r["beta"], r["eta_days"])
        ax.plot(t_grid, sw, color="gray", lw=1.4, ls=":", label=f"Single Weibull β={r['beta']:.3f} η={r['eta_days']:.0f}d")

    # K=2 full
    if "Vt_sour" in m_full:
        m = m_full["Vt_sour"]
        _mix_curve(ax, t_grid, m,
                   f"K=2 full (all) w₁={m.weight_1:.3f} η₁={m.component_1.eta:.0f}d",
                   "darkblue", lw=2.0, ls="-")

    # K=2 top-2
    if "Vt_sour" in m_top2:
        m = m_top2["Vt_sour"]
        _mix_curve(ax, t_grid, m,
                   f"K=2 top-2 (Б+Ш, n=56) w₁={m.weight_1:.3f} η₁={m.component_1.eta:.0f}d [DEGEN]",
                   "crimson", lw=1.6, ls="--")

    ax.set_xlim(0, t_max_sour)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days in service")
    ax.set_ylabel("Survival probability")
    ax.set_title("Vt_sour — model comparison (H2S wells)")
    ax.legend(fontsize=8)
    ax.axvline(14, color="orange", lw=1, ls=":", alpha=0.7)
    ax.text(15, 0.9, "η₁=14d\n(H2S attack)", fontsize=7, color="orange")
    plt.tight_layout()
    fig.savefig(fig_dir / "vt_sour_model_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: vt_sour_model_comparison.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Vt_nonsour
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 5))

    ns_all  = df[df["stratum"] == "Vt_nonsour"]
    ns_top2 = ns_all[ns_all["contractor"].isin(top2)]

    _km_plot(ax, ns_all["tte"].to_numpy(), ns_all["event"].to_numpy(),
             color="steelblue", label="KM — all contractors (n=167)")

    t_grid = np.linspace(0.5, t_max_nonsour, 600)

    if "Vt_nonsour" in ph1.index:
        r = ph1.loc["Vt_nonsour"]
        sw = _weibull_sf(t_grid, r["beta"], r["eta_days"])
        ax.plot(t_grid, sw, color="gray", lw=1.4, ls=":", label=f"Single Weibull β={r['beta']:.3f} η={r['eta_days']:.0f}d")

    if "Vt_nonsour" in m_full:
        m = m_full["Vt_nonsour"]
        _mix_curve(ax, t_grid, m,
                   f"K=2 full (all) w₁={m.weight_1:.3f} [DEGEN]",
                   "darkblue", lw=2.0, ls="-")

    if "Vt_nonsour" in m_top2:
        m = m_top2["Vt_nonsour"]
        _mix_curve(ax, t_grid, m,
                   f"K=2 top-2 (Б+Ш, n=149) w₁={m.weight_1:.3f} [DEGEN]",
                   "crimson", lw=1.6, ls="--")

    ax.set_xlim(0, t_max_nonsour)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days in service")
    ax.set_ylabel("Survival probability")
    ax.set_title("Vt_nonsour — model comparison (non-H2S wells)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "vt_nonsour_model_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: vt_nonsour_model_comparison.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 3 — Vt pooled (field-level)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 5))

    vt_all  = df[df["field_clean"] == "Vt"]
    vt_top2 = vt_all[vt_all["contractor"].isin(top2)]

    _km_plot(ax, vt_all["tte"].to_numpy(), vt_all["event"].to_numpy(),
             color="steelblue", label=f"KM — all contractors (n={int(vt_all['event'].sum())})")
    _km_plot(ax, vt_top2["tte"].to_numpy(), vt_top2["event"].to_numpy(),
             color="seagreen", label=f"KM — Борец+Шлюмб. only (n={int(vt_top2['event'].sum())})")

    t_grid = np.linspace(0.5, t_max_pool, 600)

    if "Vt" in m_field:
        m = m_field["Vt"]
        _mix_curve(ax, t_grid, m,
                   f"K=2 pooled full: w₁={m.weight_1:.3f} η₁={m.component_1.eta:.0f}d [DEGEN]",
                   "darkblue", lw=2.0, ls="-")

    if "Vt" in m_ftop2:
        m = m_ftop2["Vt"]
        _mix_curve(ax, t_grid, m,
                   f"K=2 pooled top-2: w₁={m.weight_1:.3f} η₁={m.component_1.eta:.0f}d",
                   "crimson", lw=2.0, ls="--")

    ax.set_xlim(0, t_max_pool)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days in service")
    ax.set_ylabel("Survival probability")
    ax.set_title("Vt field (pooled sour+non-sour) — model comparison")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "vt_pooled_model_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: vt_pooled_model_comparison.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 4 — RUL for non-degenerate Vt models
    # ═══════════════════════════════════════════════════════════════════════════
    from scipy.integrate import quad

    def rul(model, t0, t_cap=5000):
        s0 = float(np.asarray(latent_survival([t0], model))[0])
        if s0 < 1e-9:
            return float("nan")
        integral, _ = quad(
            lambda s: float(np.asarray(latent_survival([t0 + s], model))[0]),
            0, t_cap, limit=200,
        )
        return integral / s0

    t0_vals = [0, 30, 60, 90, 120, 180, 240, 300]

    nd_models = {}
    if "Vt_sour" in m_full and not m_full["Vt_sour"].weight_1 > 0.75:
        nd_models["Vt_sour (full, all)"] = m_full["Vt_sour"]
    elif "Vt_sour" in m_full:
        nd_models["Vt_sour (full, all) [DEGEN]"] = m_full["Vt_sour"]
    if "Vt" in m_ftop2:
        nd_models["Vt pooled (top-2)"] = m_ftop2["Vt"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["darkblue", "crimson", "seagreen", "darkorange"]
    for (label, model), color in zip(nd_models.items(), colors):
        rul_vals = [rul(model, t0) for t0 in t0_vals]
        ax.plot(t0_vals, rul_vals, "o-", color=color, lw=2, ms=5, label=label)

    ax.set_xlabel("Age at inspection t₀ (days)")
    ax.set_ylabel("Expected RUL (days)")
    ax.set_title("Vt — Remaining Useful Life by model")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / "vt_rul_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: vt_rul_comparison.png")

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
