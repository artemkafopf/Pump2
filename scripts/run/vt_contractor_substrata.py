"""Vt contractor sub-strata analysis.

Three analysis groups:
  A. Vt pooled (H2S-agnostic) × contractor         — independent K=2 (n≥40)
  B. Vt_sour × contractor                           — two-stage K=2 (n=20-39, M3 shapes)
  C. Vt_nonsour × contractor                        — K=2 or single Weibull

For each group the script:
  1. Fits Phase 1 (single Weibull, MLE) for every stratum.
  2. Fits Phase 2 (K=2 EM) where n_failures ≥ 20.
  3. Saves params CSV + grid figure per group.
  4. Saves a combined summary CSV.

Two-stage global shapes for Vt_sour:
  Uses β₁=1.347, β₂=1.392 from M3 (full Vt_sour, all contractors) so the shape
  constraints are anchored to the best-calibrated Vt_sour reference.

Outputs → results/esp_survival_vt_contractor_substrata/YYYY-MM-DD/
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.stats import weibull_min

from analysis.paths import results_dir
from analysis.workflows.esp_survival.data import load_failures_df
from analysis.workflows.esp_survival import phase2_mixture

# ── Global shapes from M3 (Vt_sour, all contractors, independent K=2 fit) ───
M3_BETA1 = 1.347
M3_BETA2 = 1.392

# Contractors to include (exclude Новомет — n=1)
KEEP_CONTRACTORS = {"Борец", "Шлюмберже", "Новые технологии"}

# Min failures for K=2 fit
MIN_K2 = 20


def _weibull_mle(durations: np.ndarray, events: np.ndarray) -> dict:
    """Fit single Weibull via MLE (scipy) on uncensored/censored data."""
    # lifelines MLE
    from lifelines import WeibullFitter
    wf = WeibullFitter()
    wf.fit(durations, events)
    beta = float(wf.rho_)     # lifelines uses rho = beta (shape)
    eta  = float(wf.lambda_)  # lifelines uses lambda = scale (= eta)
    median = float(wf.median_survival_time_)
    return {"beta": beta, "eta": eta, "median": median}


def _run_group(
    df_grp: pd.DataFrame,
    group_name: str,
    out_dir: Path,
    preset_beta1: float | None = None,
    preset_beta2: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit Phase 1 + Phase 2 for all contractor strata in this group.

    Returns (summary_df, mixture_models_dict).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    contractors = sorted(df_grp["contractor"].unique())

    # ── Phase 1: single Weibull for every contractor ─────────────────────────
    sw_rows: list[dict] = []
    for c in contractors:
        g = df_grp[df_grp["contractor"] == c]
        n_fail = int(g["event"].sum())
        n_obs = len(g)
        r = _weibull_mle(g["tte"].to_numpy(dtype=float), g["event"].to_numpy(dtype=int))
        sw_rows.append(
            {
                "group": group_name,
                "contractor": c,
                "n_obs": n_obs,
                "n_failures": n_fail,
                "beta": round(r["beta"], 4),
                "eta_days": round(r["eta"], 1),
                "median_days": round(r["median"], 1),
            }
        )
    sw_df = pd.DataFrame(sw_rows)
    sw_df.to_csv(out_dir / f"{group_name}_phase1_sw.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{group_name}] Phase 1 (single Weibull):")
    for _, row in sw_df.iterrows():
        print(
            f"  {row['contractor']:25s}  n={row['n_failures']:3d}  "
            f"β={row['beta']:.3f}  η={row['eta_days']:.0f}d  median={row['median_days']:.0f}d"
        )

    # ── Phase 2: K=2 EM for strata with n_failures ≥ MIN_K2 ────────────────
    eligible = [c for c in contractors
                if int(df_grp[df_grp["contractor"] == c]["event"].sum()) >= MIN_K2]

    if not eligible:
        print(f"[{group_name}] No strata reach {MIN_K2} failures — Phase 2 skipped.")
        return sw_df, {}

    # Build a mini-dataframe with stratum = group_name + "_" + contractor
    df_k2 = df_grp[df_grp["contractor"].isin(eligible)].copy()
    df_k2["stratum"] = group_name + "_" + df_k2["contractor"]

    out_k2 = out_dir / "phase2"
    r2 = phase2_mixture.run(
        df_k2,
        out_k2,
        preset_global_beta1=preset_beta1,
        preset_global_beta2=preset_beta2,
    )
    # Rename the saved grid figure to include group name
    grid_src = out_k2 / "phase2_mixture_grid.png"
    if grid_src.exists():
        grid_src.rename(out_dir / f"{group_name}_phase2_grid.png")

    params_src = out_k2 / "phase2_mixture_params.csv"
    if params_src.exists():
        params_src.rename(out_dir / f"{group_name}_phase2_params.csv")

    return sw_df, r2.get("models", {})


def _combined_grid(
    df_vt: pd.DataFrame,
    group_specs: list[tuple[str, str, str | None]],  # (group, h2s_class, contractor|None)
    models: dict[str, dict],  # group → {contractor: model}
    out_path: Path,
) -> None:
    """One 3×3 figure showing KM + mixture curve for each (group, contractor) cell."""
    from analysis.models.survival.latent_weibull_competing_risks import latent_survival

    groups_order = ["Vt_pooled", "Vt_sour", "Vt_nonsour"]
    contractors_order = ["Борец", "Шлюмберже", "Новые технологии"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)

    for row_i, grp in enumerate(groups_order):
        for col_j, ctr in enumerate(contractors_order):
            ax = axes[row_i][col_j]
            # filter data
            if grp == "Vt_pooled":
                mask = (df_vt["field_clean"] == "Vt") & (df_vt["contractor"] == ctr)
            elif grp == "Vt_sour":
                mask = (df_vt["stratum"] == "Vt_sour") & (df_vt["contractor"] == ctr)
            else:
                mask = (df_vt["stratum"] == "Vt_nonsour") & (df_vt["contractor"] == ctr)

            g = df_vt[mask]
            n_fail = int(g["event"].sum())
            n_obs = len(g)
            label_base = f"{grp}\n{ctr}\n(n={n_fail}/{n_obs})"

            if n_fail == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label_base, fontsize=7)
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(g["tte"].to_numpy(dtype=float), g["event"].to_numpy(dtype=int))
            try:
                ci = kmf.confidence_interval_.reset_index()
                ci.columns = ["time", "lo", "hi"]
                ci = ci[ci["time"] > 0]
                ax.fill_between(ci["time"], ci["lo"], ci["hi"], alpha=0.15, color="steelblue")
            except Exception:
                pass
            sf = kmf.survival_function_.reset_index()
            sf.columns = ["time", "survival"]
            sf = sf[sf["time"] > 0]
            ax.step(sf["time"], sf["survival"], where="post", color="steelblue", lw=2, label="KM")

            # mixture model if available
            m_key = f"{grp}_{ctr}"
            model = models.get(grp, {}).get(m_key)
            if model is not None:
                t_max = float(sf["time"].max()) * 1.15
                t_grid = np.linspace(0.5, t_max, 300)
                mix_s = np.asarray(latent_survival(t_grid, model), dtype=float)
                c1_s = np.exp(-(t_grid / model.component_1.eta) ** model.component_1.beta)
                c2_s = np.exp(-(t_grid / model.component_2.eta) ** model.component_2.beta)

                degen = model.weight_1 > 0.75
                lc = "red" if degen else "black"
                ax.plot(t_grid, mix_s, color=lc, lw=1.8, ls="-",
                        label=f"K=2 w₁={model.weight_1:.2f}" + (" [D]" if degen else ""))
                ax.plot(t_grid, model.weight_1 * c1_s, "--", color="crimson", lw=1.1,
                        label=f"C1 η={model.component_1.eta:.0f}d")
                ax.plot(t_grid, model.weight_2 * c2_s, "--", color="darkorange", lw=1.1,
                        label=f"C2 η={model.component_2.eta:.0f}d")

            # single Weibull
            from lifelines import WeibullFitter
            wf = WeibullFitter()
            wf.fit(g["tte"].to_numpy(dtype=float), g["event"].to_numpy(dtype=int))
            t_max2 = float(sf["time"].max()) * 1.15
            t_gr = np.linspace(0.5, t_max2, 200)
            sw_s = np.exp(-(t_gr / wf.lambda_) ** wf.rho_)
            ax.plot(t_gr, sw_s, ":", color="gray", lw=1.2,
                    label=f"SW β={wf.rho_:.2f}")

            ax.set_xlim(0, t_max2)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Days", fontsize=7)
            ax.set_ylabel("S(t)", fontsize=7)
            ax.set_title(label_base, fontsize=7)
            ax.legend(fontsize=5.5, loc="upper right")
            ax.tick_params(labelsize=7)

    fig.suptitle("Vt — Survival by group × contractor (KM + single Weibull + K=2 mixture)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nCombined grid saved: {out_path.name}")


def main():
    t0 = time.monotonic()
    print("\n" + "=" * 70)
    print("Vt CONTRACTOR SUB-STRATA — Phase 1 + Phase 2")
    print("=" * 70)

    df = load_failures_df()
    vt = df[(df["field_clean"] == "Vt") & df["contractor"].isin(KEEP_CONTRACTORS)].copy()

    out_root = results_dir("esp_survival_vt_contractor_substrata")
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_sw: list[pd.DataFrame] = []
    all_models: dict[str, dict] = {}  # group → {stratum_key: model}

    # ── Group A: Vt pooled × contractor ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("GROUP A: Vt pooled (sour + non-sour) × contractor")
    sw_a, models_a = _run_group(vt, "Vt_pooled", fig_dir / "groupA")
    all_sw.append(sw_a)
    # key = "Vt_pooled_<contractor>" from phase2_mixture stratum column
    all_models["Vt_pooled"] = models_a

    # ── Group B: Vt_sour × contractor (preset M3 shapes) ─────────────────────
    print("\n" + "─" * 60)
    print("GROUP B: Vt_sour × contractor (M3 global shapes β₁=1.347, β₂=1.392)")
    vt_sour = vt[vt["h2s_class"] == "sour"].copy()
    sw_b, models_b = _run_group(
        vt_sour, "Vt_sour", fig_dir / "groupB",
        preset_beta1=M3_BETA1, preset_beta2=M3_BETA2,
    )
    all_sw.append(sw_b)
    all_models["Vt_sour"] = models_b

    # ── Group C: Vt_nonsour × contractor ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("GROUP C: Vt_nonsour × contractor")
    vt_ns = vt[vt["h2s_class"] == "nonsour"].copy()
    sw_c, models_c = _run_group(vt_ns, "Vt_nonsour", fig_dir / "groupC")
    all_sw.append(sw_c)
    all_models["Vt_nonsour"] = models_c

    # ── Combined summary CSV ──────────────────────────────────────────────────
    summary = pd.concat(all_sw, ignore_index=True)
    summary.to_csv(out_root / "vt_contractor_substrata_phase1.csv", index=False,
                   encoding="utf-8-sig")
    print("\n[Summary] Phase 1 across all groups:")
    print(summary.to_string(index=False))

    # ── Combined 3×3 grid figure ──────────────────────────────────────────────
    _combined_grid(df, [], all_models, fig_dir / "vt_contractor_combined_grid.png")

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — {time.monotonic() - t0:.0f}s")
    print(f"Results: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
