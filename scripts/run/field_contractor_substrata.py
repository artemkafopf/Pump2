"""Field × contractor sub-strata analysis for all remaining fields.

Contractor grouping: Борец / Шлюмберже / Other (everyone else pooled).

Analyses fields: Ya, Ic, Az, Za (Vt was already handled in vt_contractor_substrata.py;
Mc/Da have only Борец; 'Other' field is too sparse).

For each (field × contractor) cell with n_failures >= 20: fits K=2 EM (independent if
n>=40, two-stage if 20-39).  Global shapes for two-stage strata are derived from
independent strata within that field's run; if none qualify, falls back to main
phase2 global shapes (β₁=1.145, β₂=1.4025).

Also runs Ya × H2S × contractor (Ya is fully non-sour, but kept in unified flow).

Outputs → results/esp_survival_field_contractor/YYYY-MM-DD/
"""
from __future__ import annotations

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
from lifelines import KaplanMeierFitter, WeibullFitter

from analysis.paths import results_dir, RESULTS_ROOT
from analysis.workflows.esp_survival.data import load_failures_df
from analysis.workflows.esp_survival import phase2_mixture
from analysis.models.survival.latent_weibull_competing_risks import (
    latent_survival, latent_life_quantile,
)

# ── Constants ────────────────────────────────────────────────────────────────
# Global shapes from main phase2 analysis (median of non-degenerate independent fits)
MAIN_GLOBAL_BETA1 = 1.145
MAIN_GLOBAL_BETA2 = 1.4025

FIELDS_TO_ANALYSE = ["Ya", "Ic", "Az", "Za"]   # Vt done separately; Mc/Da trivial; Other sparse
MIN_K2 = 20        # below this: single Weibull only
MIN_INDEPENDENT = 40


def ctr_group(name: str) -> str:
    if name == "Борец":
        return "Борец"
    if name == "Шлюмберже":
        return "Шлюмберже"
    return "Other"


def _sw_fit(durations: np.ndarray, events: np.ndarray) -> dict:
    wf = WeibullFitter()
    wf.fit(durations, events)
    return {
        "beta": float(wf.rho_),
        "eta_days": float(wf.lambda_),
        "B50_days": float(wf.median_survival_time_),
    }


# ── Per-field analysis ────────────────────────────────────────────────────────

def run_field(
    df_field: pd.DataFrame,
    field: str,
    out_dir: Path,
) -> dict:
    """Run Phase1 + Phase2 for one field split by contractor group.

    Returns dict with keys 'sw_rows', 'mix_params', 'models'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = ["Борец", "Шлюмберже", "Other"]

    # ── Phase 1: single Weibull per contractor ────────────────────────────────
    sw_rows: list[dict] = []
    for cg in groups:
        g = df_field[df_field["ctr_grp"] == cg]
        n_fail = int(g["event"].sum())
        if n_fail == 0:
            continue
        r = _sw_fit(g["tte"].to_numpy(dtype=float), g["event"].to_numpy(dtype=int))
        sw_rows.append({"field": field, "contractor": cg, "n_failures": n_fail,
                        "n_obs": len(g), **r})

    print(f"\n[{field}] Phase 1 (single Weibull):")
    for row in sw_rows:
        print(
            f"  {row['contractor']:12s}  n={row['n_failures']:3d}  "
            f"β={row['beta']:.3f}  η={row['eta_days']:.0f}d  B50={row['B50_days']:.0f}d"
        )

    # ── Phase 2: build stratum column from contractor group ───────────────────
    eligible = [
        cg for cg in groups
        if int(df_field[df_field["ctr_grp"] == cg]["event"].sum()) >= MIN_K2
    ]
    if not eligible:
        print(f"[{field}] No contractor group reaches {MIN_K2} failures — Phase 2 skipped.")
        return {"sw_rows": sw_rows, "mix_params": pd.DataFrame(), "models": {}}

    df_k2 = df_field[df_field["ctr_grp"].isin(eligible)].copy()
    df_k2["stratum"] = field + "_" + df_k2["ctr_grp"]

    print(f"\n[{field}] Phase 2 eligible: {[field + '_' + c for c in eligible]}")

    # Determine if we need to inject fallback global shapes.
    # If all eligible cells are two-stage (n<40), phase2_mixture would fall back to
    # (0.90, 1.60).  Use main phase2 shapes instead.
    any_independent = any(
        int(df_field[df_field["ctr_grp"] == cg]["event"].sum()) >= MIN_INDEPENDENT
        for cg in eligible
    )
    preset_b1 = None if any_independent else MAIN_GLOBAL_BETA1
    preset_b2 = None if any_independent else MAIN_GLOBAL_BETA2
    if preset_b1 is not None:
        print(f"[{field}] No independent strata — injecting main global shapes "
              f"β₁={preset_b1} β₂={preset_b2}")

    out_k2 = out_dir / "phase2"
    r2 = phase2_mixture.run(
        df_k2, out_k2,
        preset_global_beta1=preset_b1,
        preset_global_beta2=preset_b2,
    )

    # Relocate outputs (replace() overwrites on Windows unlike rename())
    for src, dst_name in [
        (out_k2 / "phase2_mixture_grid.png", f"{field}_contractor_phase2_grid.png"),
        (out_k2 / "phase2_mixture_params.csv", f"{field}_contractor_phase2_params.csv"),
    ]:
        if src.exists():
            src.replace(out_dir / dst_name)

    return {
        "sw_rows": sw_rows,
        "mix_params": r2.get("results", pd.DataFrame()),
        "models": r2.get("models", {}),
        "global_beta1": r2.get("global_beta1"),
        "global_beta2": r2.get("global_beta2"),
    }


# ── Comparison figure ─────────────────────────────────────────────────────────

def make_comparison_figure(
    mix_all: pd.DataFrame,
    phase2_baseline: pd.DataFrame,
    out_path: Path,
) -> None:
    """4-panel bar figure: w₁ by field × contractor, with baseline pooled w₁ as hline."""
    fields = FIELDS_TO_ANALYSE
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    axes_flat = axes.flatten()

    ctr_colors = {"Борец": "steelblue", "Шлюмберже": "darkorange", "Other": "seagreen"}

    for ax_i, field in enumerate(fields):
        ax = axes_flat[ax_i]

        # Contractor sub-strata w₁
        sub = mix_all[mix_all["stratum"].str.startswith(field + "_")].copy()
        if sub.empty:
            ax.set_title(f"{field} — no K=2 fits", fontsize=9)
            continue

        sub["contractor"] = sub["stratum"].str.replace(f"^{field}_", "", regex=True)

        x = np.arange(len(sub))
        bars = ax.bar(
            x, sub["w1"],
            color=[ctr_colors.get(c, "gray") for c in sub["contractor"]],
            alpha=0.8, edgecolor="black", linewidth=0.6,
        )
        # Add error bars from eta1 (size of early mode)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row['contractor']}\nn={row['n_failures']}\nη₁={row['eta1_days']:.0f}d"
             for _, row in sub.iterrows()],
            fontsize=7,
        )

        # Baseline pooled w₁ from main phase2 (stratum = field_nonsour or field_sour)
        base_strata = phase2_baseline[phase2_baseline["stratum"].str.startswith(field)]
        for _, brow in base_strata.iterrows():
            lc = "red" if brow.get("degenerate", False) else "black"
            ls = "--" if brow.get("degenerate", False) else "-"
            ax.axhline(
                brow["w1"],
                color=lc, ls=ls, lw=1.5, alpha=0.7,
                label=f"pooled {brow['stratum']} w₁={brow['w1']:.3f}"
                      + (" [D]" if brow.get("degenerate", False) else ""),
            )

        # Annotate bars with degenerate flag
        for bar_, (_, row) in zip(bars, sub.iterrows()):
            if row.get("degenerate", False):
                ax.text(
                    bar_.get_x() + bar_.get_width() / 2,
                    bar_.get_height() + 0.01,
                    "[D]", ha="center", va="bottom", fontsize=7, color="red",
                )

        ax.axhline(0.75, color="gray", ls=":", lw=1, alpha=0.5, label="degen threshold 0.75")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("w₁ (early-mode fraction)", fontsize=8)
        ax.set_title(f"{field} — w₁ by contractor sub-stratum", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Field × Contractor: early-failure fraction w₁\nvs pooled field baseline (dashed = degenerate pooled)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Comparison figure saved: {out_path.name}")


def make_km_grid(
    df: pd.DataFrame,
    mix_all: pd.DataFrame,
    all_models: dict,
    out_path: Path,
) -> None:
    """Grid: rows = fields, cols = contractor groups. KM + SW + K=2."""
    fields = FIELDS_TO_ANALYSE
    ctrs = ["Борец", "Шлюмберже", "Other"]

    fig, axes = plt.subplots(len(fields), 3, figsize=(15, 4 * len(fields)), squeeze=False)

    ctr_color = {"Борец": "steelblue", "Шлюмберже": "darkorange", "Other": "seagreen"}

    for ri, field in enumerate(fields):
        df_field = df[df["field_clean"] == field].copy()
        for ci, cg in enumerate(ctrs):
            ax = axes[ri][ci]
            g = df_field[df_field["ctr_grp"] == cg]
            n_fail = int(g["event"].sum())
            title = f"{field} × {cg}\nn={n_fail}"
            ax.set_title(title, fontsize=7)

            if n_fail == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                continue

            t = g["tte"].to_numpy(dtype=float)
            e = g["event"].to_numpy(dtype=int)
            kmf = KaplanMeierFitter()
            kmf.fit(t, e)

            try:
                ci_df = kmf.confidence_interval_.reset_index()
                ci_df.columns = ["time", "lo", "hi"]
                ci_df = ci_df[ci_df["time"] > 0]
                ax.fill_between(ci_df["time"], ci_df["lo"], ci_df["hi"],
                                alpha=0.12, color=ctr_color[cg])
            except Exception:
                pass
            sf = kmf.survival_function_.reset_index()
            sf.columns = ["time", "s"]
            sf = sf[sf["time"] > 0]
            ax.step(sf["time"], sf["s"], where="post", color=ctr_color[cg], lw=2, label="KM")

            t_max = float(sf["time"].max()) * 1.1
            t_grid = np.linspace(0.5, t_max, 300)

            # Single Weibull
            wf = WeibullFitter()
            wf.fit(t, e)
            sw_s = np.exp(-(t_grid / wf.lambda_) ** wf.rho_)
            ax.plot(t_grid, sw_s, ":", color="gray", lw=1.2,
                    label=f"SW β={wf.rho_:.2f} η={wf.lambda_:.0f}d")

            # K=2 mixture if available
            stratum_key = f"{field}_{cg}"
            model = all_models.get(stratum_key)
            if model is not None:
                mix_s = np.asarray(latent_survival(t_grid, model), dtype=float)
                c1_s = np.exp(-(t_grid / model.component_1.eta) ** model.component_1.beta)
                c2_s = np.exp(-(t_grid / model.component_2.eta) ** model.component_2.beta)
                degen = model.weight_1 > 0.75
                lc = "red" if degen else "black"
                ax.plot(t_grid, mix_s, color=lc, lw=2,
                        label=f"K=2 w₁={model.weight_1:.2f}" + (" [D]" if degen else ""))
                ax.plot(t_grid, model.weight_1 * c1_s, "--", color="crimson", lw=1.1,
                        label=f"C1 η={model.component_1.eta:.0f}d")
                ax.plot(t_grid, model.weight_2 * c2_s, "--", color="darkorange", lw=1.1,
                        label=f"C2 η={model.component_2.eta:.0f}d")
            elif n_fail >= MIN_K2:
                ax.text(0.5, 0.4, f"n={n_fail} < 40\n(two-stage not\neligible alone)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=6,
                        color="gray")

            ax.set_xlim(0, t_max)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Days", fontsize=7)
            ax.set_ylabel("S(t)", fontsize=7)
            ax.legend(fontsize=5.5, loc="upper right")
            ax.tick_params(labelsize=7)

    fig.suptitle("Field × Contractor — KM + Single Weibull + K=2 Mixture", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"KM grid saved: {out_path.name}")


def main():
    t0 = time.monotonic()
    print("\n" + "=" * 70)
    print("FIELD × CONTRACTOR SUB-STRATA  (Ya, Ic, Az, Za)")
    print("=" * 70)

    df_all = load_failures_df()
    df_all["ctr_grp"] = df_all["contractor"].apply(ctr_group)

    out_root = results_dir("esp_survival_field_contractor")
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_sw: list[dict] = []
    all_mix_rows: list[pd.DataFrame] = []
    all_models: dict[str, object] = {}   # "Field_Contractor" → model

    for field in FIELDS_TO_ANALYSE:
        print(f"\n{'=' * 60}")
        print(f"FIELD: {field}")
        df_field = df_all[df_all["field_clean"] == field].copy()
        res = run_field(df_field, field, fig_dir / field)
        all_sw.extend(res["sw_rows"])
        if not res["mix_params"].empty:
            all_mix_rows.append(res["mix_params"])
        for stratum, model in res.get("models", {}).items():
            all_models[stratum] = model
        gb1 = res.get("global_beta1")
        gb2 = res.get("global_beta2")
        if gb1:
            print(f"  [{field}] Derived global shapes: β₁={gb1:.4f}  β₂={gb2:.4f}")

    # ── Consolidated outputs ──────────────────────────────────────────────────
    sw_df = pd.DataFrame(all_sw)
    sw_df.to_csv(out_root / "field_contractor_phase1.csv", index=False, encoding="utf-8-sig")

    mix_df = pd.concat(all_mix_rows, ignore_index=True) if all_mix_rows else pd.DataFrame()
    if not mix_df.empty:
        mix_df.to_csv(out_root / "field_contractor_phase2.csv", index=False, encoding="utf-8-sig")

    # ── Load baseline (main phase2 pooled results) ────────────────────────────
    baseline_csv = sorted(
        (RESULTS_ROOT / "esp_survival_phase2_mixture").glob(
            "????-??-??/figures/phase2_mixture_params.csv"
        ),
        reverse=True,
    )[0]
    baseline = pd.read_csv(baseline_csv)

    # ── Figures ───────────────────────────────────────────────────────────────
    if not mix_df.empty:
        make_comparison_figure(mix_df, baseline, fig_dir / "w1_by_contractor_comparison.png")

    make_km_grid(df_all, mix_df, all_models, fig_dir / "field_contractor_km_grid.png")

    # ── Print summary comparison ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 70)
    if not mix_df.empty:
        cols = ["stratum", "fit_mode", "n_failures", "w1", "beta1", "eta1_days",
                "beta2", "eta2_days", "eta_ratio", "rmse", "degenerate"]
        print(mix_df[cols].to_string(index=False))

    print("\n" + "─" * 70)
    print("BASELINE (pooled field-level, main phase2):")
    print(baseline[["stratum", "n_failures", "w1", "beta1", "eta1_days",
                     "beta2", "eta2_days", "degenerate"]].to_string(index=False))

    # ── Compute w₁ ranges per field to check stratification justification ─────
    if not mix_df.empty:
        print("\n" + "─" * 70)
        print("W₁ RANGE PER FIELD (contractor stratification signal):")
        for field in FIELDS_TO_ANALYSE:
            sub = mix_df[mix_df["stratum"].str.startswith(field + "_")]
            if sub.empty:
                continue
            w1_range = sub["w1"].max() - sub["w1"].min()
            print(
                f"  {field}: w₁ range={w1_range:.3f}  "
                f"[{sub['w1'].min():.3f}–{sub['w1'].max():.3f}]  "
                f"contractors: {list(sub['stratum'].str.replace(field + '_', ''))}"
                + ("  → PROMOTE" if w1_range > 0.10 else "  → OK, pool")
            )

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — {time.monotonic() - t0:.0f}s")
    print(f"Results: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
