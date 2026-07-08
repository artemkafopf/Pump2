"""Build the versioned VBA model bundle (ESP prediction system v2).

Single source of truth for the Excel VBA registry.  Everything here is
**regenerated from the SQLite mart** on the operating-time clock (``ttf_mix``),
full population, post-label-hygiene — never consumed from stale CSVs.

Outputs (written under ``results_dir("esp_survival_vba_models")``):

  esp_models.csv        one row per VBA stratum key; params + model_kind + B50 CIs
                        + uptime_factor + pct_mixed_clock + clock + fit_date
  esp_mode_mix.csv      per field-level stratum x mode group: 90/365d incidence
                        share + operating-days-to-10%/25% cumulative incidence (CIF)
  mode_group_map.csv    frozen node -> mode map (failure_modes.py) for the audit sheet
  bundle_manifest.txt   row counts, git commit, clock, date
  vba/mdlModelSeed.bas  generated CreateModelSheet literal block (cannot drift from CSV)

Decisions carried (agents/analyses/vba_model_v2.md §0, user-confirmed 2026-07-07):
  §0.1  document-only clock — headers carry _op_d; uptime_factor exported as a
        reference/audit column (conversion left to the user), NOT applied in-model.
  §0.3  single-Weibull / degenerate strata stored as w1=0 with component 2 = the
        single Weibull, tagged model_kind in {k2, k1_aic, k1_degenerate}.
  §0.4  mode-mix planning layer + B50 CI columns included.
"""
from __future__ import annotations

import math
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data.failure_modes import MODE_GROUPS, export_mode_group_csv
from analysis.models.survival.cif import (
    aalen_johansen_cif,
    cif_by_cause,
    event_code_series,
    mode_mix_at,
    time_to_incidence,
)
from analysis.models.survival.bootstrap_ci import bootstrap_stratum_ci
from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    latent_life_quantile,
)
from analysis.models.survival.weibull_model import fit_basic_weibull
from analysis.workflows.esp_survival import phase2_mixture
from analysis.workflows.esp_survival.data_mart import load_mart_df

CLOCK = "ttf_mix"

# esp_models.csv schema — order is the contract shared with the VBA importer and
# the generated seed block.  Keep field/h2s/contractor columns (VBA lookup keys)
# ahead of the new v2 columns.
MODEL_COLUMNS: tuple[str, ...] = (
    "stratum", "field", "h2s_class", "contractor_group", "model_kind",
    "w1", "beta1", "eta1", "beta2", "eta2",
    "b20", "b50", "b80", "b50_lo", "b50_hi",
    "uptime_factor", "n_runs", "n_failures", "pct_mixed_clock", "clock", "fit_date",
)

MODEMIX_COLUMNS: tuple[str, ...] = (
    "stratum", "field", "h2s_class", "mode_group", "n_mode_failures",
    "share_90d", "share_365d",
    "t_to_10pct_op_d", "t_to_10pct_lo", "t_to_10pct_hi",
    "t_to_25pct_op_d", "t_to_25pct_lo", "t_to_25pct_hi",
    "clock",
)

_DEFAULT_N_BOOT = 200
_DEFAULT_SEED = 42


# ── small helpers ───────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _quantile(p: float, w1: float, b1: float, e1: float, b2: float, e2: float) -> float:
    model = TwoComponentLatentWeibullModel(
        weight_1=w1,
        component_1=WeibullParameters(beta=b1, eta=e1, label="C1"),
        component_2=WeibullParameters(beta=b2, eta=e2, label="C2"),
    )
    q = latent_life_quantile(p, model)
    return float(q) if np.isfinite(q) else float("nan")


def _uptime_factor(g: pd.DataFrame) -> float:
    """Per-stratum u_s = mean(ttf_mix / run_days) over runs with run_days > 0.

    Runs without telemetry have ttf_mix == run_days (ratio 1.0), so u_s is an
    honest fleet-average pulled toward 1 by the ``pct_mixed_clock`` share.
    """
    rd = pd.to_numeric(g["run_days"], errors="coerce")
    tte = pd.to_numeric(g["tte"], errors="coerce")
    m = rd.notna() & (rd > 0) & tte.notna()
    if not m.any():
        return float("nan")
    ratio = (tte[m] / rd[m]).clip(upper=1.0)
    return round(float(ratio.mean()), 4)


def _pct_mixed_clock(g: pd.DataFrame) -> float:
    """Share of runs whose ttf_mix silently falls back to calendar days."""
    miss = pd.to_numeric(g["ttf_true_best_days"], errors="coerce").isna()
    return round(float(miss.mean()), 4)


def _single_weibull_b50_ci(
    g: pd.DataFrame, *, n_boot: int, seed: int,
) -> tuple[float | None, float | None]:
    """Well-cluster bootstrap CI for a single-Weibull B50 (k1 strata)."""
    wells = g["well_key"].dropna().unique()
    if len(wells) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    by = {w: g[g["well_key"] == w] for w in wells}
    b50s: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(wells, size=len(wells), replace=True)
        boot = pd.concat([by[w] for w in drawn], ignore_index=True)
        try:
            r = fit_basic_weibull(
                boot["tte"].to_numpy(float), boot["event"].to_numpy(int)
            )
            beta, eta = float(r["beta"]), float(r["eta"])
            b50s.append(eta * (-np.log(0.5)) ** (1.0 / beta))
        except Exception:
            continue
    if len(b50s) < max(10, int(0.1 * n_boot)):
        return (None, None)
    return (round(float(np.percentile(b50s, 2.5)), 1),
            round(float(np.percentile(b50s, 97.5)), 1))


# ── registry row assembly ───────────────────────────────────────────────────

def _registry_row(
    key: str,
    g: pd.DataFrame,
    point_model: TwoComponentLatentWeibullModel,
    res_row: pd.Series,
    *,
    field: str,
    h2s: str,
    ctr: str,
    is_two_stage: bool,
    global_b1: float | None,
    global_b2: float | None,
    n_boot: int,
    seed: int,
    fit_date: str,
) -> dict:
    """Assemble one esp_models.csv row, deciding model_kind per §0.3."""
    n_runs = len(g)
    n_fail = int(g["event"].sum())

    degenerate = bool(res_row.get("degenerate", False))
    mix_wins = res_row.get("mixture_wins_aic", None)
    # None (AIC not computable) is treated as "mixture does not clearly win" → k1.
    mixture_wins = bool(mix_wins) if mix_wins is not None and not pd.isna(mix_wins) else False

    if degenerate:
        model_kind = "k1_degenerate"
    elif not mixture_wins:
        model_kind = "k1_aic"
    else:
        model_kind = "k2"

    if model_kind == "k2":
        w1 = float(point_model.weight_1)
        b1 = float(point_model.component_1.beta)
        e1 = float(point_model.component_1.eta)
        b2 = float(point_model.component_2.beta)
        e2 = float(point_model.component_2.eta)
        ci = bootstrap_stratum_ci(
            g, point_model, n_boot=n_boot, seed=seed,
            fix_beta1=global_b1 if is_two_stage else None,
            fix_beta2=global_b2 if is_two_stage else None,
        )
        b50_lo, b50_hi = ci["b50_lo"], ci["b50_hi"]
    else:
        # §0.3 — collapse to the single Weibull; store as w1=0 so the mixture
        # math uses component 2 only.  Component 1 mirrors it (never weighted in).
        sw = fit_basic_weibull(g["tte"].to_numpy(float), g["event"].to_numpy(int))
        b_sw, e_sw = float(sw["beta"]), float(sw["eta"])
        w1, b1, e1, b2, e2 = 0.0, b_sw, e_sw, b_sw, e_sw
        b50_lo, b50_hi = _single_weibull_b50_ci(g, n_boot=n_boot, seed=seed)

    b20 = _quantile(0.20, w1, b1, e1, b2, e2)
    b50 = _quantile(0.50, w1, b1, e1, b2, e2)
    b80 = _quantile(0.80, w1, b1, e1, b2, e2)

    return {
        "stratum": key,
        "field": field,
        "h2s_class": h2s,
        "contractor_group": ctr,
        "model_kind": model_kind,
        "w1": round(w1, 6),
        "beta1": round(b1, 6),
        "eta1": round(e1, 2),
        "beta2": round(b2, 6),
        "eta2": round(e2, 2),
        "b20": round(b20, 1) if np.isfinite(b20) else "",
        "b50": round(b50, 1) if np.isfinite(b50) else "",
        "b80": round(b80, 1) if np.isfinite(b80) else "",
        "b50_lo": b50_lo if b50_lo is not None else "",
        "b50_hi": b50_hi if b50_hi is not None else "",
        "uptime_factor": _uptime_factor(g),
        "n_runs": n_runs,
        "n_failures": n_fail,
        "pct_mixed_clock": _pct_mixed_clock(g),
        "clock": CLOCK,
        "fit_date": fit_date,
    }


def _global_row(df: pd.DataFrame, *, n_boot: int, seed: int, fit_date: str) -> dict:
    """Global_Pooled fallback recomputed on the full fleet."""
    g = df.copy()
    g["stratum"] = "Global_Pooled"
    r2 = phase2_mixture.run(g, Path(_tmp_figdir(fit_date, "global")))
    model = r2["models"]["Global_Pooled"]
    res_row = r2["results"].set_index("stratum").loc["Global_Pooled"]
    row = _registry_row(
        "Global_Pooled", g, model, res_row,
        field="", h2s="", ctr="Pooled",
        is_two_stage="Global_Pooled" in r2.get("two_stage_strata", []),
        global_b1=r2.get("global_beta1"), global_b2=r2.get("global_beta2"),
        n_boot=n_boot, seed=seed, fit_date=fit_date,
    )
    return row


_TMP_ROOT: Path | None = None


def _tmp_figdir(fit_date: str, tag: str) -> Path:
    from analysis.paths import results_dir
    d = results_dir("esp_survival_vba_models") / "figures" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── mode-mix (Phase B) ──────────────────────────────────────────────────────

def _mode_incidence_rows(
    cr: pd.DataFrame, *, n_boot: int, seed: int,
) -> list[dict]:
    """Per field-level stratum x mode group incidence table (CIF-based).

    Field-level strata match the model registry (field_clean x h2s_class).  Uses
    the Aalen-Johansen CIF (honest absolute risk under competing modes); times to
    10%/25% cumulative incidence carry well-cluster bootstrap CIs.
    """
    # rebuild field_clean the same way as the mart (small fields -> "Other")
    df = cr.copy()
    field_fail = df[df["event"] == 1].groupby("field")["event"].count()
    small = set(field_fail[field_fail < 10].index) | (
        set(df["field"].unique()) - set(field_fail.index)
    )
    df["field_clean"] = df["field"].apply(lambda f: "Other" if f in small else str(f))
    df["stratum_fl"] = df["field_clean"] + "_" + df["h2s_class"]

    rows: list[dict] = []
    for stratum, g in df.groupby("stratum_fl"):
        if int(g["event"].sum()) < 20:
            continue
        field_clean = g["field_clean"].iloc[0]
        h2s = g["h2s_class"].iloc[0]
        dur = g["tte"].to_numpy(float)
        code = event_code_series(g["mode_group"], g["event"], MODE_GROUPS).astype(int)
        cifs = cif_by_cause(dur, code, MODE_GROUPS)
        share90 = mode_mix_at(cifs, 90.0)
        share365 = mode_mix_at(cifs, 365.0)

        # bootstrap wells once per stratum; recompute all modes' incidence times
        wells = g["well_key"].dropna().unique()
        by = {w: g[g["well_key"] == w] for w in wells}
        rng = np.random.default_rng(seed)
        boot_times: dict[tuple[str, float], list[float]] = {
            (m, p): [] for m in MODE_GROUPS for p in (0.10, 0.25)
        }
        for _ in range(n_boot):
            drawn = rng.choice(wells, size=len(wells), replace=True)
            bs = pd.concat([by[w] for w in drawn], ignore_index=True)
            bd = bs["tte"].to_numpy(float)
            bc = event_code_series(bs["mode_group"], bs["event"], MODE_GROUPS).astype(int)
            for i, m in enumerate(MODE_GROUPS):
                curve = aalen_johansen_cif(bd, bc, i + 1)
                for p in (0.10, 0.25):
                    boot_times[(m, p)].append(
                        time_to_incidence(curve.times, curve.cif, p)
                    )

        for i, m in enumerate(MODE_GROUPS):
            curve = cifs[m]
            t10 = time_to_incidence(curve.times, curve.cif, 0.10)
            t25 = time_to_incidence(curve.times, curve.cif, 0.25)

            def _ci(key: tuple[str, float]) -> tuple[float | str, float | str]:
                arr = np.asarray([x for x in boot_times[key] if np.isfinite(x)], float)
                if len(arr) < max(10, int(0.1 * n_boot)):
                    return ("", "")
                return (round(float(np.percentile(arr, 2.5)), 1),
                        round(float(np.percentile(arr, 97.5)), 1))

            lo10, hi10 = _ci((m, 0.10))
            lo25, hi25 = _ci((m, 0.25))
            rows.append({
                "stratum": stratum,
                "field": field_clean,
                "h2s_class": h2s,
                "mode_group": m,
                "n_mode_failures": int(curve.cause_events),
                "share_90d": round(float(share90[m]), 4),
                "share_365d": round(float(share365[m]), 4),
                "t_to_10pct_op_d": round(float(t10), 1) if np.isfinite(t10) else "",
                "t_to_10pct_lo": lo10,
                "t_to_10pct_hi": hi10,
                "t_to_25pct_op_d": round(float(t25), 1) if np.isfinite(t25) else "",
                "t_to_25pct_lo": lo25,
                "t_to_25pct_hi": hi25,
                "clock": CLOCK,
            })
    return rows


# ── generated VBA seed block ────────────────────────────────────────────────

def _vba_literal(v) -> str:
    # Coerce numpy scalars to Python scalars first (numpy 2.x repr adds a dtype
    # wrapper, e.g. np.float64(0.5), which would corrupt the .bas literal).
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        s = v.strip()
        return '""' if s == "" else '"' + v.replace('"', '""') + '"'
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '""'
    if isinstance(v, float):
        return repr(round(v, 6))
    return repr(v)


def _generate_seed_bas(models: pd.DataFrame, fit_date: str, commit: str) -> str:
    """Emit a self-contained mdlModelSeed.bas whose CreateModelSheet literal block
    is generated directly from esp_models.csv — so the hardcoded fallback can
    never drift from the CSV (agents/analyses/vba_model_v2.md T1)."""
    hdr = ", ".join('"' + c + '"' for c in MODEL_COLUMNS)
    # One assignment statement per row (VBA caps a single statement at 25 line
    # continuations, so a 26-row Array( _ ... _ ) is rejected). Each row is a
    # standalone `rows(i) = Array(...)` on one physical line -- no continuations.
    n_rows = len(models)
    assign_lines = []
    for i, (_, r) in enumerate(models.iterrows()):
        cells = ", ".join(_vba_literal(r[c]) for c in MODEL_COLUMNS)
        assign_lines.append(f"    rows({i}) = Array({cells})")
    rows_assign = "\n".join(assign_lines)
    ncols = len(MODEL_COLUMNS)
    return f'''Attribute VB_Name = "mdlModelSeed"
Option Explicit

' ============================================================
' mdlModelSeed -- GENERATED, DO NOT EDIT BY HAND.
' Source : esp_models.csv  (fit_date {fit_date}, git {commit}, clock {CLOCK})
' Regenerate: python scripts/run/export_model_csv_for_vba.py
'
' CreateModelSheet builds the ESP_Models sheet from the same numbers the
' CSV importer would load, so the hardcoded fallback cannot drift from the
' bundle.  Prefer ImportModelCSV for a live update; use this only to bootstrap
' a fresh workbook offline.
' ============================================================

Private Const MODEL_SHEET As String = "ESP_Models"

Public Sub CreateModelSheet()
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(MODEL_SHEET)
    On Error GoTo 0
    If Not ws Is Nothing Then
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True
    End If
    Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    ws.Name = MODEL_SHEET

    Dim hdr As Variant
    hdr = Array({hdr})
    Dim c As Long
    For c = 0 To {ncols - 1}
        ws.Cells(1, c + 1).Value = hdr(c)
    Next c
    ws.Rows(1).Font.Bold = True

    Dim rows({n_rows - 1}) As Variant
{rows_assign}

    Dim r As Long, col As Long
    For r = 0 To UBound(rows)
        For col = 0 To {ncols - 1}
            ws.Cells(r + 2, col + 1).Value = rows(r)(col)
        Next col
    Next r

    ws.Columns("A:U").EntireColumn.AutoFit
    ' Registry reload is the caller's responsibility (RefreshModels).
End Sub
'''


# ── public entry point ──────────────────────────────────────────────────────

def build_bundle(
    out_dir: Path,
    *,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = _DEFAULT_SEED,
    fit_date: str | None = None,
) -> dict:
    """Regenerate every artifact and write the bundle to ``out_dir``."""
    fit_date = fit_date or date.today().isoformat()
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    vba_dir = out_dir / "vba"
    vba_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bundle] loading mart (clock={CLOCK}, full population)…")
    df = load_mart_df(tte_col=CLOCK)
    print(f"[bundle] {len(df):,} runs | {int(df['event'].sum()):,} failures")

    df_field = df.copy()
    df_field["stratum"] = df_field["field_clean"] + "_" + df_field["h2s_class"]
    df_ctr = df.copy()
    df_ctr["stratum"] = (
        df_ctr["field_clean"] + "_" + df_ctr["h2s_class"] + "_" + df_ctr["contractor_group"]
    )

    print("[bundle] fitting field-level K=2 mixtures…")
    r2f = phase2_mixture.run(df_field, _tmp_figdir(fit_date, "field"))
    print("[bundle] fitting contractor-level K=2 mixtures…")
    r2c = phase2_mixture.run(df_ctr, _tmp_figdir(fit_date, "ctr"))

    res_f = r2f["results"].set_index("stratum")
    res_c = r2c["results"].set_index("stratum")

    rows: list[dict] = []

    # contractor-level rows: key is already {field}_{h2s}_{ctr}
    for stratum, model in r2c["models"].items():
        g = df_ctr[df_ctr["stratum"] == stratum]
        rows.append(_registry_row(
            stratum, g, model, res_c.loc[stratum],
            field=str(g["field_clean"].iloc[0]),
            h2s=str(g["h2s_class"].iloc[0]),
            ctr=str(g["contractor_group"].iloc[0]),
            is_two_stage=stratum in r2c.get("two_stage_strata", []),
            global_b1=r2c.get("global_beta1"), global_b2=r2c.get("global_beta2"),
            n_boot=n_boot, seed=seed, fit_date=fit_date,
        ))

    # field-level rows: key is {field}_{h2s}_Pooled
    for stratum, model in r2f["models"].items():
        g = df_field[df_field["stratum"] == stratum]
        key = f"{stratum}_Pooled"
        rows.append(_registry_row(
            key, g, model, res_f.loc[stratum],
            field=str(g["field_clean"].iloc[0]),
            h2s=str(g["h2s_class"].iloc[0]),
            ctr="Pooled",
            is_two_stage=stratum in r2f.get("two_stage_strata", []),
            global_b1=r2f.get("global_beta1"), global_b2=r2f.get("global_beta2"),
            n_boot=n_boot, seed=seed, fit_date=fit_date,
        ))

    print("[bundle] fitting Global_Pooled fallback on the full fleet…")
    rows.append(_global_row(df, n_boot=n_boot, seed=seed, fit_date=fit_date))

    models_df = (
        pd.DataFrame(rows, columns=list(MODEL_COLUMNS))
        .drop_duplicates(subset=["stratum"], keep="first")
        .reset_index(drop=True)
    )

    # ── mode-mix (Phase B) ───────────────────────────────────────────────────
    print("[bundle] building mode-mix incidence table (CIF)…")
    from analysis.data.competing_risks_loader import build_competing_risks_df
    cr = build_competing_risks_df(tte_col=CLOCK)
    modemix_df = pd.DataFrame(
        _mode_incidence_rows(cr, n_boot=n_boot, seed=seed),
        columns=list(MODEMIX_COLUMNS),
    )

    # ── write everything ─────────────────────────────────────────────────────
    models_path = out_dir / "esp_models.csv"
    modemix_path = out_dir / "esp_mode_mix.csv"
    map_path = out_dir / "mode_group_map.csv"
    manifest_path = out_dir / "bundle_manifest.txt"
    seed_path = vba_dir / "mdlModelSeed.bas"

    models_df.to_csv(models_path, index=False, encoding="utf-8-sig")
    modemix_df.to_csv(modemix_path, index=False, encoding="utf-8-sig")
    map_df = export_mode_group_csv(map_path)

    commit = _git_commit()
    seed_path.write_text(
        _generate_seed_bas(models_df, fit_date, commit), encoding="utf-8"
    )

    manifest = (
        "ESP VBA model bundle v2\n"
        f"fit_date        : {fit_date}\n"
        f"git_commit      : {commit}\n"
        f"clock           : {CLOCK}\n"
        f"population       : full ({len(df)} runs, {int(df['event'].sum())} failures)\n"
        f"esp_models rows : {len(models_df)}\n"
        f"  k2            : {(models_df['model_kind'] == 'k2').sum()}\n"
        f"  k1_aic        : {(models_df['model_kind'] == 'k1_aic').sum()}\n"
        f"  k1_degenerate : {(models_df['model_kind'] == 'k1_degenerate').sum()}\n"
        f"mode_mix rows   : {len(modemix_df)}\n"
        f"mode_map rows   : {len(map_df)}\n"
        f"n_boot          : {n_boot}\n"
    )
    manifest_path.write_text(manifest, encoding="utf-8")

    print("[bundle] wrote:")
    for p in (models_path, modemix_path, map_path, manifest_path, seed_path):
        print(f"  {p}")
    print("\n" + manifest)

    return {
        "models": models_df,
        "mode_mix": modemix_df,
        "mode_map": map_df,
        "out_dir": out_dir,
        "manifest": manifest,
    }


__all__ = ["build_bundle", "MODEL_COLUMNS", "MODEMIX_COLUMNS", "CLOCK"]
