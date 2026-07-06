"""Phase B competing-risks analysis dataset (B0.3).

Builds the single modelling frame every Phase B task starts from by wrapping the
Phase A run-level covariate registry (``build_run_covariates``) and adding the
competing-risks layer:

* ``failed_node`` (already carried by the registry via ``raw__v03_runs``) →
  ``mode_group`` through :func:`assign_mode_group` (§0.1 mapping, user-confirmed).
* Per-group event indicators ``event_<group>`` — 1 iff the run failed *and* its
  failure belongs to that group, else 0.  Feeding ``event_<group>`` as the event
  column with ``tte`` as duration realises the **cause-specific** hazard: every
  other-cause failure is treated as censored at its own failure time.
* ``is_sour_flagged`` — the §0.2 option (c) covariate: ``h2s_proxy_mg_l >
  H2S_SOUR_THRESHOLD`` applied to **every** field (not just Vt), so the sour
  effect enters cause-specific Cox models without redefining baseline strata.

The loader also asserts the hygiene invariant (``failed_node == TRIM(failed_node)``)
and that there are **zero unlabelled failures** — a failed run with no node label
would silently vanish from every cause-specific event indicator.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from analysis.data.chemistry_run_features import H2S_SOUR_THRESHOLD
from analysis.data.failure_modes import MODE_GROUPS, assign_mode_group_series
from analysis.data.run_covariates import WAREHOUSE_DB, build_run_covariates

# The three groups carried into cause-specific Cox (B3/B4); "other" (n≈45,
# heterogeneous) is reported in the inventory / CIF but not modelled with a Cox.
COX_MODE_GROUPS: tuple[str, ...] = ("hydraulic", "electro-thermal", "protector")


def _assert_label_hygiene(con: sqlite3.Connection) -> None:
    """Fail loudly if new un-trimmed ``failed_node`` debris appears (design rule)."""
    n_dirty = pd.read_sql(
        "SELECT COUNT(*) AS n FROM raw__v03_runs "
        "WHERE failed_node IS NOT NULL AND failed_node <> TRIM(failed_node)",
        con,
    )["n"].iloc[0]
    if int(n_dirty) > 0:
        raise AssertionError(
            f"{int(n_dirty)} raw__v03_runs rows have un-trimmed failed_node — "
            "warehouse hygiene regressed; re-run label hygiene before Phase B."
        )


def build_competing_risks_df(tte_col: str = "ttf_mix") -> pd.DataFrame:
    """Return one row per run (2,634) with covariates + competing-risks columns.

    Adds ``mode_group``, ``event_<group>`` indicators (one per group in
    :data:`MODE_GROUPS`), a convenience ``mode_group_cox`` (NaN for 'other'), and
    ``is_sour_flagged``.  Duration is exposed as ``tte`` by the registry.
    """
    con = sqlite3.connect(WAREHOUSE_DB)
    try:
        _assert_label_hygiene(con)
    finally:
        con.close()

    df = build_run_covariates(tte_col=tte_col).copy()

    # ── mode group from failed_node ──────────────────────────────────────────
    df["failed_node_trimmed"] = df["failed_node"].astype("string").str.strip()
    df["mode_group"] = assign_mode_group_series(df["failed_node_trimmed"])

    # Zero unlabelled failures invariant: every event==1 must carry a group.
    unlabelled = int(((df["event"] == 1) & (df["mode_group"] == "")).sum())
    if unlabelled > 0:
        raise AssertionError(
            f"{unlabelled} failed runs have no failed_node label — cause-specific "
            "event indicators would drop them; investigate before fitting."
        )

    # ── per-group cause-specific event indicators ────────────────────────────
    for group in MODE_GROUPS:
        df[f"event_{group}"] = (
            (df["event"] == 1) & (df["mode_group"] == group)
        ).astype(np.int8)

    # ── §0.2 option (c): sour flag as an all-field covariate ─────────────────
    df["is_sour_flagged"] = (
        pd.to_numeric(df["h2s_proxy_mg_l"], errors="coerce") > H2S_SOUR_THRESHOLD
    ).astype(np.int8)

    return df


def event_column_for(group: str) -> str:
    """Name of the cause-specific event indicator column for ``group``."""
    if group not in MODE_GROUPS:
        raise KeyError(f"unknown mode group {group!r}; expected one of {MODE_GROUPS}")
    return f"event_{group}"


__all__ = [
    "COX_MODE_GROUPS",
    "build_competing_risks_df",
    "event_column_for",
    "WAREHOUSE_DB",
]
