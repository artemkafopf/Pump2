"""Failure-mode group mapping for Phase B competing-risks analysis.

**Single source of truth** for the §0.1 mapping of ``raw__v03_runs.failed_node``
values onto the four cause-specific hazard groups.  Confirmed by the user on
2026-07-06 (spec ``agents/analyses/phase_b_competing_risks.md`` §0.1); the
debatable assignments were resolved to the spec defaults:

* **НКТ** → hydraulic (with a with/without sensitivity run in B3).
* **Газосепаратор / Диспергатор** (gas-handling) → folded into hydraulic.
* **ТМС** (downhole sensor) → electro-thermal.

Every cause-specific result is conditional on this table.  VBA / report code must
read its CSV export (``export_mode_group_csv``) rather than re-hardcoding the map.

Group membership (failures, warehouse post-hygiene 2026-07-06, TRIM applied):

    hydraulic       ЭЦН, НКТ, Газосепаратор, Диспергатор, Входной модуль   (693)
    electro-thermal Кабельная линия, ПЭД, ТМС                              (603)
    protector       Гидрозащита                                            (186)
    other           all remaining small valve/crossover/misc labels        (45)

Physical rationale — hydraulic = flow path (scale / corrosion / abrasion / gas
interference); electro-thermal = electrical / thermal insulation & overheating;
protector = seal section wear; other = heterogeneous small categories.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# The mapping (ordered; membership is exhaustive-by-fallthrough to "other")
# ---------------------------------------------------------------------------

HYDRAULIC = "hydraulic"
ELECTRO_THERMAL = "electro-thermal"
PROTECTOR = "protector"
OTHER = "other"

MODE_GROUPS: tuple[str, ...] = (HYDRAULIC, ELECTRO_THERMAL, PROTECTOR, OTHER)

# Explicit members per group.  Keys are the *trimmed* Cyrillic failed_node
# labels exactly as they appear in raw__v03_runs after trailing-space hygiene.
GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    HYDRAULIC: ("ЭЦН", "НКТ", "Газосепаратор", "Диспергатор", "Входной модуль"),
    ELECTRO_THERMAL: ("Кабельная линия", "ПЭД", "ТМС"),
    PROTECTOR: ("Гидрозащита",),
    # OTHER is the fallthrough bucket — no explicit list; see assign_mode_group.
}

# Inverted lookup: trimmed label -> group.  Anything not present falls to OTHER.
NODE_TO_GROUP: dict[str, str] = {
    node: group
    for group, members in GROUP_MEMBERS.items()
    for node in members
}

# Physical rationale per group (for reports / CSV export).
GROUP_RATIONALE: dict[str, str] = {
    HYDRAULIC: "Flow path: scale, corrosion, abrasion, gas interference",
    ELECTRO_THERMAL: "Electrical/thermal: insulation, overheating",
    PROTECTOR: "Seal section: its own wear mechanism",
    OTHER: "Heterogeneous small categories (valves, crossovers, misc)",
}


def assign_mode_group(failed_node: object) -> str:
    """Map a single ``failed_node`` value to its cause-specific group.

    Trims surrounding whitespace before lookup (defensive against un-hygiened
    debris such as ``'НКТ '``).  Non-failures / missing labels — ``None``,
    ``NaN``, or empty string — return ``''`` (no group) so callers can keep
    censored runs out of every event indicator.  Any *labelled* value not in
    the explicit membership lists falls through to ``"other"``.
    """
    if failed_node is None:
        return ""
    if isinstance(failed_node, float) and pd.isna(failed_node):
        return ""
    label = str(failed_node).strip()
    if not label:
        return ""
    return NODE_TO_GROUP.get(label, OTHER)


def assign_mode_group_series(failed_node: pd.Series) -> pd.Series:
    """Vectorised :func:`assign_mode_group` over a pandas Series."""
    return failed_node.map(assign_mode_group)


def mode_group_frame() -> pd.DataFrame:
    """The §0.1 mapping as a tidy DataFrame (one row per explicitly-mapped node).

    ``other`` is a fallthrough bucket, so it has no explicit rows here; it is
    represented by whatever labels are *not* listed for the three named groups.
    """
    rows = []
    for group in MODE_GROUPS:
        for node in GROUP_MEMBERS.get(group, ()):  # OTHER has no explicit members
            rows.append(
                {
                    "failed_node": node,
                    "mode_group": group,
                    "rationale": GROUP_RATIONALE[group],
                }
            )
    return pd.DataFrame(rows, columns=["failed_node", "mode_group", "rationale"])


def export_mode_group_csv(path) -> pd.DataFrame:
    """Write the confirmed mapping to ``path`` (utf-8-sig) and return the frame.

    This CSV is the file report/VBA code must read; it is the serialised form of
    the single source of truth in this module.
    """
    frame = mode_group_frame()
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


__all__ = [
    "HYDRAULIC",
    "ELECTRO_THERMAL",
    "PROTECTOR",
    "OTHER",
    "MODE_GROUPS",
    "GROUP_MEMBERS",
    "NODE_TO_GROUP",
    "GROUP_RATIONALE",
    "assign_mode_group",
    "assign_mode_group_series",
    "mode_group_frame",
    "export_mode_group_csv",
]
