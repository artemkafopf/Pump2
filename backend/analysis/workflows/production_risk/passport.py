"""Current-pump install date from the equipment passport (WellsArtificialLiftBig).

Additive to the production-risk workflow.  The passport is the authoritative equipment
register (one row per спуск, current to ~2026-06).  It gives a real install date for the
pump on each well right now, so wells whose ESP history is stale/failed (age otherwise
imputed) or that have no ESP history (age otherwise 0) get a real current-pump age.

Coverage on the current data: 709/878 plan producers have a passport record and 645 have
a live pump; ~352 stale-history and ~162 new-well ages become real.  ННО is not populated
for running pumps, so age is derived from the install date by the caller.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

import openpyxl

from analysis.paths import resolve_equipment_big_path
from analysis.workflows.production_risk.crosswalk import norm_well

# Column indices in the passport sheet (header block rows 1-3, data from row 4).
_C_WELL, _C_RUNNO, _C_MOUNT, _C_FAIL, _C_DEMO, _C_NNO = 0, 6, 7, 9, 10, 11


@dataclass
class PumpState:
    mount: datetime     # install date of the current (latest) run
    running: bool       # latest run has no recorded failure/demontage date
    run_no: object      # № спуска


def _parse_dt(v) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("0", "-", "--", "----"):
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def load_current_pump_state() -> dict[str, PumpState]:
    """Latest спуск per well → current-pump install date + running flag.

    Non-fatal: returns {} if the passport is unavailable, so callers degrade gracefully
    to their existing age logic.
    """
    try:
        path = resolve_equipment_big_path()
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return {}
    try:
        ws = wb.worksheets[0]
        runs: dict[str, list[dict]] = defaultdict(list)
        for row in ws.iter_rows(min_row=4, values_only=True):
            code = norm_well(row[_C_WELL])
            mount = _parse_dt(row[_C_MOUNT])
            if code is None or mount is None:
                continue
            runs[code].append(
                dict(run_no=row[_C_RUNNO], mount=mount,
                     fail=_parse_dt(row[_C_FAIL]), demo=_parse_dt(row[_C_DEMO]))
            )
    finally:
        wb.close()
    state: dict[str, PumpState] = {}
    for code, rr in runs.items():
        rr.sort(key=lambda x: x["mount"])
        last = rr[-1]
        state[code] = PumpState(
            mount=last["mount"],
            running=(last["fail"] is None and last["demo"] is None),
            run_no=last["run_no"],
        )
    return state
