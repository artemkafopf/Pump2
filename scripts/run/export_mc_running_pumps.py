"""Свод-style export of the currently RUNNING Мирнинский (Mc + Mr) ESP pumps.

Built to audit the censored tail of the Mc Weibull fit: the KM curve keeps ~46%
of pumps alive past 663 days, and that tail rests entirely on censored runs, so
the question "which Mc/Mr pumps are actually still turning, and since when?"
needs a hand-checkable answer.

"Running" = «В работе» in the current техрежим report (the operational authority
on today's state).  «Дата монтажа» / «Принадлежность» are looked up from the
open run in WellsArtificialLiftBig, falling back to the well's active Свод run.
Anything not found is left BLANK rather than derived — an estimated install date
would defeat the purpose of the audit.

Sheet 1 mirrors the «Свод» column layout for direct copy-paste; sheet 2 carries
the provenance of every value on sheet 1.

Run:  python scripts/run/export_mc_running_pumps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.data.equipment_big import load_equipment_big
from analysis.paths import results_dir
from analysis.workflows.production_risk import crosswalk

SLUG = "production_risk_mc_running_pumps"
MC_PREFIXES = ("MC", "MR")
RUNNING_STATUS = "В работе"


def _prefix(code: str) -> str:
    text = str(code or "")
    return text.split("_", 1)[0].upper() if "_" in text else text.upper()


def is_mirny(code: str) -> bool:
    return _prefix(code) in MC_PREFIXES


def svod_style(code: str) -> str:
    """MC_010 -> Mc_010 — the «Скв.» spelling used by the Свод sheet."""
    text = str(code or "")
    if "_" not in text:
        return text
    prefix, rest = text.split("_", 1)
    return f"{prefix.capitalize()}_{rest}"


def svod_header() -> list[str]:
    path = crosswalk.resolve_prediction_workbook_path()
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb["Свод"]
        return [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    finally:
        wb.close()


def big_open_runs() -> pd.DataFrame:
    """Latest still-open (no pull date) Big run per Мирнинский well."""
    big = load_equipment_big()
    big = big.assign(code=big["well_key"].map(crosswalk.norm_well))
    big = big[big["code"].astype(str).map(is_mirny)].copy()
    big["install_date"] = pd.to_datetime(big["install_date"], errors="coerce")
    big["pull_date"] = pd.to_datetime(big["pull_date"], errors="coerce")
    open_runs = big[big["pull_date"].isna()]
    return open_runs.sort_values("install_date").groupby("code").tail(1).set_index("code")


def build_rows() -> pd.DataFrame:
    tr = crosswalk.load_current_techregime_status()
    esp = crosswalk.load_esp_source()
    big = big_open_runs()

    running = sorted(
        code for code, status in tr.items()
        if is_mirny(code) and str(status.status).strip() == RUNNING_STATUS
    )

    rows: list[dict] = []
    for code in running:
        status = tr[code]
        state = esp.states_by_well.get(code)
        svod_active = state is not None and state.is_active

        mount = contractor = None
        mount_src = contractor_src = ""

        if code in big.index:
            value = big.at[code, "install_date"]
            if pd.notna(value):
                mount, mount_src = pd.Timestamp(value).date(), "Big (открытый пробег)"
            value = big.at[code, "contractor"]
            if pd.notna(value) and str(value).strip():
                contractor, contractor_src = str(value).strip(), "Big (открытый пробег)"

        if mount is None and svod_active and state.mount is not None:
            mount, mount_src = pd.Timestamp(state.mount).date(), "Свод (активный пробег)"
        if contractor is None:
            # EspState carries the grouped contractor (brt/slb/oth); «Принадлежность»
            # needs the raw Свод spelling, which only the run record keeps.
            runs = esp.runs_by_well.get(code) or []
            raw = str(runs[-1].ctr_raw).strip() if runs and runs[-1].ctr_raw else ""
            if raw:
                contractor, contractor_src = raw, "Свод (последний пробег)"

        rows.append({
            "Скв.": svod_style(code),
            "Принадлежность": contractor,
            "Дата монтажа": mount,
            "_code": code,
            "_ТР статус": status.status,
            "_ТР ННО, сут": status.age_op,
            "_источник даты монтажа": mount_src or "НЕ НАЙДЕНА",
            "_источник принадлежности": contractor_src or "НЕ НАЙДЕНА",
            "_активна в Своде": "да" if svod_active else "нет",
            "_открытый пробег в Big": "да" if code in big.index else "нет",
        })
    return pd.DataFrame(rows)


def write_workbook(path: Path, frame: pd.DataFrame) -> Path:
    header = svod_header()
    wb = openpyxl.Workbook()

    sheet = wb.active
    sheet.title = "Свод"
    sheet.append(header)
    filled = {"Скв.", "Принадлежность", "Дата монтажа"}
    for _, row in frame.iterrows():
        sheet.append([row[col] if col in filled else None for col in header])
    for cell in sheet[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for col in ("A", "B", "C", "D", "E", "F", "G"):
        sheet.column_dimensions[col].width = 16
    for cell in sheet["G"][1:]:
        cell.number_format = "DD.MM.YYYY"

    audit = wb.create_sheet("Провенанс")
    cols = ["Скв.", "Принадлежность", "Дата монтажа"] + [c for c in frame.columns if c.startswith("_")]
    audit.append([c.lstrip("_") for c in cols])
    for _, row in frame.iterrows():
        audit.append([row[c] for c in cols])
    for cell in audit[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for idx in range(1, len(cols) + 1):
        audit.column_dimensions[openpyxl.utils.get_column_letter(idx)].width = 22

    wb.save(path)
    return path


def main() -> None:
    frame = build_rows()
    out = results_dir(SLUG)
    path = write_workbook(out / "tables" / "Mc_Mr_работающие_насосы.xlsx", frame)
    frame.to_csv(out / "tables" / "mc_mr_running_pumps.csv", index=False, encoding="utf-8-sig")

    missing_date = int((frame["_источник даты монтажа"] == "НЕ НАЙДЕНА").sum())
    print(f"Работающих Mc/Mr скважин (ТР «{RUNNING_STATUS}»): {len(frame)}")
    print(f"  дата монтажа найдена: {len(frame) - missing_date}, пусто: {missing_date}")
    print(f"  активны в Своде:      {int((frame['_активна в Своде'] == 'да').sum())}")
    print(f"  открытый пробег Big:  {int((frame['_открытый пробег в Big'] == 'да').sum())}")
    print(f"\n{path}")


if __name__ == "__main__":
    main()
