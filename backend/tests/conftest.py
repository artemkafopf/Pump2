"""Shared fixtures for the test suite.

The suite is synthetic on purpose — a test that reads the real workbook tests the
workbook.  The one thing that cannot be tested without a sheet is the *reading* of it:
which rows ``svod_nno_decomposition.load_panel`` keeps and which it silently drops.  So
this builds a nine-row Свод with **one row planted per selection stage**, and the panel
funnel is then checkable row by row against the filter it claims to describe.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import svod_nno_decomposition as D

#: (Скв., монтаж, ННО, Ql, Qном, причина остановки, отказавший узел, флаг, кислый).
#: Three survivors first, then one row per filter, in the order the filters run.
SVOD_ROWS = [
    ("YA_1", "2024-02-01", 300.0, 120.0, 250.0, "снижение подачи", "ПЭД", 1, "Некислый"),
    ("YA_2", "2024-03-01", 400.0, 130.0, 250.0, "снижение подачи", "ПЭД", 1, "Некислый"),
    ("VT_1", "2024-03-01", 250.0, 110.0, 250.0, "снижение подачи", "ПЭД", 1, "Кислый"),
    ("YA_3", "2024-04-01", 350.0, 140.0, 250.0, "снижение подачи", "ПЭД", -1, "Некислый"),
    ("ZZ_9", "2024-04-01", 360.0, 150.0, 250.0, "снижение подачи", "ПЭД", 1, "Некислый"),
    ("YA_5", "2024-05-01", np.nan, 160.0, 250.0, "", "", 0, "Некислый"),
    ("YA_6", "2024-05-01", 370.0, 0.0, 250.0, "снижение подачи", "ПЭД", 1, "Некислый"),
    ("MC_7", "2023-05-01", 380.0, 170.0, 250.0, "снижение подачи", "ПЭД", 1, "Некислый"),
    ("YA_8", "2024-06-01", 390.0, 180.0, 250.0, "ГТМ", "", 0, "Некислый"),
]


@dataclass(frozen=True)
class SvodFixture:
    """The sheet on disk plus what each of its rows was planted to prove."""

    path: Path
    #: funnel step → the single well that step is meant to remove (step 0 = raw sheet)
    planted: dict = None
    survivors: frozenset = frozenset({"YA_1", "YA_2", "VT_1"})
    n_rows: int = len(SVOD_ROWS)

    def __post_init__(self):
        object.__setattr__(self, "planted", {
            1: "YA_3",   # «Флаг отказа» = −1, исход не разобран
            2: "ZZ_9",   # префикс скважины не отображается ни на одно месторождение
            3: "YA_5",   # насос ещё работает — ННО не существует
            4: "YA_6",   # подъём есть, но ставки на строке нет
            5: "MC_7",   # Мирнинский до 2024 — когортное правило
            6: "YA_8",   # ГТМ: подъём, но не отказ
        })


@pytest.fixture(scope="session")
def svod_workbook(tmp_path_factory) -> SvodFixture:
    rows = pd.DataFrame(SVOD_ROWS, columns=[
        "Скв.", "Дата монтажа", D.COL_NNO, D.COL_QL, D.COL_QNOM,
        "Причина остановки", "Отказавший узел", "Флаг отказа", "Кислый/Некислый"])
    rows["Дата остановки"] = pd.to_datetime(rows["Дата монтажа"]) + pd.Timedelta(days=400)
    rows["Дата демонтажа"] = rows["Дата остановки"]
    rows[D.COL_DELTA] = rows[D.COL_QL] - rows[D.COL_QNOM]
    rows[D.COL_FREQ] = 50.0
    rows["Принадлежность"] = "Борец"
    path = tmp_path_factory.mktemp("svod") / "Свод.xlsx"
    rows.to_excel(path, sheet_name="Свод", index=False)
    return SvodFixture(path=path)
