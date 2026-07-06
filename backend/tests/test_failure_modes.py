"""Unit tests for the Phase B failure-mode group mapping (B0.1).

Pure-function tests — no warehouse dependency.  These freeze the confirmed
§0.1 mapping so any drift (a new debris label, a moved assignment) fails loudly.
"""
import unittest

import numpy as np
import pandas as pd

from analysis.data.failure_modes import (
    MODE_GROUPS,
    assign_mode_group,
    assign_mode_group_series,
    export_mode_group_csv,
    mode_group_frame,
)


class AssignModeGroupTests(unittest.TestCase):
    def test_hydraulic_members(self):
        for node in ("ЭЦН", "НКТ", "Газосепаратор", "Диспергатор", "Входной модуль"):
            self.assertEqual(assign_mode_group(node), "hydraulic", node)

    def test_electro_thermal_members(self):
        for node in ("Кабельная линия", "ПЭД", "ТМС"):
            self.assertEqual(assign_mode_group(node), "electro-thermal", node)

    def test_protector_member(self):
        self.assertEqual(assign_mode_group("Гидрозащита"), "protector")

    def test_small_labels_fall_to_other(self):
        for node in ("Клапан сливной", "Клапан обратный", "переводник",
                     "Дополнительное оборудование", "Мандрель", "какая-то новая деталь"):
            self.assertEqual(assign_mode_group(node), "other", node)

    def test_trailing_space_debris_still_maps(self):
        # The load-bearing defensive behaviour: 'НКТ ' must not leak to 'other'.
        self.assertEqual(assign_mode_group("НКТ "), "hydraulic")
        self.assertEqual(assign_mode_group("  ЭЦН  "), "hydraulic")

    def test_missing_returns_empty(self):
        self.assertEqual(assign_mode_group(None), "")
        self.assertEqual(assign_mode_group(np.nan), "")
        self.assertEqual(assign_mode_group(""), "")
        self.assertEqual(assign_mode_group("   "), "")

    def test_series_vectorised(self):
        s = pd.Series(["ЭЦН", "ПЭД", "Гидрозащита", "Клапан сливной", None])
        out = assign_mode_group_series(s).tolist()
        self.assertEqual(out, ["hydraulic", "electro-thermal", "protector", "other", ""])


class MappingExportTests(unittest.TestCase):
    def test_frame_covers_named_groups_only(self):
        frame = mode_group_frame()
        # OTHER is fallthrough — should not appear as explicit rows.
        self.assertEqual(set(frame["mode_group"]), {"hydraulic", "electro-thermal", "protector"})
        # Nine explicitly-mapped labels (5 + 3 + 1).
        self.assertEqual(len(frame), 9)

    def test_no_duplicate_nodes(self):
        frame = mode_group_frame()
        self.assertEqual(frame["failed_node"].nunique(), len(frame))

    def test_group_universe(self):
        self.assertEqual(MODE_GROUPS, ("hydraulic", "electro-thermal", "protector", "other"))

    def test_csv_roundtrip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mode_groups.csv"
            written = export_mode_group_csv(path)
            self.assertTrue(path.exists())
            reloaded = pd.read_csv(path)
            self.assertEqual(len(reloaded), len(written))
            self.assertListEqual(list(reloaded.columns), ["failed_node", "mode_group", "rationale"])


if __name__ == "__main__":
    unittest.main()
