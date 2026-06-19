import unittest
from unittest.mock import patch

import pandas as pd

from scripts.data_utils import add_dynamic_salt_proxies, attach_lab_chemistry


class DynamicLabProxyTests(unittest.TestCase):
    def test_attach_lab_chemistry_uses_step_fill_with_backward_extension(self):
        daily = pd.DataFrame(
            {
                "well_id": ["Vt_1", "Vt_1", "Vt_1", "Vt_1"],
                "dt": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05", "2026-01-07"]),
                "qliq": [10.0, 10.0, 10.0, 10.0],
            }
        )
        lab = pd.DataFrame(
            {
                "well_id": ["Vt_1", "Vt_1"],
                "sample_date": pd.to_datetime(["2026-01-02", "2026-01-06"]),
                "source_file": ["lab_a.xlsx", "lab_b.xlsx"],
                "chloride_mg_l": [1000.0, 2000.0],
                "sulfate_mg_l": [100.0, 200.0],
                "calcium_mg_l": [10.0, 20.0],
                "bicarbonate_mg_l": [1.0, 2.0],
                "magnesium_mg_l": [3.0, 4.0],
                "sodium_potassium_mg_l": [5.0, 6.0],
                "total_mineralization_g_l": [7.0, 8.0],
                "ph": [6.5, 6.8],
            }
        )

        with patch("scripts.data_utils._load_lab_samples", return_value=lab):
            result = attach_lab_chemistry(daily, fill_mode="step", extend_backward=True)

        self.assertEqual(result["chloride_mg_l"].tolist(), [1000.0, 1000.0, 1000.0, 2000.0])
        self.assertEqual(result["calcium_mg_l"].tolist(), [10.0, 10.0, 10.0, 20.0])
        self.assertEqual(result["lab_source_file"].astype(str).tolist(), ["lab_a.xlsx", "lab_a.xlsx", "lab_a.xlsx", "lab_b.xlsx"])

    def test_add_dynamic_salt_proxies_builds_daily_and_cumulative_loads(self):
        daily = pd.DataFrame(
            {
                "well_id": ["Vt_1", "Vt_1"],
                "dt": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "qliq": [10.0, 20.0],
            }
        )
        lab = pd.DataFrame(
            {
                "well_id": ["Vt_1"],
                "sample_date": pd.to_datetime(["2026-01-01"]),
                "source_file": ["lab.xlsx"],
                "chloride_mg_l": [1000.0],
                "sulfate_mg_l": [100.0],
                "calcium_mg_l": [10.0],
                "bicarbonate_mg_l": [1.0],
                "magnesium_mg_l": [3.0],
                "sodium_potassium_mg_l": [5.0],
                "total_mineralization_g_l": [7.0],
                "ph": [6.5],
            }
        )

        with patch("scripts.data_utils._load_lab_samples", return_value=lab):
            result = add_dynamic_salt_proxies(daily, qliq_column="qliq", fill_mode="step", extend_backward=True)

        self.assertEqual(result["daily_calcium_load_kg"].round(3).tolist(), [0.1, 0.2])
        self.assertEqual(result["daily_chloride_load_kg"].round(3).tolist(), [10.0, 20.0])
        self.assertEqual(result["daily_sulfate_load_kg"].round(3).tolist(), [1.0, 2.0])
        self.assertEqual(result["daily_salt_load_kg"].round(3).tolist(), [11.1, 22.2])
        self.assertEqual(result["cum_salt_load_kg_dynamic"].round(3).tolist(), [11.1, 33.3])


if __name__ == "__main__":
    unittest.main()
