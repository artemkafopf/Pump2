import unittest

import pandas as pd

from scripts.build_run_features import _h2s_proxy_frame


class H2SProxyTests(unittest.TestCase):
    def test_h2s_proxy_prefers_well_median_then_falls_back_to_pad_median(self):
        df = pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4, 5],
                "Скв.": ["W1", "W1", "W2", "W3", "W4"],
                "Месторождение": ["F1", "F1", "F1", "F1", "F2"],
                "Куст": ["P1", "P1", "P1", "P2", "P9"],
                "Массовая доля сероводорода, мг/дм³": [10.0, 30.0, "-", 50.0, "-"],
            }
        )

        result = _h2s_proxy_frame(df).set_index("row_id")

        self.assertEqual(float(result.loc[1, "h2s_proxy_mg_l"]), 20.0)
        self.assertEqual(str(result.loc[1, "h2s_proxy_source"]), "well_median")
        self.assertEqual(float(result.loc[2, "h2s_proxy_mg_l"]), 20.0)
        self.assertEqual(str(result.loc[2, "h2s_proxy_source"]), "well_median")
        self.assertEqual(float(result.loc[3, "h2s_proxy_mg_l"]), 20.0)
        self.assertEqual(str(result.loc[3, "h2s_proxy_source"]), "pad_median")
        self.assertEqual(float(result.loc[4, "h2s_proxy_mg_l"]), 50.0)
        self.assertEqual(str(result.loc[4, "h2s_proxy_source"]), "well_median")
        self.assertTrue(pd.isna(result.loc[5, "h2s_proxy_mg_l"]))
        self.assertEqual(str(result.loc[5, "h2s_proxy_source"]), "missing")


if __name__ == "__main__":
    unittest.main()
