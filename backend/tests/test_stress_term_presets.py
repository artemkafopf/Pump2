import unittest

import pandas as pd

from analysis.stress_term_presets import suggest_stress_term_presets


class StressTermPresetTests(unittest.TestCase):
    def test_suggest_stress_term_presets_prefers_kpod_over_duplicate_qliq_nominal_term(self):
        df = pd.DataFrame(
            {
                "Дебит жидк.": [80.0, 90.0, 100.0],
                "Ном. Произв. м₃/сут": [110.0, 110.0, 110.0],
                "Частота": [48.0, 50.0, 52.0],
                "Номинальная частота, Гц": [50.0, 50.0, 50.0],
                "Загр, Двиг,": [65.0, 70.0, 75.0],
                "Мощность, кВт": [30.0, 30.0, 30.0],
                "Кпрод.": [0.65, 0.75, 0.85],
                "Работа в кривизне": [1.0, 3.0, 5.0],
            }
        )

        presets = suggest_stress_term_presets(df)
        preset_names = [preset.name for preset in presets]

        self.assertIn("stress_low_Kpod", preset_names)
        self.assertNotIn("stress_qliq_vs_nominal", preset_names)
        self.assertIn("stress_frequency_over_reference", preset_names)
        self.assertIn("stress_high_motor_load", preset_names)
        self.assertIn("stress_low_qliq_per_kw", preset_names)
        self.assertIn("stress_high_curve_work", preset_names)

    def test_suggest_stress_term_presets_includes_qliq_vs_nominal_without_kpod(self):
        df = pd.DataFrame(
            {
                "Дебит жидк.": [80.0, 90.0, 100.0],
                "Ном. Произв. м₃/сут": [110.0, 110.0, 110.0],
            }
        )

        presets = suggest_stress_term_presets(df)
        preset_names = [preset.name for preset in presets]

        self.assertIn("stress_qliq_vs_nominal", preset_names)

    def test_suggest_stress_term_presets_builds_fit_defaults(self):
        df = pd.DataFrame(
            {
                "Частота": [45.0, 48.0, 50.0, 52.0, 55.0],
                "Загр, Двиг,": [60.0, 62.0, 67.0, 71.0, 75.0],
            }
        )

        presets = suggest_stress_term_presets(df)
        by_name = {preset.name: preset for preset in presets}

        self.assertEqual(by_name["stress_high_frequency"].term.reference_mode, "fit")
        self.assertIsNotNone(by_name["stress_high_frequency"].term.reference_init)
        self.assertEqual(by_name["stress_high_motor_load"].term.reference_mode, "fit")
        self.assertIsNotNone(by_name["stress_high_motor_load"].term.reference_init)


if __name__ == "__main__":
    unittest.main()
