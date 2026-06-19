import unittest

import pandas as pd

from analysis.derived_feature_presets import suggest_derived_presets
from analysis.derived_features import apply_derived_columns, evaluate_derived_formula


class DerivedFeaturesTests(unittest.TestCase):
    def test_evaluate_derived_formula_supports_bracketed_columns(self):
        df = pd.DataFrame(
            {
                "Qliq_m3d": [80.0, 100.0],
                "pump_nominal_rate_m3d": [100.0, 100.0],
            }
        )

        result = evaluate_derived_formula(df, "[Qliq_m3d] / [pump_nominal_rate_m3d]")

        self.assertEqual(result.round(3).tolist(), [0.8, 1.0])

    def test_apply_derived_columns_supports_chained_formulas(self):
        df = pd.DataFrame(
            {
                "Qliq_m3d": [50.0, 75.0, 100.0],
                "pump_nominal_rate_m3d": [100.0, 100.0, 100.0],
            }
        )

        processed, notes = apply_derived_columns(
            df,
            [
                {"name": "Kpod", "formula": "[Qliq_m3d] / [pump_nominal_rate_m3d]"},
                {"name": "stress_low_Kpod", "formula": "max(0, 0.7 - [Kpod])"},
            ],
        )

        self.assertIn("Kpod", processed.columns)
        self.assertIn("stress_low_Kpod", processed.columns)
        self.assertEqual(processed["Kpod"].round(3).tolist(), [0.5, 0.75, 1.0])
        self.assertEqual(processed["stress_low_Kpod"].round(3).tolist(), [0.2, 0.0, 0.0])
        self.assertEqual(len(notes), 2)

    def test_apply_derived_columns_raises_for_missing_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})

        with self.assertRaisesRegex(ValueError, "referenced in formula was not found"):
            apply_derived_columns(df, [{"name": "b", "formula": "[missing] + 1"}])

    def test_apply_derived_columns_raises_for_duplicate_names(self):
        df = pd.DataFrame({"a": [1, 2, 3]})

        with self.assertRaisesRegex(ValueError, "defined more than once"):
            apply_derived_columns(
                df,
                [
                    {"name": "dup", "formula": "[a] + 1"},
                    {"name": "dup", "formula": "[a] + 2"},
                ],
            )

    def test_suggest_derived_presets_includes_kpod_and_pressure_ratio_family(self):
        df = pd.DataFrame(
            {
                "Qliq_m3d": [80.0],
                "pump_nominal_rate_m3d": [100.0],
                "reference_frequency_hz": [50.0],
                "frequency_hz": [45.0],
                "P_intake_atm": [80.0],
                "Pbubble_atm": [100.0],
            }
        )

        presets = suggest_derived_presets(df)
        preset_names = [preset.name for preset in presets]

        self.assertIn("Kpod", preset_names)
        self.assertIn("Kpod_freq_adjusted", preset_names)
        self.assertIn("pressure_ratio", preset_names)
        self.assertIn("frequency_to_reference_ratio", preset_names)
        self.assertIn("frequency_over_reference_hz", preset_names)
        self.assertIn("qliq_per_hz", preset_names)
        self.assertEqual(presets[0].formula, "[Qliq_m3d] / [pump_nominal_rate_m3d]")
        self.assertEqual(presets[2].formula, "[P_intake_atm] / [Pbubble_atm]")
        self.assertEqual(presets[0].category, "Hydraulic")
        self.assertEqual(presets[0].stress_preset.transform, "negative_excess")
        self.assertEqual(presets[0].stress_preset.reference_value, 0.7)
        self.assertEqual(presets[2].stress_preset.name, "stress_low_pressure_ratio")

    def test_suggest_derived_presets_includes_design_and_gas_features(self):
        df = pd.DataFrame(
            {
                "Qliq_m3d": [80.0],
                "pump_nominal_rate_m3d": [100.0],
                "design_Qliq_m3d": [100.0],
                "GLF_m3m3": [200.0],
                "design_GLF_m3m3": [150.0],
                "free_gas_intake_percent": [35.0],
                "design_free_gas_intake_percent": [20.0],
                "free_gas_limit_fraction": [0.25],
                "Pbubble_atm": [100.0],
                "P_intake_atm": [80.0],
                "design_Pbubble_atm": [110.0],
                "design_Pintake_atm": [88.0],
                "design_Kpod": [0.85],
                "motor_load_percent": [70.0],
                "design_motor_load_percent": [65.0],
                "watercut_percent": [60.0],
                "design_watercut_percent": [50.0],
                "frequency_hz": [52.0],
                "reference_frequency_hz": [50.0],
                "motor_power_kw": [30.0],
            }
        )

        presets = suggest_derived_presets(df)
        preset_names = [preset.name for preset in presets]

        self.assertIn("qliq_to_design_ratio", preset_names)
        self.assertIn("glf_to_design_ratio", preset_names)
        self.assertIn("free_gas_to_design_ratio", preset_names)
        self.assertIn("gas_excess_percent", preset_names)
        self.assertIn("motor_load_to_design_ratio", preset_names)
        self.assertIn("frequency_to_reference_ratio", preset_names)
        self.assertIn("design_pressure_ratio", preset_names)
        self.assertIn("kpod_to_design_ratio", preset_names)
        self.assertIn("pressure_ratio_to_design_ratio", preset_names)
        self.assertIn("motor_load_over_design_pp", preset_names)
        self.assertIn("frequency_over_reference_hz", preset_names)
        self.assertIn("watercut_to_design_ratio", preset_names)
        self.assertIn("qliq_per_hz", preset_names)
        self.assertIn("qliq_per_kw", preset_names)
        self.assertIn("glf_excess_over_design", preset_names)
        self.assertIn("free_gas_excess_over_design", preset_names)

    def test_suggest_derived_presets_recognizes_russian_column_aliases(self):
        df = pd.DataFrame(
            {
                "Дебит жидкости, м3/сут": [80.0],
                "Подача насоса": [100.0],
                "Частота, Гц": [48.0],
                "Базовая частота": [50.0],
                "Давление на приеме": [84.0],
                "Давление насыщения": [105.0],
                "Обводненность, %": [65.0],
                "Обводненность по подбору, %": [55.0],
            }
        )

        presets = suggest_derived_presets(df)
        preset_names = [preset.name for preset in presets]

        self.assertIn("Kpod", preset_names)
        self.assertIn("Kpod_freq_adjusted", preset_names)
        self.assertIn("pressure_ratio", preset_names)
        self.assertIn("frequency_to_reference_ratio", preset_names)
        self.assertIn("frequency_over_reference_hz", preset_names)
        self.assertIn("qliq_per_hz", preset_names)
        self.assertIn("watercut_to_design_ratio", preset_names)

    def test_suggest_derived_presets_recognizes_workbook_specific_aliases(self):
        df = pd.DataFrame(
            {
                "Дебит жидк.": [80.0],
                "Ном. Произв. м₃/сут": [100.0],
                "Частота": [48.0],
                "Номинальная частота, Гц": [50.0],
                "Загр, Двиг,": [72.0],
                "Мощность, кВт": [30.0],
                "Ток x.x": [28.0],
                "Ном. ток/ A": [30.0],
                "Рпл.": [180.0],
                "Рзаб": [95.0],
                "Дав. Нас": [105.0],
                "Глубина спуска УЭЦН, по НКТ": [2200.0],
                "ВГ, м": [1850.0],
                "Ном.напор (50Гц)": [1650.0],
                "Кол.ступеней": [180.0],
                "Работа в кривизне": [3.5],
                "Длина УЭЦН/метр": [28.0],
            }
        )

        presets = suggest_derived_presets(df)
        preset_names = [preset.name for preset in presets]

        self.assertIn("Kpod", preset_names)
        self.assertIn("Kpod_freq_adjusted", preset_names)
        self.assertIn("frequency_to_reference_ratio", preset_names)
        self.assertIn("frequency_over_reference_hz", preset_names)
        self.assertIn("qliq_per_hz", preset_names)
        self.assertIn("qliq_per_kw", preset_names)
        self.assertIn("current_to_nominal_ratio", preset_names)
        self.assertIn("qliq_per_current", preset_names)
        self.assertIn("bhp_to_reservoir_ratio", preset_names)
        self.assertIn("pressure_margin_to_bubble", preset_names)
        self.assertIn("submergence_margin_m", preset_names)
        self.assertIn("submergence_margin_ratio", preset_names)
        self.assertIn("nominal_head_per_stage", preset_names)
        self.assertIn("motor_load_per_hz", preset_names)
        self.assertIn("curve_work_per_meter", preset_names)

    def test_suggest_derived_presets_recognizes_glf_from_gzhf_column(self):
        df = pd.DataFrame(
            {
                "ГЖФ": [220.0],
                "design_GLF_m3m3": [180.0],
            }
        )

        presets = suggest_derived_presets(df)
        preset_names = [preset.name for preset in presets]

        self.assertIn("glf_to_design_ratio", preset_names)

    def test_suggest_derived_presets_includes_salt_and_scale_proxies(self):
        df = pd.DataFrame(
            {
                "Ca₂⁺, мг/л": [120.0],
                "Cl⁻, мг/л": [2200.0],
                "SO₄²⁻, мг/л": [340.0],
                "Дебит жидк.": [150.0],
                "Наработка (сут)": [180.0],
            }
        )

        presets = suggest_derived_presets(df)
        by_name = {preset.name: preset for preset in presets}

        self.assertIn("cum_calcium_load_kg", by_name)
        self.assertIn("cum_chloride_load_kg", by_name)
        self.assertIn("cum_sulfate_load_kg", by_name)
        self.assertIn("cum_salt_load_kg", by_name)
        self.assertIn("gypsum_scale_proxy", by_name)
        self.assertEqual(
            by_name["cum_calcium_load_kg"].formula,
            "([Ca₂⁺, мг/л] * [Дебит жидк.] * [Наработка (сут)]) / 1000",
        )
        self.assertEqual(
            by_name["cum_salt_load_kg"].formula,
            "([Ca₂⁺, мг/л] + [Cl⁻, мг/л] + [SO₄²⁻, мг/л]) * [Дебит жидк.] * [Наработка (сут)] / 1000",
        )


if __name__ == "__main__":
    unittest.main()
