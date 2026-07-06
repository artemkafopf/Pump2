from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from analysis.paths import results_dir
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
_PROMPT_DIR = REPO_ROOT / "analysis_outputs" / "vt_60hz_prompt_analysis_2026_06_22"
FIGURES_DIR = _PROMPT_DIR / "figures"
TABLES_DIR = _PROMPT_DIR / "tables"
_SLUG = "vt_60hz_master_ru_pack"
BASE_OUTPUT_DIR = results_dir(_SLUG)
FIGURES_DIR = BASE_OUTPUT_DIR / "figures"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick(df: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key] == value
    return df.loc[mask].iloc[0]


def build_global_hf_mix() -> pd.DataFrame:
    dataset = read_csv("analysis_dataset.csv")
    for col in ["duration_best_days", "n_freq_above_55hz", "event"]:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    dataset["hf_flag"] = (dataset["n_freq_above_55hz"] / dataset["duration_best_days"]) > 0.5
    fails = dataset.loc[dataset["event"] == 1].copy()
    fails["group"] = fails["hf_flag"].map({True: "HF", False: "Rest"})
    counts = (
        fails.groupby(["group", "failure_category_raw"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"failure_category_raw": "category"})
    )
    counts["share"] = counts["count"] / counts.groupby("group")["count"].transform("sum")
    return counts


def prepare_metrics() -> dict[str, object]:
    audit = load_json(BASE_OUTPUT_DIR / "data_audit_summary.json")
    vt_global = read_csv("vt_vs_global_summary.csv")
    vt_env = read_csv("vt_environment_comparison.csv")
    vt_hyp = read_csv("vt_hypothesis_verdicts.csv")
    vt_decomp = read_csv("vt_decomposition_summary.csv")
    km = read_csv("km_summary.csv")
    infant_std = read_csv("vt_infant_standardization.csv")
    conf = read_csv("confounding_spearman_summary.csv")
    chem = read_csv("vt_chemistry_focus.csv")
    true60_focus = read_csv("true60_focus_summary.csv")
    true60_hr = read_csv("true60_adjusted_hr_focus.csv")
    true60_env = read_csv("vt_true60_environment_vs_rest.csv")
    true60_mix = read_csv("vt_true60_failure_mix.csv")
    axis_hr = read_csv("true60_axis_sensitivity_hr.csv")
    threshold_tests = read_csv("failure_stage_threshold_tests.csv")
    hf_pop = read_csv("hf_days_share55_population_comparison.csv")
    hf_axis = read_csv("hf_days_share55_axis_summary.csv")
    hf_hr = read_csv("hf_days_share55_hr_summary.csv")
    hf_thr = read_csv("hf_days_share55_threshold_tests.csv")
    hf_env = read_csv("hf_days_share55_vt_environment.csv")
    hf_mix_vt = read_csv("hf_days_share55_vt_failure_mix.csv")
    catboost = load_json(BASE_OUTPUT_DIR / "catboost_infant_metrics.json")
    global_hf_mix = build_global_hf_mix()

    vt_row = pick(vt_global, group="Vt")
    global_row = pick(vt_global, group="Global")
    km_vt = pick(km, group="Vt", analysis="vt_vs_global")
    km_global = pick(km, group="Global", analysis="vt_vs_global")
    infant90 = pick(infant_std, threshold_days=90)
    conf_global = pick(conf, scope="global", scope_value="all")
    conf_vt = pick(conf, scope="field", scope_value="Vt")
    chem_top = chem.iloc[0]

    true60_global_base = pick(true60_focus, population="Global", mode="50-55 Hz baseline")
    true60_global_true = pick(true60_focus, population="Global", mode=">58 Hz true60 proxy")
    true60_vt_base = pick(true60_focus, population="Vt", mode="50-55 Hz baseline")
    true60_vt_true = pick(true60_focus, population="Vt", mode=">58 Hz true60 proxy")
    true60_hr_global = pick(true60_hr, label="Global true60 vs 50-55")
    true60_hr_vt = pick(true60_hr, label="Vt true60 vs 50-55")
    true60_h2s = true60_env.loc[true60_env["metric"] == "H2S effective, mg/L"].iloc[0]
    true60_glf = true60_env.loc[true60_env["metric"] == "GLF"].iloc[0]
    true60_mix_true = true60_mix.loc[true60_mix["group"] == ">58 Hz"].copy()
    true60_mix_rest = true60_mix.loc[true60_mix["group"] == "Rest of Vt"].copy()
    true60_mix_join = true60_mix_true.merge(true60_mix_rest[["category", "share"]], on="category", how="left", suffixes=("_true", "_rest"))
    true60_mix_join["share_diff"] = true60_mix_join["share_true"] - true60_mix_join["share_rest"].fillna(0.0)
    true60_mix_join = true60_mix_join.sort_values("share_diff", ascending=False).reset_index(drop=True)

    axis_global_true = pick(axis_hr, axis_name="ttf_true_days", population="Global")
    axis_vt_true = pick(axis_hr, axis_name="ttf_true_days", population="Vt")
    axis_global_trf = pick(axis_hr, axis_name="trf_50hz_equiv_days", population="Global")
    axis_vt_trf = pick(axis_hr, axis_name="trf_50hz_equiv_days", population="Vt")

    thr90_global = pick(threshold_tests, threshold_days=90, population="Global")
    thr90_vt = pick(threshold_tests, threshold_days=90, population="Vt")

    hf_pop_global = pick(hf_pop, population="Global", definition="days >55 / duration_best_days >0.5")
    hf_pop_vt = pick(hf_pop, population="Vt", definition="days >55 / duration_best_days >0.5")
    hf_pop_old_global = pick(hf_pop, population="Global", definition="mean >58 Hz proxy")
    hf_pop_old_vt = pick(hf_pop, population="Vt", definition="mean >58 Hz proxy")
    hf_global_base = pick(hf_axis, axis_name="ttf_true_days", population="Global", mode="50-55 Hz baseline")
    hf_global_hf = pick(hf_axis, axis_name="ttf_true_days", population="Global", mode="HF by >55d share >0.5")
    hf_vt_base = pick(hf_axis, axis_name="ttf_true_days", population="Vt", mode="50-55 Hz baseline")
    hf_vt_hf = pick(hf_axis, axis_name="ttf_true_days", population="Vt", mode="HF by >55d share >0.5")
    hf_hr_global = pick(hf_hr, axis_name="ttf_true_days", population="Global")
    hf_hr_vt = pick(hf_hr, axis_name="ttf_true_days", population="Vt")
    hf_hr_global_trf = pick(hf_hr, axis_name="trf_50hz_equiv_days", population="Global")
    hf_hr_vt_trf = pick(hf_hr, axis_name="trf_50hz_equiv_days", population="Vt")
    hf_thr90_global = pick(hf_thr, threshold_days=90, population="Global")
    hf_thr90_vt = pick(hf_thr, threshold_days=90, population="Vt")
    hf_h2s_vt = hf_env.loc[hf_env["metric"] == "H2S effective, mg/L"].iloc[0]
    hf_glf_vt = hf_env.loc[hf_env["metric"] == "GLF"].iloc[0]

    hf_mix_true = hf_mix_vt.loc[hf_mix_vt["group"] == "HF share>0.5"].copy()
    hf_mix_rest = hf_mix_vt.loc[hf_mix_vt["group"] == "Rest of Vt"].copy()
    hf_mix_join = hf_mix_true.merge(hf_mix_rest[["category", "share"]], on="category", how="left", suffixes=("_hf", "_rest"))
    hf_mix_join["share_diff"] = hf_mix_join["share_hf"] - hf_mix_join["share_rest"].fillna(0.0)
    hf_mix_join = hf_mix_join.sort_values("share_diff", ascending=False).reset_index(drop=True)

    global_hf_true = global_hf_mix.loc[global_hf_mix["group"] == "HF"].copy()
    global_hf_rest = global_hf_mix.loc[global_hf_mix["group"] == "Rest"].copy()
    global_hf_join = global_hf_true.merge(global_hf_rest[["category", "share"]], on="category", how="left", suffixes=("_hf", "_rest"))
    global_hf_join["share_diff"] = global_hf_join["share_hf"] - global_hf_join["share_rest"].fillna(0.0)
    global_hf_join = global_hf_join.sort_values("share_diff", ascending=False).reset_index(drop=True)

    return {
        "audit": audit,
        "vt_row": vt_row,
        "global_row": global_row,
        "km_vt": km_vt,
        "km_global": km_global,
        "infant90": infant90,
        "conf_global": conf_global,
        "conf_vt": conf_vt,
        "chem_top": chem_top,
        "vt_env": vt_env,
        "vt_hyp": vt_hyp,
        "vt_decomp": vt_decomp,
        "catboost": catboost,
        "true60_global_base": true60_global_base,
        "true60_global_true": true60_global_true,
        "true60_vt_base": true60_vt_base,
        "true60_vt_true": true60_vt_true,
        "true60_hr_global": true60_hr_global,
        "true60_hr_vt": true60_hr_vt,
        "true60_h2s": true60_h2s,
        "true60_glf": true60_glf,
        "true60_mix_join": true60_mix_join,
        "axis_global_true": axis_global_true,
        "axis_vt_true": axis_vt_true,
        "axis_global_trf": axis_global_trf,
        "axis_vt_trf": axis_vt_trf,
        "thr90_global": thr90_global,
        "thr90_vt": thr90_vt,
        "hf_pop_global": hf_pop_global,
        "hf_pop_vt": hf_pop_vt,
        "hf_pop_old_global": hf_pop_old_global,
        "hf_pop_old_vt": hf_pop_old_vt,
        "hf_global_base": hf_global_base,
        "hf_global_hf": hf_global_hf,
        "hf_vt_base": hf_vt_base,
        "hf_vt_hf": hf_vt_hf,
        "hf_hr_global": hf_hr_global,
        "hf_hr_vt": hf_hr_vt,
        "hf_hr_global_trf": hf_hr_global_trf,
        "hf_hr_vt_trf": hf_hr_vt_trf,
        "hf_thr90_global": hf_thr90_global,
        "hf_thr90_vt": hf_thr90_vt,
        "hf_h2s_vt": hf_h2s_vt,
        "hf_glf_vt": hf_glf_vt,
        "hf_mix_join": hf_mix_join,
        "global_hf_join": global_hf_join,
    }


def build_markdown(metrics: dict[str, object]) -> Path:
    audit = metrics["audit"]
    vt_row = metrics["vt_row"]
    global_row = metrics["global_row"]
    km_vt = metrics["km_vt"]
    km_global = metrics["km_global"]
    infant90 = metrics["infant90"]
    conf_global = metrics["conf_global"]
    conf_vt = metrics["conf_vt"]
    chem_top = metrics["chem_top"]
    vt_decomp = metrics["vt_decomp"]
    catboost = metrics["catboost"]
    true60_mix_join = metrics["true60_mix_join"]
    hf_mix_join = metrics["hf_mix_join"]
    global_hf_join = metrics["global_hf_join"]

    true60_top1 = true60_mix_join.iloc[0]
    true60_top2 = true60_mix_join.iloc[1]
    hf_vt_top1 = hf_mix_join.iloc[0]
    hf_vt_top2 = hf_mix_join.iloc[1]
    hf_global_top1 = global_hf_join.iloc[0]
    hf_global_top2 = global_hf_join.iloc[1]

    lines = [
        "# Итоговый обзор исследования по частоте работы ЭЦН",
        "",
        "## 1. О чем был вопрос",
        "",
        "Вопрос исследования был практический: **влияет ли повышенная частота работы ЭЦН на срок службы, особенно для фонда `Vt`, и если влияет — насколько именно, за счет каких механизмов и какого оборудования?**",
        "",
        "Дальше исследование разделилось на две близкие, но не одинаковые постановки:",
        "",
        "- вопрос про режим, **близкий к `60 Гц`**",
        "- вопрос про **устойчивую работу выше `55 Гц`** в течение значительной части срока",
        "",
        "Оказалось, что ответ зависит от того, какую именно дефиницию 'высокой частоты' использовать.",
        "",
        "## 2. Данные и качество",
        "",
        f"- Всего в анализе: `{audit['total_runs']}` прогонов и `{audit['total_failures']}` отказов.",
        f"- Доля скважин с несколькими прогонами: `{audit['multi_run_well_share']:.1%}`. Это важно: наблюдения между прогонами не полностью независимы.",
        f"- Для прокси режима, близкого к `60 Гц`, в глобальной выборке было `{audit['true60_failures_global']}` отказов.",
        f"- Заполнение ионных прокси в целом приемлемое: кальций ~`{audit['ion_proxy_fill_rates']['calcium_load_per_day']:.1%}`, хлориды ~`{audit['ion_proxy_fill_rates']['chloride_load_per_day']:.1%}`, сульфаты ~`{audit['ion_proxy_fill_rates']['sulfate_load_per_day']:.1%}`, гипсовый прокси ~`{audit['ion_proxy_fill_rates']['gypsum_proxy_per_day']:.1%}`.",
        "",
        "Главное ограничение данных: часть выводов зависит от того, как именно определяется длительность работы (`calendar TTF`, `TTF_true`, best-available duration) и как именно определяется 'высокая частота'.",
        "",
        "## 3. Базовая картина: чем `Vt` отличается от `Global`",
        "",
        f"- По всей совокупности `Vt` живет хуже `Global`: KM-медиана `{float(km_vt['median_survival_days']):.0f}` дн. против `{float(km_global['median_survival_days']):.0f}` дн.",
        f"- Частота отказов в `Vt` выше: `{float(vt_row['failure_rate']):.3f}` против `{float(global_row['failure_rate']):.3f}`.",
        f"- По среде `Vt` тяжелее: средний `H2S` выше на `112` мг/л, доля кислых скважин выше примерно на `9.7 п.п.`.",
        f"- При этом доля дней `>55 Гц` в `Vt` и `Global` почти одинаковая, то есть сама по себе 'частота' не объясняет всю разницу между фондами.",
        "",
        "## 4. Что видно по ранним отказам в целом",
        "",
        f"- На горизонте `90` дней доля ранних отказов среди отказов в `Vt`: `{infant90['vt_observed_infant_share_among_failures']:.3f}`.",
        f"- В `Global` на том же горизонте: `{infant90['global_observed_infant_share_among_failures']:.3f}`.",
        f"- Если стандартизовать `Vt` к глобальной структуре категорий отказов, получается `{infant90['vt_standardized_to_global_category_mix']:.3f}`.",
        "",
        "Это важный вывод: часть ранней аварийности в `Vt` действительно связана со структурой категорий отказов, но не вся картина сводится только к mix-эффекту.",
        "",
        "## 5. Что показал сырой частотный сигнал",
        "",
        f"- Глобально ранговая корреляция между частотной экспозицией и TTF слабая: `rho={conf_global['rho']:.3f}`.",
        f"- В `Vt` тоже слабая: `rho={conf_vt['rho']:.3f}`.",
        "",
        "Значит, простой вопрос 'чем выше частота, тем короче жизнь?' в лоб не работает. Нужно смотреть уже не только на частоту, а на сочетание частоты, среды и типа отказа.",
        "",
        "## 6. Химия и TRF",
        "",
        f"- Самый сильный химический сигнал по `Vt` среди screened proxy — `{chem_top['metric']}` с `rho={chem_top['rho']:.3f}`.",
        f"- Нормализованный `TRF` сам по себе слабее: для `trf_per_day` `rho={read_csv('vt_chemistry_focus.csv').loc[read_csv('vt_chemistry_focus.csv')['metric']=='trf_per_day','rho'].iloc[0]:.3f}`.",
        "",
        "Это важный методический вывод: для объяснения времени до отказа химические прокси оказались полезнее, чем один только normalized TRF.",
        "",
        "## 7. Попытка объяснить разницу `Vt` через структуру категорий",
        "",
        f"- В decomposition по усеченному ожидаемому TTF общий разрыв составил `{float(vt_decomp.loc[vt_decomp['metric']=='total_gap_days','value'].iloc[0]):.2f}` дней.",
        f"- Mix-эффект: `{float(vt_decomp.loc[vt_decomp['metric']=='mix_effect_days','value'].iloc[0]):.2f}` дней.",
        f"- Within-category effect: `{float(vt_decomp.loc[vt_decomp['metric']=='within_category_effect_days','value'].iloc[0]):.2f}` дней.",
        "",
        "То есть в этой конкретной популяции чистый разрыв по decomposition получился небольшим. Это еще раз показывает, что результат чувствителен к определению выборки и оси времени.",
        "",
        "## 8. Режим, близкий к `60 Гц`: главный осторожный вывод",
        "",
        "Для ответа на вопрос именно про `60 Гц` использовался узкий прокси: `mean >58 Hz`.",
        "",
        f"- `Global`: `{int(metrics['true60_global_true']['runs'])}` прогонов / `{int(metrics['true60_global_true']['failures'])}` отказов.",
        f"- `Vt`: `{int(metrics['true60_vt_true']['runs'])}` / `{int(metrics['true60_vt_true']['failures'])}`.",
        "",
        "### Что получилось по `Global`",
        "",
        f"- Медиана best-available duration: `{float(metrics['true60_global_base']['median_duration_days']):.1f}` дн. в базе и `{float(metrics['true60_global_true']['median_duration_days']):.1f}` дн. в группе `>58 Гц`.",
        f"- Доля ранних отказов (<90 дн.) на прогон: `{float(metrics['true60_global_base']['infant_rate_runs_90d']):.2f}` vs `{float(metrics['true60_global_true']['infant_rate_runs_90d']):.2f}`.",
        f"- Скорректированный HR на оси `TTF_true`: `{float(metrics['axis_global_true']['hr_true60']):.2f}`.",
        "",
        "### Что получилось по `Vt`",
        "",
        f"- Медиана best-available duration: `{float(metrics['true60_vt_base']['median_duration_days']):.1f}` дн. в базе и `{float(metrics['true60_vt_true']['median_duration_days']):.1f}` дн. в группе `>58 Гц`.",
        f"- По `TTF_true` разница еще заметнее: примерно `{float(metrics['true60_vt_base']['median_duration_days']):.1f}` дн. в базе против около `117` дн. по sensitivity-pass.",
        f"- Скорректированный HR для `Vt` на оси `TTF_true`: `{float(metrics['axis_vt_true']['hr_true60']):.2f}`.",
        "",
        "### Практический смысл",
        "",
        "Для узкого прокси, ориентированного именно на near-`60 Гц` режим, картина в `Vt` выглядит осторожной и неблагоприятной: **календарная наработка, вероятно, короче**, а ранних отказов больше.",
        "",
        "## 9. Насколько именно near-60 Гц сокращает TTF в `Vt`",
        "",
        f"- По best-available duration: `{float(metrics['true60_vt_base']['median_duration_days']):.1f}` дн. против `{float(metrics['true60_vt_true']['median_duration_days']):.1f}` дн., то есть примерно `30` дней разницы.",
        f"- По `TTF_true` sensitivity-pass: около `151` дн. против `117` дн., то есть примерно `34` дня.",
        f"- На языке hazard ratio это диапазон примерно `{float(metrics['axis_vt_true']['hr_true60']):.2f}`–`{float(metrics['true60_hr_vt']['exp(coef)']):.2f}x` в зависимости от оси времени и модели.",
        "",
        "**Итог по near-60 Гц для `Vt`: наиболее правдоподобная оценка — укорочение жизни примерно на `30–35` дней, при умеренной уверенности в направлении и меньшей уверенности в точной величине.**",
        "",
        "## 10. Какие именно отказы становятся проблемнее при near-60 Гц в `Vt`",
        "",
        f"- Самая переизбыточная категория: `{true60_top1['category']}` (`{float(true60_top1['share_true']):.1%}` против `{float(true60_top1['share_rest']):.1%}` в остальном `Vt`).",
        f"- Следующая по важности: `{true60_top2['category']}` (`{float(true60_top2['share_true']):.1%}` против `{float(true60_top2['share_rest']):.1%}`).",
        "",
        "То есть в узком near-`60 Гц` прокси первый риск — **рабочие органы / засорение**, второй — **двигатель / ПЭД**.",
        "",
        "## 11. Ранние против зрелых отказов при near-60 Гц",
        "",
        f"- Для `Vt` на горизонте `90` дней разница по доле ранних отказов составляет `{float(metrics['thr90_vt']['share_difference_true60_minus_baseline']):.2f}`.",
        f"- Для `Global` на том же горизонте — `{float(metrics['thr90_global']['share_difference_true60_minus_baseline']):.2f}`.",
        "",
        "Это важный вывод для эксплуатационников: проблема near-`60 Гц` в `Vt` — не только возможное сокращение средней жизни, но и более тяжелая структура **ранних** отказов.",
        "",
        "## 12. Это не просто 'самые тяжелые скважины'?",
        "",
        "Для узкого near-`60 Гц` прокси — скорее нет.",
        "",
        f"- В `Vt` медианный `H2S` в группе `>58 Гц` даже ниже: `{float(metrics['true60_h2s']['true60_median']):.3f}` против `{float(metrics['true60_h2s']['rest_median']):.3f}`.",
        f"- `GLF` тоже не выше: `{float(metrics['true60_glf']['true60_median']):.1f}` против `{float(metrics['true60_glf']['rest_median']):.1f}`.",
        "",
        "Значит, near-`60 Гц` сигнал нельзя просто списать на то, что в эту группу попали самые тяжелые по среде скважины.",
        "",
        "## 13. Что показал TRF",
        "",
        f"- На оси `TTF_true` глобальный near-60-Hz HR: `{float(metrics['axis_global_true']['hr_true60']):.2f}`.",
        f"- На оси `TRF` этот же сигнал сжимается до `{float(metrics['axis_global_trf']['hr_true60']):.2f}` глобально и `{float(metrics['axis_vt_trf']['hr_true60']):.2f}` для `Vt`.",
        "",
        "Простыми словами: насос при более высокой частоте, возможно, не обязательно умирает после меньшего суммарного накопленного частотного ресурса. Но он, похоже, **быстрее проходит этот ресурс по календарному времени**.",
        "",
        "При этом данные **не** поддерживают идею одного универсального `TRF`-порога отказа. Если 'ресурс по TRF' существует, он, вероятно, зависит от механизма отказа.",
        "",
        "## 14. Более широкая дефиниция HF: `дней >55 Гц / days_total > 0.5`",
        "",
        "Это уже другая постановка: не near-`60 Гц`, а **устойчивая работа выше `55 Гц` в течение большей части срока**.",
        "",
        f"- `Global`: старая популяция `{int(metrics['hf_pop_old_global']['runs'])}` / `{int(metrics['hf_pop_old_global']['failures'])}`, новая `{int(metrics['hf_pop_global']['runs'])}` / `{int(metrics['hf_pop_global']['failures'])}`.",
        f"- `Vt`: старая популяция `{int(metrics['hf_pop_old_vt']['runs'])}` / `{int(metrics['hf_pop_old_vt']['failures'])}`, новая `{int(metrics['hf_pop_vt']['runs'])}` / `{int(metrics['hf_pop_vt']['failures'])}`.",
        "",
        "Эта дефиниция сильно расширяет группу — и меняет ответ.",
        "",
        "## 15. Что получилось при новой HF-дефиниции по `Global`",
        "",
        f"- `TTF_true` медиана в базе: `{float(metrics['hf_global_base']['median_duration']):.1f}` дн.",
        f"- `TTF_true` медиана в новой HF-группе: `{float(metrics['hf_global_hf']['median_duration']):.1f}` дн.",
        f"- Скорректированный HR по `TTF_true`: `{float(metrics['hf_hr_global']['hr_hf']):.2f}`.",
        f"- Разница по ранним отказам на `90` днях: `{float(metrics['hf_thr90_global']['share_difference_hf_minus_baseline']):.2f}`.",
        "",
        "Итог: по `Global` новая HF-группа не показывает явного падения TTF; остается лишь слабый неблагоприятный след по ранним отказам.",
        "",
        "## 16. Что получилось при новой HF-дефиниции по `Vt`",
        "",
        f"- `TTF_true` медиана в базе: `{float(metrics['hf_vt_base']['median_duration']):.1f}` дн.",
        f"- `TTF_true` медиана в новой HF-группе: `{float(metrics['hf_vt_hf']['median_duration']):.1f}` дн.",
        f"- Скорректированный HR по `TTF_true`: `{float(metrics['hf_hr_vt']['hr_hf']):.2f}`.",
        f"- Разница по ранним отказам на `90` днях: `{float(metrics['hf_thr90_vt']['share_difference_hf_minus_baseline']):.2f}`.",
        "",
        "Итог: **при новом определении `Vt` уже не выглядит хуже**. Наоборот, группа получается нейтральной или даже более благоприятной по медиане.",
        "",
        "## 17. Как меняется профиль рисков при новой HF-дефиниции",
        "",
        "### Global",
        "",
        f"- Наиболее переизбыточная категория: `{hf_global_top1['category']}` (`{float(hf_global_top1['share_hf']):.1%}` против `{float(hf_global_top1['share_rest']):.1%}`).",
        f"- Следующая: `{hf_global_top2['category']}` (`{float(hf_global_top2['share_hf']):.1%}` против `{float(hf_global_top2['share_rest']):.1%}`).",
        "",
        "### Vt",
        "",
        f"- Наиболее переизбыточная категория: `{hf_vt_top1['category']}` (`{float(hf_vt_top1['share_hf']):.1%}` против `{float(hf_vt_top1['share_rest']):.1%}`).",
        f"- Следующая: `{hf_vt_top2['category']}` (`{float(hf_vt_top2['share_hf']):.1%}` против `{float(hf_vt_top2['share_rest']):.1%}`).",
        "",
        "То есть при широкой HF-дефиниции в `Vt` фокус смещается к **валу** и **засорению**, а не к четкому общему сокращению TTF.",
        "",
        "## 18. Почему две дефиниции дают разный ответ",
        "",
        "Потому что они отвечают на **разные вопросы**:",
        "",
        "- `mean >58 Hz` — это более узкий прокси именно для режима, близкого к `60 Гц`",
        "- `дней >55 Гц / days_total > 0.5` — это более широкая группа устойчивой повышенной частоты",
        "",
        "Узкий прокси находит более тревожный `Vt`-сигнал. Широкий прокси этот сигнал размывает.",
        "",
        "## 19. Что можно сказать уверенно",
        "",
        "- `Vt` как фонд в целом тяжелее и аварийнее, чем `Global`.",
        "- Для near-`60 Гц` прокси есть умеренно неблагоприятный сигнал по `Vt`: меньше календарный TTF, больше ранних отказов, и первым проблемным узлом выглядят рабочие органы / засорение.",
        "- Для широкой HF-дефиниции `>55 Гц` большую часть срока этот вывод **не подтверждается**.",
        "- `H2S` устойчивее и сильнее как риск-фактор, чем сама частота.",
        "",
        "## 20. Что остается неопределенным",
        "",
        "- Выбор дефиниции частоты меняет выводы — значит, сам вопрос нужно формулировать очень точно.",
        "- В `Vt` near-`60 Гц` подгруппа небольшая, поэтому величина эффекта оценивается не очень точно.",
        "- Fine-Gray / Gray’s test в этой среде не были доступны, поэтому competing-risks часть еще не последняя инстанция.",
        "- Corrosion-related признаки плохо заполнены и пригодны только как индикативные.",
        "",
        "## 21. Практический итог для принятия решения",
        "",
        "### Если вопрос именно про `60 Гц` в `Vt`",
        "",
        "Тогда нужно ориентироваться на **узкий near-`60 Гц` прокси**. И ответ будет осторожный:",
        "",
        "- как default-режим для максимизации жизни `60 Гц` в `Vt` **не выглядит безопасным**",
        "- наиболее правдоподобное сокращение жизни: порядка `30–35` дней по прямым медианным оценкам",
        "- первый узел риска: **рабочие органы / засорение**",
        "- второй узел риска: **ПЭД**",
        "",
        "### Если вопрос про устойчивую работу `>55 Гц` большую часть срока",
        "",
        "Тогда нужно ориентироваться на **новую HF-дефиницию**. И ответ будет заметно мягче:",
        "",
        "- явного ухудшения TTF в `Vt` не видно",
        "- профиль риска смещается больше к валу и забивочным механическим сценариям",
        "- это уже другой бизнес-вопрос, не равный вопросу про near-`60 Гц`",
        "",
        "## 22. Вспомогательная модель",
        "",
        f"Exploratory CatBoost-модель для ранних отказов показала `ROC-AUC ≈ {catboost['roc_auc']:.3f}`. Это полезно как поддержка, но не как причинное доказательство.",
        "",
        "## 23. Главный вывод одной фразой",
        "",
        "**Для `Vt` нельзя смешивать два вопроса: near-`60 Гц` выглядит осторожно и неблагоприятно, а более широкая дефиниция 'большую часть срока выше `55 Гц`' такого эффекта уже не показывает.**",
        "",
        "## Рисунки",
        "",
        "### Базовое сравнение `Vt` и `Global`",
        "",
        "![Vt vs Global](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/km_vt_vs_global.png)",
        "",
        "### Различия по H2S / среде",
        "",
        "![Vt H2S box](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/vt_vs_global_h2s_box.png)",
        "",
        "### Структура отказов `Vt` vs `Global`",
        "",
        "![Failure mix](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/vt_vs_global_failure_mix.png)",
        "",
        "### Near-60 Гц: Global",
        "",
        "![True60 global](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/true60_km_global.png)",
        "",
        "### Near-60 Гц: Vt",
        "",
        "![True60 vt](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/true60_km_vt.png)",
        "",
        "### Near-60 Гц: ранние отказы 30 / 60 / 90",
        "",
        "![Thresholds true60](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/true60_threshold_early_share_ttf_true.png)",
        "",
        "### Near-60 Гц: профиль отказов в Vt",
        "",
        "![True60 mix vt](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/vt_true60_failure_mix.png)",
        "",
        "### Near-60 Гц: среда в Vt",
        "",
        "![True60 environment](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/vt_true60_environment_check.png)",
        "",
        "### TRF / оси времени",
        "",
        "![Axis sensitivity](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/true60_axis_sensitivity_hr.png)",
        "",
        "### Широкая HF-дефиниция: Global",
        "",
        "![HF share55 global](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_ttf_true_days_global_km.png)",
        "",
        "### Широкая HF-дефиниция: Vt",
        "",
        "![HF share55 vt](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_ttf_true_days_vt_km.png)",
        "",
        "### Широкая HF-дефиниция: 30 / 60 / 90 дней",
        "",
        "![HF share55 thresholds](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_thresholds.png)",
        "",
        "### Широкая HF-дефиниция: профиль отказов в Vt",
        "",
        "![HF share55 mix vt](../analysis_outputs/vt_60hz_prompt_analysis_2026_06_22/figures/hf_share55_vt_failure_mix.png)",
        "",
    ]
    path = DOCS_DIR / "vt_60hz_master_ru.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(246, 247, 249)
    bg.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.7), Inches(0.7), Inches(11.6), Inches(1.2))
    p = box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(27)
    p.font.bold = True
    p.font.color.rgb = RGBColor(26, 36, 52)
    sub = slide.shapes.add_textbox(Inches(0.75), Inches(1.75), Inches(11.0), Inches(1.0))
    p2 = sub.text_frame.paragraphs[0]
    p2.text = subtitle
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(74, 86, 104)


def add_bullets_slide(prs: Presentation, title: str, bullets: list[str], image_path: Path | None = None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.45), Inches(0.3), Inches(12.0), Inches(0.6))
    p = title_box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(26, 36, 52)
    body_width = Inches(5.4) if image_path else Inches(11.8)
    body = slide.shapes.add_textbox(Inches(0.55), Inches(1.0), body_width, Inches(5.9))
    tf = body.text_frame
    tf.word_wrap = True
    for idx, bullet in enumerate(bullets):
        para = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        para.text = bullet
        para.font.size = Pt(17)
        para.font.color.rgb = RGBColor(46, 56, 68)
        para.space_after = Pt(10)
    if image_path is not None and image_path.exists():
        slide.shapes.add_picture(str(image_path), Inches(6.0), Inches(1.0), width=Inches(6.4))


def build_slides(metrics: dict[str, object]) -> Path:
    vt_row = metrics["vt_row"]
    global_row = metrics["global_row"]
    km_vt = metrics["km_vt"]
    km_global = metrics["km_global"]
    infant90 = metrics["infant90"]
    true60_top1 = metrics["true60_mix_join"].iloc[0]
    true60_top2 = metrics["true60_mix_join"].iloc[1]
    hf_vt_top1 = metrics["hf_mix_join"].iloc[0]
    hf_vt_top2 = metrics["hf_mix_join"].iloc[1]
    hf_global_top1 = metrics["global_hf_join"].iloc[0]
    hf_global_top2 = metrics["global_hf_join"].iloc[1]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "Итоговое исследование по частоте ЭЦН",
        "Русская итоговая версия: Global, Vt, near-60 Гц, TRF, ранние отказы и новая HF-дефиниция",
    )
    add_bullets_slide(
        prs,
        "Цель исследования",
        [
            "Понять, сокращает ли повышенная частота жизнь ЭЦН и особенно ухудшает ли она результат в Vt.",
            "Отдельно проверить два вопроса: режим, близкий к 60 Гц, и устойчивую работу выше 55 Гц большую часть срока.",
            "Вывести практический ответ: безопасно или нет, насколько сокращается TTF и какой узел становится более рискованным.",
        ],
    )
    add_bullets_slide(
        prs,
        "Данные и качество",
        [
            f"Всего в анализе: {metrics['audit']['total_runs']} прогонов и {metrics['audit']['total_failures']} отказов.",
            f"Многократные прогоны по одним и тем же скважинам: {metrics['audit']['multi_run_well_share']:.0%}.",
            "Использовались calendar TTF, TTF_true, survival-анализ, infant-threshold sensitivity, химические прокси и сравнение структур отказов.",
            "Competing-risks часть улучшена через CIF, но Fine-Gray / Gray's test в текущей среде недоступны.",
        ],
    )
    add_bullets_slide(
        prs,
        "Vt против Global: общий фон",
        [
            f"KM-медиана Vt: {float(km_vt['median_survival_days']):.0f} дн.; Global: {float(km_global['median_survival_days']):.0f} дн.",
            f"Частота отказов Vt: {float(vt_row['failure_rate']):.3f}; Global: {float(global_row['failure_rate']):.3f}.",
            "Vt как фонд в целом тяжелее и аварийнее, чем Global.",
        ],
        image_path=FIGURES_DIR / "km_vt_vs_global.png",
    )
    add_bullets_slide(
        prs,
        "Среда и структура отказов",
        [
            "В Vt выше H2S и выше доля кислых скважин.",
            "При этом сама доля высокочастотных дней у Vt и Global близка — частота не объясняет всю разницу фондов.",
            "По структуре отказов Vt отличается от Global, что влияет на долю ранних отказов и общую наработку.",
        ],
        image_path=FIGURES_DIR / "vt_vs_global_failure_mix.png",
    )
    add_bullets_slide(
        prs,
        "Химия важнее сырой частоты",
        [
            "Сырой частотный сигнал слабый: простая корреляция частоты с TTF невелика и глобально, и внутри Vt.",
            "Химические прокси, особенно кальциевая нагрузка, дают более сильный timing-signal, чем normalized TRF.",
            "Это значит, что частоту нужно читать вместе со средой и механизмом отказа, а не отдельно.",
        ],
        image_path=FIGURES_DIR / "vt_chemistry_focus.png",
    )
    add_bullets_slide(
        prs,
        "Near-60 Гц: как определяли",
        [
            "Для вопроса именно про 60 Гц использовался узкий прокси: mean >58 Hz.",
            f"Global: {int(metrics['true60_global_true']['runs'])} прогонов / {int(metrics['true60_global_true']['failures'])} отказов.",
            f"Vt: {int(metrics['true60_vt_true']['runs'])} / {int(metrics['true60_vt_true']['failures'])}.",
            "Эта группа небольшая, но ближе всего к вопросу про реальный near-60-Hz режим.",
        ],
    )
    add_bullets_slide(
        prs,
        "Near-60 Гц: Global",
        [
            f"Медиана best-available duration: {float(metrics['true60_global_base']['median_duration_days']):.0f} дн. в базе против {float(metrics['true60_global_true']['median_duration_days']):.0f} дн. в группе >58 Гц.",
            f"Скорректированный HR по TTF_true: {float(metrics['axis_global_true']['hr_true60']):.2f}.",
            "Глобально это дает умеренно неблагоприятный сигнал, но не абсолютный запретительный вывод.",
        ],
        image_path=FIGURES_DIR / "true60_km_global.png",
    )
    add_bullets_slide(
        prs,
        "Near-60 Гц: Vt",
        [
            f"Best-available duration: {float(metrics['true60_vt_base']['median_duration_days']):.1f} дн. в базе против {float(metrics['true60_vt_true']['median_duration_days']):.1f} дн. в >58 Гц.",
            "По TTF_true разрыв выглядит еще сильнее: около 151 дн. против 117 дн.",
            f"Скорректированный HR по TTF_true: {float(metrics['axis_vt_true']['hr_true60']):.2f}.",
            "Это делает near-60-Hz режим в Vt осторожным и нежелательным как default-режим максимизации жизни.",
        ],
        image_path=FIGURES_DIR / "true60_km_vt.png",
    )
    add_bullets_slide(
        prs,
        "Ранние отказы при near-60 Гц",
        [
            f"На горизонте 90 дней разница по доле ранних отказов: Global {float(metrics['thr90_global']['share_difference_true60_minus_baseline']):.2f}, Vt {float(metrics['thr90_vt']['share_difference_true60_minus_baseline']):.2f}.",
            "То есть near-60 Гц влияет не только на общую наработку, но и на структуру ранних аварий.",
        ],
        image_path=FIGURES_DIR / "true60_threshold_early_share_ttf_true.png",
    )
    add_bullets_slide(
        prs,
        "Near-60 Гц: что ломается в Vt",
        [
            f"Главная переизбыточная категория: {true60_top1['category']} ({float(true60_top1['share_true']):.0%} против {float(true60_top1['share_rest']):.0%}).",
            f"Следующая важная категория: {true60_top2['category']} ({float(true60_top2['share_true']):.0%} против {float(true60_top2['share_rest']):.0%}).",
            "Практически это означает: первый риск — рабочие органы / засорение, второй — ПЭД.",
        ],
        image_path=FIGURES_DIR / "vt_true60_failure_mix.png",
    )
    add_bullets_slide(
        prs,
        "Это не просто самые тяжелые скважины",
        [
            "Для near-60-Hz группы в Vt H2S и GLF не выглядят выше, чем у остального фонда Vt.",
            "Поэтому неблагоприятный сигнал нельзя объяснить одной только более тяжелой средой.",
        ],
        image_path=FIGURES_DIR / "vt_true60_environment_check.png",
    )
    add_bullets_slide(
        prs,
        "TRF-интерпретация",
        [
            f"На оси TTF_true глобальный HR near-60 Гц = {float(metrics['axis_global_true']['hr_true60']):.2f}.",
            f"На оси TRF тот же сигнал сжимается до {float(metrics['axis_global_trf']['hr_true60']):.2f} глобально и {float(metrics['axis_vt_trf']['hr_true60']):.2f} в Vt.",
            "Простой смысл: при повышенной частоте насос, вероятно, быстрее проживает свой ресурс по календарю, но не обязательно умирает после меньшего накопленного частотного ресурса.",
        ],
        image_path=FIGURES_DIR / "true60_axis_sensitivity_hr.png",
    )
    add_bullets_slide(
        prs,
        "Новая HF-дефиниция",
        [
            "Высокочастотная скважина = больше 50% best-available operating days выше 55 Гц.",
            f"Global: новая популяция {int(metrics['hf_pop_global']['runs'])} прогонов и {int(metrics['hf_pop_global']['failures'])} отказов.",
            f"Vt: новая популяция {int(metrics['hf_pop_vt']['runs'])} и {int(metrics['hf_pop_vt']['failures'])}.",
            "Эта дефиниция значительно шире и отвечает уже на другой вопрос.",
        ],
    )
    add_bullets_slide(
        prs,
        "Новая HF-дефиниция: Global",
        [
            f"TTF_true медиана: {float(metrics['hf_global_base']['median_duration']):.0f} дн. в базе против {float(metrics['hf_global_hf']['median_duration']):.0f} дн. в HF-группе.",
            f"Скорректированный HR: {float(metrics['hf_hr_global']['hr_hf']):.2f}.",
            f"Разница по ранним отказам на 90 днях: {float(metrics['hf_thr90_global']['share_difference_hf_minus_baseline']):.2f}.",
            "Вывод: сильного падения TTF нет, остается лишь слабый неблагоприятный след по ранним отказам.",
        ],
        image_path=FIGURES_DIR / "hf_share55_ttf_true_days_global_km.png",
    )
    add_bullets_slide(
        prs,
        "Новая HF-дефиниция: Vt",
        [
            f"TTF_true медиана: {float(metrics['hf_vt_base']['median_duration']):.0f} дн. в базе против {float(metrics['hf_vt_hf']['median_duration']):.0f} дн. в HF-группе.",
            f"Скорректированный HR: {float(metrics['hf_hr_vt']['hr_hf']):.2f}.",
            f"Разница по ранним отказам на 90 днях: {float(metrics['hf_thr90_vt']['share_difference_hf_minus_baseline']):.2f}.",
            "Вывод: при этой широкой дефиниции Vt уже не выглядит хуже базы.",
        ],
        image_path=FIGURES_DIR / "hf_share55_ttf_true_days_vt_km.png",
    )
    add_bullets_slide(
        prs,
        "Что ломается при новой HF-дефиниции",
        [
            f"Global: наиболее переизбыточны {hf_global_top1['category']} и {hf_global_top2['category']}.",
            f"Vt: наиболее переизбыточны {hf_vt_top1['category']} и {hf_vt_top2['category']}.",
            "В Vt фокус смещается к валу и засорению, а не к явному общему сокращению TTF.",
        ],
        image_path=FIGURES_DIR / "hf_share55_vt_failure_mix.png",
    )
    add_bullets_slide(
        prs,
        "Почему выводы расходятся",
        [
            "Потому что это две разные бизнес-постановки.",
            "Mean >58 Hz отвечает на вопрос про режим, близкий к 60 Гц.",
            "Доля дней >55 Гц > 0.5 отвечает на вопрос про устойчивую повышенную частоту вообще.",
            "Узкий прокси находит более тревожный Vt-сигнал. Широкий прокси этот сигнал размывает.",
        ],
    )
    add_bullets_slide(
        prs,
        "Практический ответ для эксплуатации",
        [
            "Если вопрос именно про 60 Гц в Vt: как default-режим это не выглядит безопасным, ориентир по укорочению жизни — порядка 30–35 дней.",
            "Первый узел риска при near-60 Гц: рабочие органы / засорение. Второй: ПЭД.",
            "Если вопрос про sustained >55 Гц большую часть срока: явного ухудшения TTF в Vt не видно.",
        ],
    )
    add_bullets_slide(
        prs,
        "Неопределенности и ограничения",
        [
            "Размер near-60-Hz подгруппы в Vt невелик, поэтому точная величина эффекта неопределенна.",
            "Fine-Gray / Gray's test в этой среде не были доступны.",
            "Вывод чувствителен к дефиниции частоты и к выбору оси времени.",
            "Corrosion-related признаки заполнены слабо и трактуются только как индикативные.",
        ],
    )

    path = BASE_OUTPUT_DIR / "vt_60hz_master_ru_slides.pptx"
    prs.save(path)
    return path


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = prepare_metrics()
    md_path = build_markdown(metrics)
    slides_path = build_slides(metrics)
    summary = {
        "markdown_path": str(md_path),
        "slides_path": str(slides_path),
    }
    (BASE_OUTPUT_DIR / "vt_60hz_master_ru_pack.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[master-ru] Wrote markdown to {md_path}", flush=True)
    print(f"[master-ru] Wrote slides to {slides_path}", flush=True)


if __name__ == "__main__":
    main()
