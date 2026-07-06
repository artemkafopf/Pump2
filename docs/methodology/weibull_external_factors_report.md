# Внешние факторы и ресурс УЭЦН: Weibull-анализ

**База:** `data/warehouse/pump2.db` · `mart__vt_freq55`  
**Дата:** 2026-06-29  
**Таблицы:** `../../results/weibull_survival_uvch/2026-06-29/tables/`  
**Графики:** `../../results/weibull_survival_uvch/2026-06-29/figures/`

---

## Методология

Для каждого параметра — бинирование, затем на каждом бине:

- **KM-кривая** (Kaplan-Meier)
- **Weibull MLE** (1-comp, Nelder-Mead) поверх KM
- **Log-rank test** (Mantel-Cox, omnibus)

Анализ по 4 скоупам: **Vt / весь фонд / фонд без Vt** × **raw / Кэкспл>0.5**

Химические прокси нормированы: `cum_XXX_load_kg / run_days` — суточная нагрузка, без конфаундинга с длиной прогона.

### Границы бинов и распределение параметров

![Распределение Ca, Cl, SO₄, Гипс с границами tertile-бинов](../../results/weibull_survival_uvch/2026-06-29/figures/bin_boundaries_chemistry.png)

*Синий = низкий бин, зелёный = средний, красный = высокий. Цифры внутри: n / медиана raw / медиана Кэкспл>0.5 (дней). Штриховые линии = границы бинов (q₃₃ и q₆₇ по флоту).*

---

## 1. H2S

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_h2s.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_h2s.csv)  
→ Графики: [`survival_h2s.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_h2s.png) · [`survival_h2s_fleet_compare.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_h2s_fleet_compare.png)

**Бины (лог. шкала):** =0 / 0–0.1 / 0.1–1 / 1–10 / 10–100 / ≥100 мг/л

| Бин H2S | Vt raw, д | Fleet raw, д | Fleet_no_Vt raw, д | n в fleet_no_Vt |
|---------|----------|-------------|-------------------|-----------------|
| =0 | 183 | 251 | 266 | 282 |
| 0–0.1 | 210 | 256 | 261 | 217 |
| 0.1–1 | 220 | 285 | 294 | 521 |
| 1–10 | 179 | 218 | 232 | 139 |
| 10–100 | — | 164 | 167 | 156 |
| **≥100** | **68** | **80** | **136** ⚠ | **24** |

**log-rank p ≈ 0.000** во всех скоупах

### Вывод: пороговый эффект, сильно Vt-зависим

- При H2S < 10 мг/л выживаемость статистически неотличима от нулевого H2S.
- Порог — около **100 мг/л**: медиана Vt падает с 183 до 68 дн (в 2.7×), fleet — с 251 до 80 дн (в 3.1×).
- **76 из 100** прогонов fleet с H2S≥100 приходятся на Vt. Вне Vt n=24 — выборка мала, медиана 136 дн.
- Механизм: сульфидное растрескивание металлов и деградация изоляции кабеля при высоком H2S.

> **Для RUL-модели:** H2S использовать как binary feature: 0/1 при пороге ≥100 мг/л. Применимо прежде всего к Vt.

![H2S — KM + Weibull, Vt и весь фонд](../../results/weibull_survival_uvch/2026-06-29/figures/survival_h2s.png)

![H2S — сравнение весь фонд vs фонд без Vt](../../results/weibull_survival_uvch/2026-06-29/figures/survival_h2s_fleet_compare.png)

---

## 2. Кальций (Ca-нагрузка, кг/сут)

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_ca.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_ca.csv)  
→ Графики: [`survival_ca.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_ca.png) · [`survival_ca_fleet_compare.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_ca_fleet_compare.png)

**Бины:** tertiles по флоту: <3 680 (низкий) / 3 680–22 600 (средний) / >22 600 кг/сут (высокий)

| Бин Ca | Vt raw, д | Vt kekspl, д | Fleet raw, д | Fleet kekspl, д | Fleet_no_Vt raw, д |
|--------|----------|-------------|------------|----------------|-------------------|
| низкий | 208 | 280 | 392 | 554 | 415 |
| средний | 193 | 237 | 359 | 403 | **416** |
| **высокий** | **105** | **132** | **142** | **170** | **153** |

**log-rank p < 0.001** во всех скоупах

### Вывод: сильный тренд, частично Vt-зависим

- Высокая Ca-нагрузка сокращает медиану в **2.7–3.3×** (fleet: 392→142 raw, 554→170 kekspl).
- Без Vt в raw-данных бины низкий и средний почти идентичны (415 vs 416 дн) — тренд становится step-функцией: только высокий Ca резко падает. После Кэкспл-фильтра монотонный тренд восстанавливается (599→456→180 дн).
- Механизм: карбонатные (кальцитовые) отложения на рабочих колёсах и кабеле.

![Ca-нагрузка — KM + Weibull, Vt и весь фонд](../../results/weibull_survival_uvch/2026-06-29/figures/survival_ca.png)

![Ca-нагрузка — сравнение весь фонд vs фонд без Vt](../../results/weibull_survival_uvch/2026-06-29/figures/survival_ca_fleet_compare.png)

---

## 3. Хлориды (Cl-нагрузка, кг/сут)

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_cl.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_cl.csv)  
→ Графики: [`survival_cl.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_cl.png) · [`survival_cl_fleet_compare.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_cl_fleet_compare.png)

**Бины:** tertiles: <13 700 / 13 700–71 000 / >71 000 кг/сут

| Бин Cl | Vt raw, д | Vt kekspl, д | Fleet raw, д | Fleet kekspl, д | Fleet_no_Vt kekspl, д |
|--------|----------|-------------|------------|----------------|----------------------|
| низкий | 233 | 310 | 403 | 583 | 629 |
| средний | 187 | 226 | 336 | 387 | 387 |
| **высокий** | **87** | **112** | **140** | **166** | **179** |

**log-rank p ≈ 0.000** во всех скоупах

### Вывод: наиболее устойчивый тренд из всей химии

- **Ratio высокий/низкий = 0.28–0.37** — наибольший из всех химических параметров.
- Тренд monotone_neg сохраняется во всех скоупах, включая fleet_no_vt — это **общефлотовая закономерность**, не артефакт Vt.
- Механизм: хлориды коррелируют с общей минерализацией, прямое ускорение коррозии металлов и деградации изоляции.

> **Для RUL-модели:** наиболее информативный химический предиктор. Рекомендуется включить.

![Cl-нагрузка — KM + Weibull, Vt и весь фонд](../../results/weibull_survival_uvch/2026-06-29/figures/survival_cl.png)

![Cl-нагрузка — сравнение весь фонд vs фонд без Vt](../../results/weibull_survival_uvch/2026-06-29/figures/survival_cl_fleet_compare.png)

---

## 4. Сульфаты (SO4-нагрузка, кг/сут)

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_so4.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_so4.csv)  
→ Графики: [`survival_so4.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_so4.png) · [`survival_so4_fleet_compare.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_so4_fleet_compare.png)

**Бины:** tertiles: <24 / 24–184 / >184 кг/сут

| Бин SO4 | Vt raw, д | Vt kekspl, д | Fleet raw, д | Fleet kekspl, д | Fleet_no_Vt kekspl, д |
|---------|----------|-------------|------------|----------------|----------------------|
| низкий | 219 | 289 | 385 | 524 | 581 |
| средний | 154 | 181 | 301 | 338 | 355 |
| **высокий** | **123** | **156** | **172** | **206** | **217** |

**log-rank p < 0.01 (Vt), p < 0.001 (фонд)**

### Вывод: монотонный тренд, слабее Ca и Cl, высокоустойчив

- Ratio 0.37–0.56 — умеренный эффект.
- Monotone_neg во всех скоупах включая fleet_no_vt — устойчивый общефлотовый сигнал.
- SO4 добавляет независимую информацию к Ca и Cl.

![SO4-нагрузка — KM + Weibull, Vt и весь фонд](../../results/weibull_survival_uvch/2026-06-29/figures/survival_so4.png)

![SO4-нагрузка — сравнение весь фонд vs фонд без Vt](../../results/weibull_survival_uvch/2026-06-29/figures/survival_so4_fleet_compare.png)

---

## 5. Гипс-прокси (осадки, gypsum_proxy_m_mean)

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_gypsum.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_gypsum.csv)  
→ Графики: [`survival_gypsum.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_gypsum.png) · [`survival_gypsum_fleet_compare.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_gypsum_fleet_compare.png)

| Бин Gyp | Fleet raw, д | Fleet_no_Vt raw, д | p |
|---------|------------|-------------------|---|
| низкий | 427 | 441 | 0.76 |
| средний | 443 | — | |
| высокий | 465 | 484 | |

**p = 0.19–0.76 во всех скоупах**

### Вывод: незначим

При >300 событиях на бин (достаточная мощность) p=0.76 — реальное отсутствие эффекта. Ca и SO4 по отдельности имеют значимые тренды, их совместный продукт (CaSO4, гипс) — нет. Гипсовые отложения, вероятно, формируются преимущественно в НКТ/устьевом оборудовании, а не в насосе.

> **Для RUL-модели:** исключить из feature set.

![Гипс-прокси — KM + Weibull, Vt и весь фонд](../../results/weibull_survival_uvch/2026-06-29/figures/survival_gypsum.png)

![Гипс-прокси — сравнение весь фонд vs фонд без Vt](../../results/weibull_survival_uvch/2026-06-29/figures/survival_gypsum_fleet_compare.png)

---

## 6. Подрядчик

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_contractor.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_contractor.csv)  
→ Графики: [`survival_contractor.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_contractor.png) · [`survival_contractor_fleet_compare.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_contractor_fleet_compare.png)

| Подрядчик | Vt raw, д | Fleet raw, д | Fleet_no_Vt raw, д | n fleet | n на Vt |
|-----------|----------|------------|-------------------|---------|---------|
| **Борец** | **179** | **281** | **295** | 1251 | 123 |
| Шлюмберже | 157 | 237 | 260 | 666 | 132 |
| Нов. технологии | **71** | **89** | **119** | 90 | 52 |
| Прочие | — | 111 | 108 | 39 | — |

**log-rank p ≈ 0.000** во всех скоупах

### Вывод: НТ в 2.5× хуже Борца даже без Vt

- **58% прогонов НТ приходится на Vt** (52 из 90). Vt — специфическое месторождение с высоким H2S.
- Без Vt медиана НТ вырастает с 89 до 119 дн (+30 дн — вклад условий Vt).
- Но **НТ остаётся в 2.5× хуже Борца** (119 vs 295 дн) вне Vt — эффект подрядчика реален.

> **Caveat:** возможен selection bias — НТ могут преимущественно ставиться на осложнённый фонд. Без stratification по условиям скважины интерпретировать как причинно-следственный нельзя.

![Подрядчик — KM + Weibull, Vt и весь фонд](../../results/weibull_survival_uvch/2026-06-29/figures/survival_contractor.png)

![Подрядчик — сравнение весь фонд vs фонд без Vt](../../results/weibull_survival_uvch/2026-06-29/figures/survival_contractor_fleet_compare.png)

---

## 7. GLF (газовый фактор)

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_glf.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_glf.csv)  
→ Графики: [`survival_glf.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_glf.png) · [`survival_glf_fleet_compare.png`](../../results/weibull_survival_uvch/2026-06-29/figures/survival_glf_fleet_compare.png)

**Бины (фиксированные):** 0–100 / 100–300 / 300–700 / 700+

| Бин GLF | Vt raw, д | Fleet raw, д | Fleet_no_Vt raw, д |
|---------|----------|------------|-------------------|
| 0–100 | — (n=2, исключён) | 171 | 170 |
| 100–300 | 144 | 258 | 269 |
| 300–700 | 136 | 234 | 302 |
| **700+** | **181** | **293** | **317** |

**log-rank p < 0.001** для fleet и fleet_no_vt. Для Vt: p=0.15 (3 бина, n меньше).

### Вывод: нелинейный (U-образный) эффект, значим на флоте

- Тренд **non-monotone**: GLF 0–100 — наихудший бин (171 дн raw), GLF 700+ — наилучший (293 дн). Средние бины — промежуточные.
- Тренд **monotone_pos** на fleet_no_vt: без Vt медианы монотонно растут от 170 до 317 дней.
- Механизм: очень низкий GLF (0–100) → недостаточный газ на входе → нестабильная работа, срывы, кавитация. Высокий GLF (700+) → скважины с хорошей газосепарацией, часто более стабильные.
- На Vt только 3 бина (GLF 0–100 n=2 исключён), паттерн менее устойчив.

> **Для RUL-модели:** использовать как binned/categorical feature, не linear. Нелинейный эффект требует либо бинирования, либо сплайнов.

### Распределение и границы бинов GLF

![GLF — распределение с границами бинов](../../results/weibull_survival_uvch/2026-06-29/figures/bin_boundaries_glf.png)

![GLF — KM + Weibull, Vt и весь фонд](../../results/weibull_survival_uvch/2026-06-29/figures/survival_glf.png)

![GLF — сравнение весь фонд vs фонд без Vt](../../results/weibull_survival_uvch/2026-06-29/figures/survival_glf_fleet_compare.png)

---

## 8. Сводная матрица значимости

→ [`../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_param_summary.csv`](../../results/weibull_survival_uvch/2026-06-29/tables/weibull_by_param_summary.csv)

| Параметр | Тренд Vt | Тренд fleet | Тренд fleet_no_Vt | Устойчивость | Приоритет для RUL |
|----------|----------|-------------|-------------------|--------------|------------------|
| **Cl-нагрузка** | monotone_neg | monotone_neg | monotone_neg | ✅ Высокая | 🔴 Включить |
| **SO4-нагрузка** | monotone_neg | monotone_neg | monotone_neg | ✅ Высокая | 🔴 Включить |
| **Ca-нагрузка** | monotone_neg | monotone_neg | частично | ✅ Средняя | 🟡 Включить |
| **H2S** | пороговый | пороговый | слабый (n↓) | 🟡 Vt-зависим | 🟡 Binary ≥100 мг/л |
| **GLF** | n/s (p=0.15) | non-mono | monotone_pos | 🟡 Нелинейный | 🟡 Binned feature |
| **Подрядчик** | mono_neg | non-mono | monotone_neg | ✅ Высокая | 🟡 С осторожностью |
| Гипс | нет | нет | нет | ✅ Нет | ⚪ Исключить |

### Рекомендации

1. **Cl и SO4** — наиболее надёжные предикторы, работают везде. Включить как continuous features.
2. **Ca** — сильный предиктор, совместно с Cl несёт информацию о минерализации. Включить.
3. **H2S** — threshold feature: 1 при ≥100 мг/л. Применимо в первую очередь к Vt.
4. **GLF** — нелинейный (U-образный) эффект. Использовать как binned/categorical, не linear.
5. **Подрядчик** — значимый дифференциатор. Включить как categorical feature, но учесть возможный selection bias.
6. **Гипс** — исключить.

---

## 8. Взаимосвязи между параметрами (наблюдения)

- **НТ × Vt**: НТ концентрирован на Vt (58% прогонов), Vt — высокий H2S и Cl. Часть плохого результата НТ — это условия Vt, но не вся.
- **Ca × Cl корреляция**: оба отражают минерализацию пластовой воды. Возможна мультиколлинеарность в многофакторной модели — рекомендуется тест VIF перед включением обоих.
- **Кэкспл-фильтр усиливает химические эффекты**: после исключения нестабильных прогонов ratio high/low для Ca и Cl ухудшается (0.36→0.30 для Cl). Химические факторы актуальны именно для стабильно работающего оборудования.
