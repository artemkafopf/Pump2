# Текущая модель риска отказов УЭЦН

Дата состояния: 2026-07-15  
Пакет модели: `2026-07-15-mc2023plus`  
Исполняемый пакет: `dist/Pump2ProductionRisk/`  
Основные выходные Excel-файлы: `results/`

## Короткий ответ: куда смотреть

Да, для готового поставочного варианта нужно смотреть в:

`D:\GitHub\Pump2\dist\Pump2ProductionRisk`

Там лежит:

- `Pump2ProductionRisk.exe` — собранный исполняемый файл;
- `ProductionRiskLauncher.xlsm` — Excel/VBA-лаунчер;
- `data/inputs/Отказы свод с анализом.xlsx` — источник отказов, скопированный в пакет;
- `results/esp_survival_vba_models/2026-07-15-mc2023plus/` — модельный bundle внутри frozen-пакета;
- `results/production_risk_forecast/2026-07-15/` — результаты smoke-test запуска из EXE.

Для проверки/аудита в исходном репозитории также важны:

- `results/esp_survival_vba_models/2026-07-15-mc2023plus/` — тот же принятый bundle до упаковки;
- `results/Риск_добычи_УЭЦН_2026_2027.xlsx`;
- `results/Прогноз_ремонтов.xlsx`;
- `results/Прогноз_ремонтов_hazard.xlsx`.

## Что модель делает сейчас

Модель прогнозирует риск отказа УЭЦН по скважинам и агрегирует эффект в:

1. ожидаемые отказы по месяцам;
2. ожидаемый простой ремонта;
3. потери добычи нефти/жидкости;
4. приоритеты замены насосов;
5. базовый и стрессовый сценарии.

Главная линия для презентации и планирования сейчас — `base_p50`: baseline survival
модель без ручных поправок, с P50 простоем ремонта. Стрессовая линия `stress_p75`
использует P75 простой и hazard overlay, включая Ql.

Период факта для проверки отказов: `2024-01..2026-06`.  
Период прогноза: `2026-07..2027-12`.

## Что было изменено в последней версии

### 1. Принят новый bundle

По умолчанию workflow теперь использует:

`results/esp_survival_vba_models/2026-07-15-mc2023plus/`

Это зафиксировано в `backend/analysis/workflows/production_risk/config.py`:

`BUNDLE_DATE = "2026-07-15-mc2023plus"`

### 2. Мирнинский baseline заменён на Mc `install_2023plus`

Главное изменение по survival-модели: строка `Mc_nonsour_Pooled` заменена с all-vintage
на вариант `install_2023plus`.

Причина: all-vintage хорошо смотрелся по строгому KM-критерию, но был слишком длинным
для текущей производственной популяции Мирнинского и недопрогнозировал фактические
отказы 2024-2026. Вариант `install_2023plus` попал в D/E fact/model gate:

| Поле | факт | модель | факт/модель |
|---|---:|---:|---:|
| Мирнинский УН | 41 | 41.27 | 0.99 |
| Ярактинский УН | 212 | 223.65 | 0.95 |
| Верхнетирский УН | 223 | 218.52 | 1.02 |
| ГЛОБАЛЬНО | 680 | 707.00 | 0.96 |

Все четыре контрольные строки проходят целевой коридор `0.90..1.10`.

### 3. Ручные калибровки сняты

Старые ручные множители больше не применяются:

- Ya/Vt model-field calibration снята;
- reporting-field calibration, включая stale Мирнинский 0.929, снята;
- глобальная строка больше не подгоняется через сумму ручных УН-поправок.

В коде:

`_CALIBRATION_FACTORS = {}`  
`_REPORTING_FIELD_CALIBRATION_FACTORS = {}`

### 4. Наблюдаемые отказы выровнены с A-population

В bundle добавлен файл:

`esp_observed_failures.csv`

Он задаёт фактический числитель отказов в той же популяции, на которой валидировалась
модель D/E. Это важно: сырой production счётчик давал `915` глобальных отказов, тогда
как A-population validation numerator даёт `680`. Теперь production chart использует
тот же числитель, что и validation gate.

### 5. Добавлен time-map / idle-hazard policy

В bundle есть `esp_time_map.csv`. Он используется как:

- карта размещения операционных дней внутри календарных месяцев;
- источник idle-hazard fraction для полей, где idle-risk прошёл D0 gate.

Важно: time-map не считается самостоятельной «уровневой» причиной исправления модели.
По D0 он полезен в основном как placement-map.

### 6. Excel, VBA и EXE пересобраны

Пересобрано:

- три deliverable Excel workbook;
- VBA launcher;
- PyInstaller EXE.

Launcher note теперь прямо говорит, что старые ручные поправки сняты.

## Базовая survival-модель

Каждая stratum-модель задаёт функцию выживания по операционным дням:

`S(t) = w1 * exp(-(t / eta1)^beta1) + (1 - w1) * exp(-(t / eta2)^beta2)`

Для `k1_*` моделей `w1 = 0`, и обе компоненты фактически совпадают. Для `k2` моделей
используется смесь двух Weibull-компонент.

Месячная вероятность отказа считается условно от текущего возраста:

`P(fail during next exposure) = 1 - S(age + op_days) / S(age)`

Где `age` и `op_days` измеряются в операционных днях, не в календарных.

### Ключевые параметры фокусных strata

| stratum | model | w1 | beta1 | eta1 | beta2 | eta2 | B20 | B50 | B80 | runs | failures | censored | Big censored | Mc window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Mc_nonsour_Pooled | k1_aic | 0.000 | 1.252 | 674.02 | 1.252 | 674.02 | 203.4 | 503.0 | 985.7 | 166 | 42 | 124 | 59 | install_2023plus |
| Ya_nonsour_Pooled | k1_aic | 0.000 | 0.798 | 706.05 | 0.798 | 706.05 | 107.9 | 446.1 | 1281.4 | 2118 | 1192 | 926 | 353 | |
| Vt_nonsour_Pooled | k2_high_w1 | 0.888 | 0.837 | 491.22 | 1.010 | 3404.02 | 94.2 | 375.8 | 1129.5 | 452 | 198 | 254 | 123 | |
| Vt_sour_Pooled | k1_aic | 0.000 | 0.920 | 146.76 | 0.920 | 146.76 | 28.8 | 98.5 | 246.2 | 164 | 111 | 53 | 29 | |
| Global_Pooled | k1_aic | 0.000 | 0.780 | 699.10 | 0.780 | 699.10 | 102.2 | 437.0 | 1286.8 | 4036 | 2096 | 1940 | 775 | |

Полный список параметров лежит в:

`results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_models.csv`

### Все strata в текущем registry

| stratum | model | field | h2s | contractor | B50 | B80 | runs | failures |
|---|---|---|---|---|---:|---:|---:|---:|
| Az_nonsour_brt | k1_aic | Az | nonsour | brt | 402.5 | 1112.0 | 183 | 90 |
| Az_nonsour_slb | k2 | Az | nonsour | slb | 420.9 | 901.3 | 101 | 56 |
| Da_nonsour_brt | k1_degenerate | Da | nonsour | brt | 1443.3 | 4293.9 | 122 | 47 |
| Ic_nonsour_brt | k1_aic | Ic | nonsour | brt | 540.2 | 1449.1 | 153 | 86 |
| Ic_nonsour_slb | k1_aic | Ic | nonsour | slb | 552.9 | 1636.6 | 115 | 60 |
| Vt_nonsour_brt | k1_degenerate | Vt | nonsour | brt | 555.9 | 1559.8 | 216 | 78 |
| Vt_nonsour_oth | k2 | Vt | nonsour | oth | 196.7 | 502.2 | 56 | 23 |
| Vt_nonsour_slb | k1_aic | Vt | nonsour | slb | 298.1 | 825.2 | 180 | 97 |
| Vt_sour_brt | k1_aic | Vt | sour | brt | 150.0 | 381.0 | 53 | 29 |
| Vt_sour_oth | k1_aic | Vt | sour | oth | 72.1 | 171.0 | 63 | 46 |
| Vt_sour_slb | k1_aic | Vt | sour | slb | 95.8 | 232.5 | 48 | 36 |
| Ya_nonsour_brt | k1_aic | Ya | nonsour | brt | 492.2 | 1432.6 | 1440 | 770 |
| Ya_nonsour_oth | k2 | Ya | nonsour | oth | 114.8 | 499.5 | 91 | 58 |
| Ya_nonsour_slb | k1_aic | Ya | nonsour | slb | 411.2 | 1084.3 | 587 | 364 |
| Za_nonsour_brt | k2 | Za | nonsour | brt | 332.5 | 733.3 | 167 | 85 |
| Za_nonsour_oth | k1_aic | Za | nonsour | oth | 193.7 | 557.6 | 42 | 24 |
| Za_nonsour_slb | k2 | Za | nonsour | slb | 426.5 | 952.5 | 80 | 36 |
| Az_nonsour_Pooled | k1_aic | Az | nonsour | Pooled | 359.1 | 1042.6 | 315 | 164 |
| Da_nonsour_Pooled | k1_aic | Da | nonsour | Pooled | 1388.3 | 4124.2 | 127 | 49 |
| Ic_nonsour_Pooled | k1_aic | Ic | nonsour | Pooled | 536.3 | 1517.0 | 275 | 151 |
| Vt_nonsour_Pooled | k2_high_w1 | Vt | nonsour | Pooled | 375.8 | 1129.5 | 452 | 198 |
| Vt_sour_Pooled | k1_aic | Vt | sour | Pooled | 98.5 | 246.2 | 164 | 111 |
| Ya_nonsour_Pooled | k1_aic | Ya | nonsour | Pooled | 446.1 | 1281.4 | 2118 | 1192 |
| Za_nonsour_Pooled | k2 | Za | nonsour | Pooled | 356.5 | 753.2 | 289 | 145 |
| Global_Pooled | k1_aic | Global | Pooled | Pooled | 437.0 | 1286.8 | 4036 | 2096 |
| Mc_nonsour_Pooled | k1_aic | Mc | nonsour | Pooled | 503.0 | 985.7 | 166 | 42 |

## Time map и idle-hazard

Файл: `esp_time_map.csv`.

Time-map оценивает ожидаемую долю операционных дней внутри календарного месяца по:

- field;
- возрастной группе по операционным дням;
- месяцу года;
- fallback `GLOBAL`.

Возрастные группы:

- `000_030`;
- `030_090`;
- `090_180`;
- `180_365`;
- `365_730`;
- `730_inf`;
- `ALL`.

В D0 true holdout:

| split | model | monthly op-day MAE | median total ННО abs % |
|---|---|---:|---:|
| holdout | scalar_field_trainfit | 8.86 | 19.2% |
| holdout | age_season_map_trainfit_unscaled | 7.91 | 22.3% |
| holdout | age_season_map_scaled_to_run_nno | 14.16 | 0.0% |

Вывод: map улучшает monthly placement примерно на `10.75%`, но unscaled total ННО хуже.
Поэтому в истории с известным ННО модель использует source ННО как clock total, а
time-map — как веса размещения внутри месяцев.

### Idle hazard

D0 re-attribution проверила, являются ли отказы в неработающих месяцах реальным idle
risk или артефактом даты съёма/остановки.

| field | recorded idle frac | re-attributed idle frac | решение |
|---|---:|---:|---|
| Мирнинский УН | 1.199 | 0.901 | real idle gate pass |
| Ярактинский УН | 0.672 | 0.328 | sensitivity/misdated |
| Верхнетирский УН | 0.571 | 0.414 | real idle gate pass |
| ГЛОБАЛЬНО | 0.688 | 0.412 | real idle gate pass |

В принятом scenario `A_plus_idle_hazard` idle exposure используется для строк, прошедших
gate. Это помогло особенно Vt/global и вместе с Mc 2023+ дало прохождение D1.

## Hazard layer

Hazard layer живёт в двух местах:

1. `esp_cox_coeffs.csv` — статические/ранние Cox-like коэффициенты;
2. Ql dynamic hazard в `config.py`.

Важно: hazard layer сейчас является стрессовым/чувствительным сценарием, а не причиной
прохождения основного D1 gate. Основной `base_p50` проходит без Ql stress overlay.

### Статические hazard coefficients

Файл: `esp_cox_coeffs.csv`.

Все строки имеют `enabled=TRUE`, но `ship_reason=physical_sensitivity_not_oos_validated`.
То есть слой поставляется как физическая sensitivity-линия, не как OOS-доказанное
улучшение baseline.

| covariate | ship name | group | beta | HR | p | reference | expected dir |
|---|---|---|---:|---:|---:|---:|---|
| log_glf_mean_opdays | GLF | GLF | -0.093609 | 0.9106 | 0.00600 | 5.621112 | - |
| load_std_early | load_variability | load | 0.019991 | 1.0202 | 0.00003 | 6.705360 | + |
| load_mean | load_level | load | 0.000527 | 1.0005 | 0.70623 | 41.428521 | + |
| frac_kpod_below_0p7 | chronic_underload | kpod | 0.200673 | 1.2222 | 0.01721 | 0.390201 | + |
| kpod_freq_mean | delivery_coef | kpod | -0.038754 | 0.9620 | 0.58084 | 0.799889 | ? |
| freq_above_55hz_pct_early | freq_above_55hz | frequency | 0.002483 | 1.0025 | 0.00734 | 19.934982 | + |
| n_freq_steps_per_100d | freq_instability | frequency | 0.004307 | 1.0043 | 0.10682 | 4.295203 | + |
| curvature_deg10m | curvature | curvature | -0.067567 | 0.9347 | 0.15846 | 0.725964 | + |
| log_run_seq | run_seq | well_history | 0.343544 | 1.4099 | 0.00004 | 1.245238 | + |
| log_days_since_prev_failure | days_since_fail | well_history | 0.015307 | 1.0154 | 0.53619 | 3.033118 | + |
| install_pre2020 | vintage_pre2020 | vintage | 0.442248 | 1.5562 | 0.00002 | 0.243201 | + |
| install_2023plus | vintage_2023plus | vintage | 0.208865 | 1.2323 | 0.01190 | 0.413180 | + |

### Ql hazard

Ql hazard включён конфигурационно:

`QL_HAZARD_ENABLED = True`

Но применяется в stress/hazard сценарии, а не в baseline. Формула covariate:

`z = clip(log((1 + Ql_m3d) / (1 + Ql_ref_field_m3d)), +/- log(5))`

Где:

- `Ql_m3d` — жидкость, нормированная на отработанные дни;
- reference — field-specific log reference;
- cap = `log(5) = 1.6094379124341003`.

Возрастная функция:

`theta = exp(z * (beta + gamma * log(max(age, 1))))`

Параметры (исправленный знак, Workstream C, 2026-07-15):

- `beta = 0.070` (было `-0.587058`);
- `gamma = 0.0` (было `0.093473` — возрастное взаимодействие убрано);
- source = `production_risk_hazard_refit_c/2026-07-15`;
- covariate name = `log_mean_qliq_m3d_field_ref_clip5`.

Исправление знака: прежний Extended-Cox давал физически обратный эффект (больше Ql → ниже
риск у молодого насоса) из-за коллинеарного члена `z*log(age)`. Serve-consistent месячный
рефит показал, что эмпирическая связь положительная (больше Ql → больше отказов). Проверка
лагов (lead t+3 beta=+0.84 против lag t-3 beta=+0.09) показала, что большая часть сырой
связи — обратная причинность (скважины «шумят» перед подъёмом), остаётся небольшой прямой
эффект. Поэтому Ql теперь — простой положительный PH-множитель (`gamma=0`, `beta=+0.07`),
только в stress-сценарии, НЕ в `base_p50` (OOS не улучшает). При 5× Ql поля theta ≈ 1.12.

Field reference logs:

| field | ref log |
|---|---:|
| Az | 3.946800 |
| Da | 3.689805 |
| Ic | 3.889624 |
| Mc | 3.782201 |
| Vt | 4.016012 |
| Ya | 4.253127 |
| Za | 3.832684 |
| GLOBAL | 4.050671 |

Интерпретация Ql:

- знак теперь единый на всех возрастах: больше Ql → выше риск (физически прямой);
- эффект скромный и обрезан (±log5): при 5× Ql поля theta ≈ 1.12, при 1/5 ≈ 0.90;
- это stress/sensitivity overlay (what-if по интенсификации), не в `base_p50`;
- он не является обязательной частью объяснения, почему основной baseline прошёл D1.

## Scenarios

Текущие сценарии:

| scenario_id | смысл |
|---|---|
| base_p25 | baseline survival + P25 repair downtime |
| base_p50 | primary baseline survival + P50 repair downtime |
| base_p75 | baseline survival + P75 repair downtime |
| stress_p75 | hazard/stress overlay + P75 repair downtime |

Downtime fallback:

| percentile | days |
|---|---:|
| P25 | 7 |
| P50 | 16 |
| P75 | 44 |

Primary scenario: `base_p50`.

## Как маппятся скважины на strata

Код скважины маппится по prefix:

| prefix | model field |
|---|---|
| YA | Ya |
| VT, VTI, VTB, VTBB | Vt |
| IC | Ic |
| MC, MR | Mc |
| AZ, AZA | Az |
| AU, AUY, AUZ | Za |
| DA, DAD | Da |

Некоторые prefix сознательно отправлены в `Global_Pooled`, а не в чужое поле:

`BT`, `KI`, `MSH`, `NE`, `AM`, `ZYI`, `YAY`.

Их нужно воспринимать как fallback, а не как ошибку маппинга.

## Что в Excel/EXE считается источником истины

Для packaged поставки:

`D:\GitHub\Pump2\dist\Pump2ProductionRisk\Pump2ProductionRisk.exe`

читает bundle из:

`D:\GitHub\Pump2\dist\Pump2ProductionRisk\_internal\results\esp_survival_vba_models\2026-07-15-mc2023plus\`

При запуске из исходного репозитория Python читает bundle из:

`D:\GitHub\Pump2\results\esp_survival_vba_models\2026-07-15-mc2023plus\`

Обе версии были проверены. Frozen EXE smoke-test дал те же focus ratios:

| field | observed | predicted | fact/model |
|---|---:|---:|---:|
| Мирнинский УН | 41 | 41.27 | 0.99 |
| Ярактинский УН | 212 | 223.65 | 0.95 |
| Верхнетирский УН | 223 | 218.52 | 1.02 |
| ГЛОБАЛЬНО | 680 | 707.00 | 0.96 |

## Что ещё важно понимать

1. Mc 2023+ принят осознанно: strict A6 `dS < 0.05` не прошёл, но production fact/model
   на текущей популяции прошёл лучше all-vintage. Это governance decision.
2. Time-map — не «магическая коррекция уровня», а mostly placement model.
3. Ql hazard — stress/sensitivity, не основной baseline.
4. Старые ручные поправки сняты; если они появятся снова, это будет regression.
5. Minor fields с Global_Pooled fallback остаются менее объяснимыми; фокусная
   презентационная зона — Mc/Ya/Vt/global.

## Где лежат артефакты

Параметры модели:

- `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_models.csv`
- `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_time_map.csv`
- `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_cox_coeffs.csv`
- `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_run_covariates.csv`
- `results/esp_survival_vba_models/2026-07-15-mc2023plus/esp_observed_failures.csv`

Validation:

- `results/production_risk_joint_validation/2026-07-15-mc2023plus/`
- `results/production_risk_mc_variant_validation/2026-07-15/`

Deliverables:

- `results/Риск_добычи_УЭЦН_2026_2027.xlsx`
- `results/Прогноз_ремонтов.xlsx`
- `results/Прогноз_ремонтов_hazard.xlsx`
- `dist/Pump2ProductionRisk/Pump2ProductionRisk.exe`
- `dist/Pump2ProductionRisk/ProductionRiskLauncher.xlsm`
