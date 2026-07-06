# Независимая инженерно-статистическая оценка программы УВЧ для ЭЦН

## Входной файл

`C:\Users\alexe\Downloads\ВТЛУ_Программа УВЧ.xlsx`

---

## ГЛАВНАЯ ЗАДАЧА

Выполнить независимую оценку программы увеличения частоты ЭЦН для
списка скважин из Excel-файла и ответить на три вопроса:

1. Что произойдёт с режимом работы каждой скважины при переводе на
   целевую частоту?
2. Как изменится ресурс ЭЦН: ННО, МРП и риск раннего отказа?
3. Каков остаточный ресурс RUL на текущий момент и как он изменится
   после перевода на новый режим?

**Фиксированное допущение по всему анализу:**

- `freq_after = Fмакс.возм` из файла — единственный сценарий uplift
- задача не оптимизировать целевую частоту, а анализировать именно
  сценарий перевода на максимально допустимую частоту из файла

---

## КОНТЕКСТ ПРЕДЫДУЩЕГО АНАЛИЗА

Использовать как prior и calibration layer. Не подменять анализ
текущего файла. Если локальные данные противоречат prior — указывать явно.

### Ключевые prior-выводы

1. **Устойчивый режим >55 Hz (Y > 0.5)** в Vt: HR = 0.96, медиана
   TTF +20 дней, ранних отказов 27% vs 34% базиса — нейтральный
   результат, без автоматического штрафа по ресурсу.
2. **Near-60 proxy (mean > 58 Hz)**: HR = 1.34 в Vt (p=0.36, малая
   выборка n=24), −30–34 дня к медиане TTF — осторожно неблагоприятный
   сигнал. Для глобального фонда: HR = 1.31, p = 0.024.
3. Главные устойчивые драйверы короткой жизни в Vt: H₂S, химия,
   failure mix, ранние отказы — важнее частоты.
4. Фиксированного TRF-бюджета не существует (4 теста, KW p=0.0001).
5. Окно первых 90 дней критично: при near-60 в Vt **57% произошедших
   отказов** случились до 90-го дня против 34% в базисе. Это показатель
   структуры наблюдавшихся отказов, а не безусловная вероятность
   `P(fail <= 90d)` для всех насосов.
6. КЛ — #1 причина отказа в Vt (25%), η=91 дн. Большинство кабельных
   отказов — в первые 3 месяца.
7. ПЭД — единственный wear-out компонент (β=1.12). Во всём остальном
   фонде β < 1.
8. Majority-days >55 Hz и near-60 mean >58 Hz — разные определения
   экспозиции. Ни одно из них само по себе не доказывает стабильность
   или cycling: это проверяется отдельно по временным рядам телеметрии.
9. Near-60 Гц равномерно повышает риск по всем категориям отказа
   (HR 1.06–1.36 глобально). Нет одной специфической мишени.

### Параметры prior-моделей для подстановки в расчёты

**Однокомпонентный Weibull по категории отказа (Vt, 410 прогонов):**

| Категория    | β    | η (дн.) | Паттерн          |
|---|---|---|---|
| КЛ (кабель)  | 0.73 | **91**  | Детская смертность |
| ПЭД          | **1.12** | 155 | Wear-out (единственный) |
| Засорение РО | <1   | —       | Детская смертность |
| Слом вала    | <1   | —       | Детская смертность |
| НКТ          | <1   | —       | Детская смертность |

**Однокомпонентный Weibull M0 (Vt, fallback prior для основного RUL):**

```
eta_M0  = 184.4 дн.
beta_M0 = 0.808
n = 410 прогонов, 268 отказов
```

M0 и M2 были оценены на одной гибридной оси `duration`:
`ttf_true_best_days`, если доступна, иначе `run_days`. При наличии
Source B модель необходимо повторно оценить на одной явно выбранной
оси времени; не смешивать параметры моделей, медианы и RUL с разных
определений длительности.

**Двухкомпонентная смесь M2 (Vt, только sensitivity-модель):**

```
π_early  = 0.054,  η_early  = 4.6 дн.,  β_early  = 2.41
π_normal = 0.946,  η_normal = 202 дн.,  β_normal = 0.91

Медиана TTF_true (Vt)      ≈ 150 дн.
Медиана best_available (Vt) ≈ 235 дн.   ← описательный benchmark,
                                               не параметр RUL-модели
```

⚠ M2 плохо идентифицирована на Vt: Bootstrap CI для π_early =
[0.038, 0.787]. Поэтому M2 запрещено использовать как единственный
или основной производственный RUL. Если RUL из M2 и основной модели
расходятся более чем на 20% — приводить оба значения и понижать confidence.

**AJ CIF — базовые вероятности отказа (Vt, текущий режим):**

| Причина          | @30d  | @90d      | @180d | @365d |
|---|---|---|---|---|
| КЛ (кабель)      | 0.028 | 0.099     | 0.153 | 0.203 |
| ПЭД (двигатель)  | 0.014 | 0.064     | 0.118 | 0.172 |
| Засорение РО     | 0.018 | 0.073     | 0.095 | 0.118 |
| Слом вала        | 0.020 | 0.078     | 0.107 | 0.131 |
| **P(any failure)** | **~0.10** | **~0.34** | **~0.51** | **~0.67** |

**Cox HR эффекта частоты (калиброванные мультипликаторы):**

| Сценарий          | λ-multiplier | Источник |
|---|---|---|
| Majority-days >55 Hz (Y>0.5) | × 0.96 | HR Vt широкая дефиниция экспозиции |
| Near-60 mean >58 Hz | × 1.34 [0.71–2.51] | HR Vt near-60, TTF_true |
| Нет uplift        | × 1.00 | Baseline |

Показатель `57% / 34% = 1.68` не использовать как множитель риска.
Его знаменатель — число уже произошедших отказов, а не все насосы под
риском. Показывать его только как описательную метрику
`early_failure_composition_90d`. Вероятности `P(fail <= horizon)`
получать только из согласованной survival-модели с учётом цензурирования.

---

## ИЕРАРХИЯ ИСТОЧНИКОВ ДАННЫХ

Для каждой численной метрики указывать источник.

| Код | Источник | Приоритет |
|---|---|---|
| **A** | Excel-файл программы УВЧ | Основной |
| **B** | Локальная БД: mart__vt_freq55, raw__v03_runs, proc__daily_merged, ttf_true_best_days | Выше C при наличии |
| **C** | Prior из предыдущего анализа | Fallback |

Маркировка каждой метрики:
- `source_used`: A / A+B / A+C / B / C
- `is_direct_measurement`: True / False
- `confidence`: High / Medium / Low

---

## РАЗДЕЛ A. АУДИТ ВХОДНОГО ФАЙЛА

### A.1 Структура файла

Проверить структуру самостоятельно и явно зафиксировать итог.
Предварительная информация (верифицировать):
- Лист: **СВОД**
- Заголовок: строка 4
- Данные: строки 5–58 (54 скважины)
- Состав: преимущественно `Vt_`; также `Bt_009р`, `Bt_1503`

### A.2 Нормализация таблицы

| Поле | Источник / формула |
|---|---|
| `well` | col C |
| `nno_days` | col D — **текущая наработка** с последнего ремонта, дни |
| `q_liq_cur` | col E (м³/сут) |
| `q_oil_cur` | col F (т/сут) |
| `glf` | col G (ГЖФ) |
| `gas_before_sep_pct` | col H (% газа до сепарации) |
| `gas_after_sep_pct` | col I (% газа после сепарации) |
| `freq_cur` | col J (Гц) |
| `motor_load_pct` | col K (%) |
| `freq_after` | col L (Fмакс.возм — целевая частота uplift) |
| `q_liq_expected_file` | col M |
| `q_oil_expected_file` | col N |
| `d_freq` | = freq_after − freq_cur |
| `freq_ratio` | = freq_after / freq_cur |
| `q_liq_affinity` | = q_liq_cur × freq_ratio |
| `q_oil_affinity` | = q_oil_cur × freq_ratio |
| `motor_load_after_est` | = motor_load_pct × freq_ratio³ |
| `dq_oil_file` | = q_oil_expected_file − q_oil_cur |
| `is_bt_well` | = well.startswith("Bt_") |
| `is_vt_well` | = well.startswith("Vt_") |

**Определения (зафиксировать явно):**
- `ННО (nno_days)` = текущая наработка с последнего ремонта. Это
  возраст прогона, не исторический MTBF.
- `МРП` = межремонтный период = ожидаемая полная длительность
  прогона. В файле отсутствует. Fleet prior: медиана МРП ≈ 235 дн.
- `RUL` = remaining useful life = условное оставшееся время,
  вычисленное с учётом уже прожитого возраста nno_days.

### A.3 Скрининговые флаги

| Флаг | Условие |
|---|---|
| `flag_motor_overload` | motor_load_after_est > 85 |
| `flag_motor_high` | 75 < motor_load_after_est ≤ 85 |
| `flag_motor_near_limit_cur` | motor_load_pct > 75 (уже сейчас близко к лимиту) |
| `flag_gas_critical` | gas_after_sep_pct > 55 |
| `flag_gas_elevated` | 30 < gas_after_sep_pct ≤ 55 |
| `flag_headroom_minimal` | d_freq < 2 |
| `flag_headroom_low` | 2 ≤ d_freq < 4 |
| `flag_no_uplift` | freq_after = freq_cur (нет пространства) |
| `flag_near60_target` | freq_after ≥ 58 (прямой near-60 сценарий) |
| `flag_young_run` | nno_days < 30 |
| `flag_very_young` | nno_days < 90 |
| `flag_bt_well` | is_bt_well |
| `flag_file_deviation_liq` | \|q_liq_affinity − q_liq_expected_file\| / q_liq_affinity > 0.15 |
| `flag_file_deviation_oil` | \|q_oil_affinity − q_oil_expected_file\| / q_oil_affinity > 0.15 |

### A.4 Физические несоответствия

Явно перечислить:
- Скважины с `motor_load_after_est > 100%`
- Скважины с `flag_no_uplift`
- Скважины с `glf > 800`
- Скважины с `flag_gas_critical`
- `Bt_`-скважины: prior из Vt неприменим — выделить отдельно

---

## РАЗДЕЛ B. НЕЗАВИСИМАЯ ОЦЕНКА ОЖИДАЕМОГО РЕЖИМА

### B.1 Постановка

Цель: независимо оценить ожидаемый режим при `freq_after = Fмакс.возм`,
сравнить с тем, что записано в Excel, объяснить расхождения.

### B.2 Двухслойная оценка

**Layer 1 — Аффинный baseline (верхняя огибающая)**

```
q_liq_affinity = q_liq_cur × freq_ratio
q_oil_affinity = q_oil_cur × freq_ratio
```

Это не финальный прогноз, а идеальное максимальное значение.
Файловые значения тяготеют к этой формуле (без поправок).

**Layer 2 — Скорректированная независимая оценка**

Шаг 1: Газовая поправка (применять всегда)

```
gas_after_sep_pct > 55%   →  gas_correction = 0.80
gas_after_sep_pct 30–55%  →  gas_correction = 0.90
gas_after_sep_pct ≤ 30%   →  gas_correction = 1.00
```

Механизм: рост частоты сокращает время флюида в газосепараторе,
снижает эффективность сепарации и реальный прирост дебита жидкости.

Шаг 2: Моторная поправка (только при `use_motor_load_proxy = True`)

```
motor_load_after_est > 90%    →  motor_correction = 0.85
motor_load_after_est 85–90%   →  motor_correction = 0.92
иначе                         →  motor_correction = 1.00
```

Шаг 3: Финальные значения

```
q_liq_after_ind = q_liq_affinity × gas_correction × motor_correction
q_oil_after_ind = q_liq_after_ind × (q_oil_cur / q_liq_cur)
dq_oil_ind      = q_oil_after_ind − q_oil_cur
```

### B.3 Переключатель motor_load_proxy

Обязательно выдать оба варианта:

- `with_motor_load_proxy`: motor_correction применён
- `without_motor_load_proxy`: motor_correction = 1.00

⚠ Даже при `use_motor_load_proxy = True` трактовать
`motor_load_after_est = motor_load_pct × freq_ratio³` только как
скрининговый proxy. Формула предполагает постоянный КПД и
идеальные аффинные законы, что нарушается при сдвиге рабочей
точки на кривой H-Q. Для скважин с `flag_motor_near_limit_cur`
риск реальнее, чем кажется по формуле.

В portfolio summary: показать, как меняется N_green / N_red при
переключении proxy.

### B.4 Две независимые оси: частота и стабильность режима

Не выводить стабильность или cycling только из `freq_after`, загрузки
ПЭД либо газосодержания. Сначала присвоить непересекающийся целевой
диапазон частоты:

| `target_freq_band` | Условия, применять сверху вниз |
|---|---|
| **T0: no_uplift** | flag_no_uplift |
| **T1: below_55** | freq_after < 55 Hz |
| **T2: high_target_55_58** | 55 ≤ freq_after < 58 Hz |
| **T3: near60_target** | freq_after ≥ 58 Hz |

`T3` означает только целевую частоту около 60 Гц. Это не синоним
нестабильности, cycling или повышенного числа остановок.

Отдельно определить `engineering_constraint`:

| Значение | Условия |
|---|---|
| `constrained` | flag_motor_overload ИЛИ flag_gas_critical |
| `caution` | flag_motor_high ИЛИ flag_gas_elevated ИЛИ flag_headroom_low |
| `screen_ok` | перечисленных флагов нет |

### B.5 Проверка стабильности по телеметрии

Если доступна Source B, для каждой скважины рассчитать минимум за
последние 90 дней и, при наличии покрытия, за 180 дней:

- `telemetry_coverage_days`
- `active_day_share`: доля дней с `freq > 0` и/или `q_liq > 0`
- `zero_freq_days_share` и `zero_production_days_share`
- `stop_count`: число переходов из рабочего состояния в останов
- `stop_count_per_90d`
- `zero_interval_count` и медианная длительность zero-интервала
- `freq_mean_active`, `freq_sd_active`, `freq_cv_active`
- `days_share_gt55` и `days_share_ge58`

Сформировать два разных поля:

- `historical_stability`: `stable_history` / `cycling_history` /
  `uncertain_history` / `no_telemetry`
- `forward_stability`: `expected_stable` / `cycling_risk` / `unknown`

`historical_stability` определять по наблюдаемой телеметрии. Пороговые
значения для stop count, zero share и CV сначала оценить по распределению
Vt и зафиксировать как `[БД]` либо `[ЭКСПЕРТ]`; провести sensitivity к
порогам. `forward_stability` может быть `expected_stable` только при
достаточном покрытии телеметрии и явном плане непрерывной эксплуатации.
Если временных рядов или плана нет, использовать `unknown`, а не делать
вывод из одной строки Excel.

Отсутствие моторных или газовых флагов не является доказательством
стабильности. Эти флаги остаются только на оси engineering constraint.

### B.6 Сравнение с файлом

Для каждой скважины:
- **Реалистично**: расхождение < 15%, нет поправочных флагов
- **Файл оптимистичен**: q_liq_expected_file > q_liq_after_ind на >15%
- **Файл консервативен**: q_liq_expected_file < q_liq_after_ind на >15%
- **Требует проверки**: расхождение > 30%

Вероятная системная причина расхождений: файл использует чистый
аффинный закон без газовой и моторной поправок.

---

## РАЗДЕЛ C. ОЦЕНКА РЕСУРСА: ННО, МРП, RUL

### C.1 Основная модель текущего RUL

Основной RUL строить на одной заранее зафиксированной временной оси.
При наличии Source B:

1. Выбрать и описать одну duration-переменную, правила event/censoring
   и дату среза.
2. Построить KM и параметрическую M0 Weibull/AFT на той же выборке.
3. Выбрать основную модель по calibration и out-of-sample validation;
   при нарушении PH отдавать приоритет AFT/conditional KM, а Cox-HR
   использовать только как сценарный множитель будущего риска.

Если Source B недоступна, использовать Vt M0 как fallback:

```python
eta = 184.4
beta = 0.808

def H0(t):
    return (t / eta) ** beta

def S0(t):
    return exp(-H0(t))

def S_cond_current(u, a):
    # Пережить ещё u дней при уже прожитом возрасте a.
    return S0(a + u) / S0(a)

P_fail_current(horizon, a) = 1 - S_cond_current(horizon, a)
RUL_median_current: min u such that S_cond_current(u, a) <= 0.50
RUL_B10_current:    min u such that S_cond_current(u, a) <= 0.90
```

Для KM использовать ту же условную конструкцию
`S_KM(a + u) / S_KM(a)` только внутри надёжно наблюдаемого диапазона;
за последним устойчивым участком KM не экстраполировать.

### C.2 M2 только как sensitivity-модель

```python
def S_mix(t):
    S_e = exp(-(t / 4.6) ** 2.41)
    S_n = exp(-(t / 202) ** 0.91)
    return 0.054 * S_e + 0.946 * S_n

def S_mix_cond_current(u, a):
    return S_mix(a + u) / S_mix(a)
```

Не назначать M2 основной моделью и не смешивать её параметры с
медианой `best_available = 235 дн.`. Для молодых прогонов posterior
веса компонентов особенно чувствительны к плохо идентифицированной
`pi_early`; для зрелых прогонов ранняя компонента почти исчезает.
Если M2 и основная модель расходятся более чем на 20%, показать оба
результата, понизить confidence и использовать основной результат
для recommendation.

### C.3 Baseline при текущем режиме

Для каждой скважины вычислить:

- `p_fail_30d_cur`, `p_fail_90d_cur`, `p_fail_180d_cur`
- `rul_median_cur`, `rul_b10_cur`
- `rul_median_cur_m2_sensitivity`, `rul_b10_cur_m2_sensitivity`
- `nno_days` — только наблюдаемое значение из Excel, не model estimate
- `projected_run_length_p50_cur = nno_days + rul_median_cur`
- `mrp_vt_benchmark = 235 дн.` — только справочное fleet-значение на
  другой оси, не складывать с RUL и не использовать в формулах

При наличии Source B откалибровать модель локально и пометить
`source = A+B`. Без Source B результаты являются prior-driven.

### C.4 Сценарный эффект uplift

Множитель частоты задаётся по `target_freq_band`, а стабильность и
engineering constraints показываются отдельными полями. Не изменять
HR из-за motor/gas-флагов без отдельно оценённой модели взаимодействий.

| Целевой диапазон | Base multiplier `m` | Интерпретация |
|---|---|---|
| T0: no_uplift | 1.00 | Режим не меняется |
| T1: below_55 | 1.00 | Нет калиброванного uplift-эффекта |
| T2: high_target_55_58 | 0.96 | Exploratory перенос broad `Y>0.5`; применять только если план действительно предполагает большинство дней >55 Hz |
| T3: near60_target | 1.34 | Осторожный перенос HR для исторического `mean >58 Hz`, CI [0.71, 2.51] |

Для `forward_stability = unknown` обязательно показать отдельный
нейтральный сценарий `m = 1.00`. Для `cycling_risk` не придумывать
дополнительный multiplier: отметить риск качественно либо оценить его
заново по Source B. Target near-60 не считать доказательством cycling.

Показатель `57% vs 34%` выводить рядом только как
`early_failure_composition_90d_prior`; не умножать на него ни одну
вероятность отказа.

### C.5 Piecewise RUL после смены режима

Uplift начинается в текущем возрасте `a = nno_days`. Новый multiplier
должен влиять только на будущий интервал, не переписывая прошлую
эксплуатацию.

Для основной Weibull/PH-сценарной модели:

```python
def S_after(u, a, m):
    delta_H = H0(a + u) - H0(a)
    return exp(-m * delta_H)

P_fail_after(horizon, a, m) = 1 - S_after(horizon, a, m)
RUL_median_after: min u such that S_after(u, a, m) <= 0.50
RUL_B10_after:    min u such that S_after(u, a, m) <= 0.90
```

Для M2 sensitivity сначала вычислить posterior веса на момент uplift:

```python
S_e_a = exp(-(a / 4.6) ** 2.41)
S_n_a = exp(-(a / 202) ** 0.91)
w_e_a = 0.054 * S_e_a / S_mix(a)
w_n_a = 0.946 * S_n_a / S_mix(a)

def S_mix_after(u, a, m):
    dH_e = ((a + u) / 4.6) ** 2.41 - (a / 4.6) ** 2.41
    dH_n = ((a + u) / 202) ** 0.91 - (a / 202) ** 0.91
    return w_e_a * exp(-m * dH_e) + w_n_a * exp(-m * dH_n)
```

Если локальная модель показывает разные эффекты частоты для компонент,
использовать `m_e` и `m_n`; без такой оценки применять общий `m`.
Запрещено просто заменять `eta` во всём `S(a+u)/S(a)`: это
ретроспективно применяет новый режим к уже прожитому возрасту.

### C.6 Запрещённые упрощения

- Запрещено: `life_after = life_before × freq_cur / freq_after`
- Запрещено: использовать TRF как fixed budget
- Запрещено: считать uplift автоматически вредным
- Запрещено: выводить stable/cycling только из целевой частоты
- Запрещено: применять `1.68` как множитель `P(fail <= 90d)`
- Запрещено: смешивать endpoints и параметры M0/M2 разных выборок

### C.7 Диапазоны неопределённости

Для T3 near-60 использовать HR напрямую:

| Вариант | `m` |
|---|---|
| Optimistic | 0.71 |
| Base | 1.34 |
| Conservative | 2.51 |

Не масштабировать эти границы относительно других сценариев. Для T2
использовать `m=0.96` как exploratory base и `m=1.00` как нейтральный
sensitivity, пока не получен собственный CI. Дополнительно показать
sensitivity with / without motor_load_proxy и M0 / M2.

---

## РАЗДЕЛ D. RUL ПОСЛЕ UPLIFT

### D.1 Required метрики (два состояния: current / after)

- `rul_median_cur` / `rul_median_after`
- `rul_b10_cur` / `rul_b10_after`
- `p_fail_30d_cur` / `p_fail_30d_after`
- `p_fail_90d_cur` / `p_fail_90d_after`
- `p_fail_180d_cur` / `p_fail_180d_after`
- `projected_run_length_p50_cur` / `projected_run_length_p50_after`
- `rul_median_cur_m2_sensitivity` / `rul_median_after_m2_sensitivity`
- `early_failure_composition_90d_prior` — descriptor, не probability

### D.2 Дельты

- `delta_rul_median` = rul_median_after − rul_median_cur
- `delta_rul_b10`
- `delta_p_fail_30d`, `delta_p_fail_90d`, `delta_p_fail_180d`

### D.3 value_score (ранжирование скважин)

```
value_score = dq_oil_ind × rul_median_after / 1000
```

Proxy NPV uplift-а. Используется для ранжирования в portfolio
summary: топ-5 выигрышных и топ-5 с максимальными потерями ресурса.

---

## РАЗДЕЛ E. РЕКАЛИБРОВКА НА ЛОКАЛЬНЫХ ДАННЫХ

### E.1 Попытаться привязать скважины к БД

Проверить наличие доступа к:
- `mart__vt_freq55`
- `raw__v03_runs` / `proc__daily_merged`
- `ttf_true_best_days`

### E.2 При наличии данных (Source B)

- Привязать wells из Excel по well_id к историческим прогонам
- Для matched wells: использовать локальные survival данные,
  chemistry / H2S proxies, historical frequency behavior
- Выбрать одну duration/event/censoring схему и использовать её
  одинаково для KM, M0/AFT, M2 и проверки эффекта частоты
- Откалибровать baseline life и RUL локально с проверкой calibration
- Рассчитать telemetry stability metrics из B.5; отдельно проверить,
  действительно ли near-60 скважины имеют больше остановок и
  zero-интервалов, чем matched Vt control
- Не называть near-60 cycling, если это не подтверждено телеметрией
- Пометить: `source = A+B`

### E.3 При отсутствии данных

- Зафиксировать явно: `source = A+C (prior-driven)`
- Confidence автоматически снижается на один уровень

### E.4 Маркировка источника

Для каждой итоговой оценки:
- `source = A` / `A+B` / `A+C` / `B` / `C`
- `is_direct_measurement = True / False`

---

## РАЗДЕЛ F. СПЕЦИАЛЬНАЯ ЛОГИКА Vt И Bt

### F.1 Vt-скважины

- Prior из предыдущего анализа применять с явной uncertainty
- Near-60 penalty трактовать как диапазон [× 0.71, × 2.51] по оси HR,
  не как точечную оценку
- Majority-days >55 Hz не считать автоматически вредным или стабильным
- H2S / химия / early-life risk имеют больший вес, чем одна частота

### F.2 Bt-скважины (Bt_009р, Bt_1503)

- Prior из Vt не является fully transferable — другое поле
- Анализировать отдельно с явной пометкой
- `confidence = Low` автоматически
- `recommendation = defer pending more data`
- В таблице: `field_group = Bt`, `source_note = Vt prior applied
  outside calibration domain`

---

## РАЗДЕЛ G. ИТОГОВЫЕ ВЫХОДЫ

### G.1 Компактный скрининг (все 54 скважины, одна строка)

Колонки: well | field_group | freq_cur | freq_after | d_freq |
target_freq_band | historical_stability | forward_stability |
engineering_constraint | motor_load_cur | motor_load_after_est |
gas_after_sep_pct | active_flags | rag

### G.2 Детальная таблица

| Колонка | Описание |
|---|---|
| `well` | Название |
| `field_group` | Vt / Bt |
| `freq_cur` / `freq_after` / `d_freq` | Частоты |
| `motor_load_cur` / `motor_load_after_est` | Загрузка ПЭД |
| `gas_after_sep_pct` | % газа после сепарации |
| `q_liq_cur` / `q_oil_cur` | Текущие дебиты |
| `q_liq_affinity` / `q_oil_affinity` | Аффинный baseline (Layer 1) |
| `q_liq_expected_file` / `q_oil_expected_file` | Из файла |
| `q_liq_after_ind` / `q_oil_after_ind` | Независимая оценка (Layer 2) |
| `dq_oil_file` / `dq_oil_ind` | Прирост нефти |
| `file_vs_ind_deviation_pct` | Расхождение file vs ind, % |
| `nno_days` | Текущий возраст прогона |
| `target_freq_band` | T0 / T1 / T2 / T3 |
| `historical_stability` | stable / cycling / uncertain / no telemetry |
| `forward_stability` | expected_stable / cycling_risk / unknown |
| `engineering_constraint` | constrained / caution / screen_ok |
| `rul_median_cur` / `rul_median_after` | RUL медиана, дни |
| `rul_b10_cur` / `rul_b10_after` | RUL B10, дни |
| `rul_median_cur_m2_sensitivity` / `rul_median_after_m2_sensitivity` | Проверка чувствительности M2 |
| `projected_run_length_p50_cur` / `projected_run_length_p50_after` | nno_days + соответствующий RUL P50 |
| `p_fail_30d_cur` / `p_fail_30d_after` | P(fail ≤ 30d) |
| `p_fail_90d_cur` / `p_fail_90d_after` | P(fail ≤ 90d) |
| `p_fail_180d_cur` / `p_fail_180d_after` | P(fail ≤ 180d) |
| `delta_rul_median` / `delta_rul_b10` | Изменение RUL |
| `delta_p_fail_90d` | Изменение риска 90d |
| `early_failure_composition_90d_prior` | 57% vs 34% среди произошедших отказов; не P(fail) |
| `motor_proxy_sensitivity` | Разница оценок with/without proxy |
| `value_score` | dq_oil_ind × rul_median_after / 1000 |
| `source_used` | A / A+B / A+C |
| `confidence_level` | High / Medium / Low |
| `flags` | Список активных флагов |
| `rag` | Green / Yellow / Red |
| `recommendation` | approve / pilot / defer / no |
| `comment` | Ключевое инженерное обоснование |

### G.3 RAG-критерии (применять последовательно, побеждает строже)

**Red — не повышать:**
- `motor_load_after_est > 85%`, ИЛИ
- `gas_after_sep_pct > 55%`, ИЛИ
- `flag_no_uplift` (нет пространства), ИЛИ
- `nno_days < 30` И `target_freq_band = T3`, ИЛИ
- `flag_bt_well` без локальных данных

**Yellow — только пилот с мониторингом 30/60/90 дней:**
- `motor_load_after_est ∈ (75%, 85%]`, ИЛИ
- `gas_after_sep_pct ∈ (30%, 55%]`, ИЛИ
- `d_freq ∈ [2, 4)`, ИЛИ
- `nno_days < 90`, ИЛИ
- `p_fail_90d_after > 0.55`

**Green — допустимо повышать:**
- motor_load_after ≤ 75%, gas_after ≤ 30%,
  d_freq ≥ 4, nno_days ≥ 90

**Confidence:**

| Level | Условие |
|---|---|
| High | nno_days ≥ 90, Vt, ≤1 yellow-флага, Source B доступен |
| Medium | nno_days 30–90, или Bt, или 1–2 yellow-флага, Source C |
| Low | nno_days < 30, или motor_after > 90%, или ≥3 флагов, или Bt без данных |

**Рекомендации:**

| Вариант | Условие |
|---|---|
| `approve uplift` | Green + High/Medium confidence |
| `approve only as monitored pilot` | Yellow, любая confidence |
| `defer pending more data` | Low confidence или Bt |
| `do not uplift` | Red |

### G.4 Portfolio summary

1. Распределение RAG: N_green / N_yellow / N_red
2. Распределение диапазонов: N_T0 / N_T1 / N_T2 / N_T3
3. Распределение `historical_stability` и доля `no_telemetry`
4. Суммарный и средний `dq_oil_ind` по Green + Yellow скважинам
5. Средний `delta_rul_median` по портфелю и `target_freq_band`
6. Tail risk: скважины с `p_fail_90d_after > 0.60` — перечислить явно
7. Топ-5 по `value_score` — лучший trade-off добыча/ресурс
8. Топ-5 по `delta_rul_median` в минус — наибольшая потеря ресурса
9. Конфликтные скважины: `dq_oil_ind > медианного` И
   `delta_rul_median < −30 дн.`
10. Sensitivity к motor_load_proxy: изменение N_green / N_red при
   `use_motor_load_proxy = False`
11. **Selection bias note:** этот портфель предварительно отобран
    как кандидаты на УВЧ. Fleet Vt prior включает тяжёлые среды
    и ранние отказы — вероятно, переоценивает риск для данного
    subset. Все RUL-оценки считать консервативными upper bound.

---

## РАЗДЕЛ H. ТРЕБОВАНИЯ К ИНТЕРПРЕТАЦИИ

1. Писать по-русски.
2. Явно разделять: **[ФАЙЛ]** — факт из Excel, **[БД]** — из
   локальной БД, **[PRIOR]** — из прошлого анализа, **[МОДЕЛЬ]** —
   вычисление по формуле, **[ЭКСПЕРТ]** — инженерное суждение.
3. Не скрывать неопределённость. Если диапазон широк — давать
   три сценария: optimistic / base / conservative.
4. Каждую используемую формулу выписывать явно.
5. Если `motor_load_after_est` меняет вывод — показать явно.
6. Если M2 sensitivity и основная KM/M0/AFT-модель дают RUL с
   расхождением > 20% — приводить оба числа, понижать confidence и
   не заменять основной результат значением M2.
7. Если вывод чувствителен к `use_motor_load_proxy` — показать
   явно в строке скважины и в portfolio summary.
8. Если оценка неустойчива из-за отсутствия Source B — написать
   прямо: `source = A+C, confidence = Medium/Low`.
9. Если расчёт из файла и независимая оценка расходятся — объяснить
   наиболее вероятную инженерную причину.
10. Не использовать TRF-budget reasoning.
11. Не считать uplift автоматически вредным.
12. Target frequency, фактическую экспозицию и стабильность режима
    показывать раздельно. Near-60 не называть cycling без проверки
    временных рядов.

---

## КРАТКАЯ ЦЕЛЬ

Независимая инженерно-статистическая оценка trade-off между
приростом добычи нефти и потерей ресурса ЭЦН для 54 скважин
при фиксированном сценарии `freq_after = Fмакс.возм`.

**Обязательный минимум выхода:**
- текущий RUL из основной conditional KM/M0/AFT-модели
- RUL после uplift по piecewise-формуле: новый режим действует только
  после текущего `nno_days`
- M2 только как sensitivity RUL
- delta ННО / МРП / P(fail 30/90/180d)
- sensitivity: with / without motor_load_proxy
- telemetry stability или явный статус `no_telemetry` / `unknown`
- RAG + recommendation по каждой из 54 скважин
- portfolio summary с tail risk и value_score ranking
