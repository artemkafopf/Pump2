# ННО calculator — self-contained implementation prompts (Vt)

Copy-paste specs for implementing the УЭЦН time-to-failure (ННО) calculation in any language
(VBA / Python / …). The model is **not re-fit** — these carry the ready-made parameters and
self-verify against check values. Derived from the Vt operating-covariate survival analysis
(`vt_ttf_covariates.py`; see `docs/notes/vt_model_slides.md`, slide 18).

Two variants:
- **(A)** — two working patterns (`Ql-only` + `all-three`), the ~90% deployment case. Simplest.
- **(B)** — the full 7-pattern availability family (each pattern has its own β and η_ref).

Reference point for both: **дебит 250 / частота 50 Гц / Кпод 0.70** (brt). Reporting standard:
**RMST(0, 730) headline, median for reference**.

---

## Variant (A) — two patterns

```text
ЦЕЛЬ. Реализовать функции расчёта срока службы УЭЦН по ГОТОВЫМ параметрам (модель не
переобучать). Четыре показателя: МЕДИАНА, СРЕДНИЙ РЕСУРС (MRL при 0), РЕСУРС ЗА ГОРИЗОНТ
(RMST за τ), ОСТАТОЧНЫЙ РЕСУРС (MRL в возрасте t). Язык — любой (VBA/Python/…).

ПАРАМЕТРЫ (ВСТРОЕНЫ — перенести на лист как есть). Два паттерна × класс H₂S:
 паттерн       класс     beta  eta_ref  дебит_оп  g_дебит  g_част   g_Кпод   brt  slb    oth
 только-дебит  некислые  1.15  571      250       -0.305   —        —        1.0  0.79   0.54
 только-дебит  кислые    1.37  149      —(g=0)     0        —        —        1.0  1.05   0.645
 все-три       некислые  1.28  570      250       -0.130   -0.0142  -0.729   1.0  0.728  0.635
   («все-три» опорные: частота_оп=50, Кпод_оп=0.70. Кислые «все-три»=«только-дебит»: режим не влияет.)
 Границы применимости (некислые; ЗАЖИМАТЬ, не экстраполировать): дебит 47..823, частота 31..56,
   Кпод 0.2..1.2.  Горизонт tau по умолчанию 730 сут.

ВЫБОР ПАТТЕРНА: некислые и известны дебит+частота+Кпод → «все-три»; иначе (или кислые) →
 «только-дебит» (fallback). Недостающие слагаемые Δ ОПУСТИТЬ.

ШАГ 1 — множитель φ через exp-log по ОТКЛОНЕНИЯМ Δ от опорных (только известные ковариаты):
  q = min(max(дебит, 47), 823)                     // зажать дебит к границам
  s =  g_дебит  · (ln(q) − ln(дебит_оп))            // дебит: отклонение по ЛОГАРИФМУ
     + g_частота · (частота − 50)                    // ОПУСТИТЬ, если частоты нет
     + g_Кпод   · (Кпод   − 0.70)                    // ОПУСТИТЬ, если Кпода нет
  φ = множитель_подрядчика · exp(s);   eta = eta_ref · φ

ШАГ 2 — показатели из (eta, beta). Кривая доживаемости S(t)=exp(−(t/eta)^beta):
  МЕДИАНА              = eta · ln(2)^(1/beta)
  СРЕДНИЙ РЕСУРС MRL(0) = eta · Γ(1 + 1/beta)
  РЕСУРС ЗА ГОРИЗОНТ   = eta · Γ(1 + 1/beta) · P(1/beta, (tau/eta)^beta)     // = RMST(0,tau)
  ОСТАТОЧНЫЙ РЕСУРС(t) = eta · Γ(1 + 1/beta) · Q(1/beta, (t/eta)^beta) / exp(−(t/eta)^beta)
      где Γ — гамма-функция; P(s,x) — регуляризованная НИЖНЯЯ неполная гамма; Q(s,x)=1−P(s,x).

СПЕЦ-ФУНКЦИИ (единственная нетривиальная часть):
  Γ(z):   Python math.gamma(z)                | VBA Exp(WorksheetFunction.GammaLn(z))
  P(s,x): Python scipy.special.gammainc(s,x)  | VBA WorksheetFunction.Gamma_Dist(x,s,1,True)
          без библиотек: γ(s,x)=x^s·e^−x·Σ_{k≥0} x^k/(s·(s+1)…(s+k)); P=γ(s,x)/Γ(s) (x≲s+1; иначе через Q)
  Q(s,x) = 1 − P(s,x)

ПРОВЕРКА (все три строки должны сойтись):
  A) некислые/slb «все-три», дебит400 частота52 Кпод0.90:
     s = −0.130·ln(400/250) − 0.0142·(52−50) − 0.729·(0.90−0.70) = −0.235;  exp(s)=0.790;
     φ = 0.728·0.790 = 0.576;  eta = 570·0.576 = 328  → МЕДИАНА 247, СРЕДНИЙ 304, RMST730 292.
  B) некислые/slb «только-дебит», дебит400: φ = 0.79·exp(−0.305·ln(400/250)) = 0.686;
     eta = 571·0.686 = 392  → 285 / 373 / 335.
  C) кислые/slb (g=0, режим не влияет): φ = 1.05;  eta = 149·1.05 = 156  → МЕДИАНА 120, СРЕДНИЙ 143, RMST730 143.

ЗАМЕЧАНИЯ. Медиана и средний ресурс масштабируются линейно по φ; ресурс за горизонт —
сублинейно (tau фиксирован). Докладывать «ресурс за горизонт» как основной, медиану — справочно.
```

---

## Variant (B) — full 7-pattern availability family

```text
ЦЕЛЬ. Реализовать функции расчёта срока службы УЭЦН по ГОТОВЫМ параметрам (модель не
переобучать). Четыре показателя: МЕДИАНА, СРЕДНИЙ РЕСУРС (MRL при 0), РЕСУРС ЗА ГОРИЗОНТ
(RMST за τ), ОСТАТОЧНЫЙ РЕСУРС (MRL в возрасте t). Язык — любой (VBA/Python/…).

ПАРАМЕТРЫ (ВСТРОЕНЫ). НЕКИСЛЫЕ — 7 паттернов доступности; опорные: дебит 250, частота 50,
Кпод 0.70; brt=1.0. У КАЖДОГО паттерна СВОИ beta и eta_ref (отдельная подгонка):
 паттерн             beta  eta_ref  g_дебит   g_част    g_Кпод    slb    oth
 дебит+частота+Кпод  1.28  570      -0.130    -0.0142   -0.729    0.728  0.635
 дебит+Кпод          1.18  549      -0.191     —        -0.599    0.756  0.633
 дебит+частота       1.21  592      -0.190    -0.0164    —        0.777  0.514
 частота+Кпод        1.26  590       —        -0.0176   -0.808    0.704  0.643
 дебит               1.14  574      -0.240     —         —        0.791  0.551
 частота             1.18  628       —        -0.0221    —        0.744  0.504
 Кпод                1.15  582       —         —        -0.722    0.718  0.627
 КИСЛЫЕ (любой паттерн): beta=1.37, eta_ref=149, режимные g=0 → срок = 149·подрядчик;
   brt=1.0, slb=1.05, oth=0.645.
 Границы применимости (некислые; ЗАЖИМАТЬ, не экстраполировать): дебит 47..823, частота
   31..56, Кпод 0.2..1.2.  Горизонт tau по умолчанию 730 сут.
 (Примечание: строка «дебит» — простой лог; ВАЛИДИРОВАННЫЙ дефолт при одном дебите — с
   отсечками: beta=1.15, eta_ref=571, g_дебит=-0.305, slb=0.79, oth=0.54 — предпочтителен.)

ВЫБОР ПАТТЕРНА: взять строку, ТОЧНО совпадающую с набором известных ковариат (не подставлять
 недостающую и не брать более полный паттерн — двойной счёт). Кислые → всегда база (g=0).

ШАГ 1 — множитель φ через exp-log по отклонениям Δ от опорных (слагаемые только паттерна):
  q = min(max(дебит, 47), 823)
  s = g_дебит·(ln q − ln 250) + g_частота·(частота − 50) + g_Кпод·(Кпод − 0.70)  // g отсутствующих = 0
  φ = множитель_подрядчика · exp(s);   eta = eta_ref(паттерн) · φ

ШАГ 2 — показатели из (eta, beta). Кривая доживаемости S(t)=exp(−(t/eta)^beta):
  МЕДИАНА              = eta · ln(2)^(1/beta)
  СРЕДНИЙ РЕСУРС MRL(0) = eta · Γ(1 + 1/beta)
  РЕСУРС ЗА ГОРИЗОНТ   = eta · Γ(1 + 1/beta) · P(1/beta, (tau/eta)^beta)     // = RMST(0,tau)
  ОСТАТОЧНЫЙ РЕСУРС(t) = eta · Γ(1 + 1/beta) · Q(1/beta, (t/eta)^beta) / exp(−(t/eta)^beta)
      Γ — гамма; P(s,x) — регуляризованная НИЖНЯЯ неполная гамма; Q(s,x)=1−P(s,x).

СПЕЦ-ФУНКЦИИ:
  Γ(z):   Python math.gamma(z)                | VBA Exp(WorksheetFunction.GammaLn(z))
  P(s,x): Python scipy.special.gammainc(s,x)  | VBA WorksheetFunction.Gamma_Dist(x,s,1,True)
          без библиотек: γ(s,x)=x^s·e^−x·Σ_{k≥0} x^k/(s·(s+1)…(s+k)); P=γ(s,x)/Γ(s) (x≲s+1; иначе через Q)
  Q(s,x)=1−P(s,x)

ПРОВЕРКА (должны сойтись):
  A) некислые/slb «дебит+частота+Кпод», 400/52/0.90:
     s=−0.130·ln(400/250)−0.0142·(52−50)−0.729·(0.90−0.70)=−0.235; exp=0.790; φ=0.728·0.790=0.576;
     eta=570·0.576=328 → МЕДИАНА 247, СРЕДНИЙ 304, RMST730 292.
  B) некислые/slb «дебит» (только дебит), 400: φ=0.791·exp(−0.240·ln(400/250))=0.707;
     eta=574·0.707=406 → МЕДИАНА 294, СРЕДНИЙ 387, RMST730 343.
  C) кислые/slb (g=0): φ=1.05; eta=149·1.05=156 → МЕДИАНА 120, СРЕДНИЙ 143, RMST730 143.

ЗАМЕЧАНИЯ. Медиана и средний ресурс масштабируются линейно по φ; ресурс за горизонт —
сублинейно (tau фиксирован). Докладывать «ресурс за горизонт» как основной, медиану — справочно.
```

---

## Notes on provenance / caveats

- Parameters: base Ql-only from `revised_operating_model_v2.csv` (capped log-Ql, validated,
  replicated on Ya); multi-covariate patterns re-fit from `availability_life_multipliers.csv`
  logic (plain log-Ql, per-pattern β/η_ref computed at the 250/50/0.70 reference).
- **(A) is preferred for deployment** (Ql-only is the ТМ-06-available covariate; freq/Kpod are
  the same throughput signal and not on the plan). Use **(B)** only if freq/Kpod are genuinely
  known — the 2-covariate patterns are fit on smaller n and are noisier.
- The "дебит" family row in (B) uses plain log-Ql (β=1.14) and differs slightly from the
  validated capped default (β=1.15, η_ref=571, γ=−0.305), which is preferred when only Ql is known.
- All effects observational (wells not randomised); good for fleet/group planning, weak for
  single-pump timing (concordance ≈0.58). Sour: operating regime null → contractor only.
