"""Covariate binning of the Свод fact panel — the estimator layer under the explorer app.

Started as the 2 Hz frequency panel (Ya failures, ННО vs running frequency); the axis is now
a parameter.  :data:`PROPERTIES` registers each one — frequency, Ql, Qном, Kпод, ГЖ,
обводнённость, Pзаб, Pзаб/Pнас — with its own plausible window, anchor, quantum and
**scale**, and nothing below the registry knows which axis it is looking at.  Adding a
covariate is one entry there plus, if it needs a join, one in :data:`ATTACHERS`.

**1. The bins are a choice, not a measurement.**  A grid of edges is an analyst's hypothesis
about where the axis has structure, and the honest way to defend a shape is to show it
survives being re-cut.  So edges here are *data*: a sorted list that can be moved, merged,
split and re-derived (:func:`uniform_edges`, :func:`equal_count_edges`, :func:`move_edge`,
:func:`merge_bin`, :func:`split_bin`, :func:`merge_thin_bins`).  Every operation returns a new
edge list and leaves the input alone, so an undo stack is just a list of lists.

Rate-like axes are built in **log10** — see :class:`Property` — so a "quantum" is a ratio
there and the grid widens with the value, which is both how the workflow's own rate layers
are parameterised and the only way a 5 → 2000 m³/d axis gets usable resolution at both ends.

**2. Mean and median disagree, and the disagreement is the finding.**  On the 2 Hz panel the
mean peaks at 46–48 Hz and the median at 48–50; at 60–62 Hz the mean reads 409 d against a
median of 89 d.  Three distinct causes are separable here and the module reports all three:

* **Skew.**  Life is right-skewed, so mean > median always.  The clean summary for a
  multiplicative variable is the **geometric mean** ``exp(mean(log t))`` — for a lognormal it
  estimates the same centre as the median but uses every observation, so it has the median's
  resistance to the tail and much of the mean's efficiency.  ``sigma_log`` (the sd of log t)
  is the spread that generates the gap: ``mean / geomean ≈ exp(σ²/2)``.
* **Noise.**  A 12-run bin's median has a very wide interval.  Every statistic here ships a
  **bootstrap CI** (:func:`bin_stats`), and :func:`permutation_flatness` asks the prior
  question — whether the binned curve departs from flat at all — separately per statistic, so
  "mean and median disagree" can be tested rather than eyeballed.
* **Selection.**  ННО exists only for pumps that have been pulled, so the fact panel is
  failures-only *by construction*, and a mean/median split at high frequency is the exact
  signature the repo has already caught on this axis (the ~51 Hz inverted-U halves and dies
  once censored runs enter).  :func:`km_bin_stats` is the answer: the same bins, the same
  Свод «Частота» column, but the full population including running pumps, summarised by
  **KM median and RMST(0, 730)** — the estimands the project's standing rule asks for.

On top of the bins, :func:`fit_trend` puts a weighted curve through **only** the bins that
carry enough failures to be worth fitting — thin bins are excluded outright, not
down-weighted — and reports the support it was fitted on, so the curve is never drawn
somewhere it knows nothing about.

**3. A marginal panel confounds; the fitted layers can be taken back off.**
:func:`remove_layers` rescales the time axis by a fitted model's θ (``t · Πθ^(1/β)``, an AFT
offset, not a residual), so a frequency panel can be read with the rate layer already
removed.  :data:`LAYER_MODELS` wires Ya v2 (θ_Ql), Ya v2.1 (θ_Qном) and Vt v4; the result is
still a life in days on the same axis, so adjusted and raw curves compare directly.

The censored panel deliberately reuses the Свод covariate columns rather than the telemetry
run-means: the point of that comparison is to change *one* thing (who is in the population),
and swapping the covariate definition at the same time would confound it.  ГЖ and
обводнённость have no Свод column at all and come from the daily t0 window
(:func:`warehouse_covariates`), joined exactly on ``(code, install)``.

⚠ Every panel here is **marginal**.  These covariates are not independent — Kпод *is*
Ql/Qном, ГЖ *is* Qг/Ql, Pзаб/Pнас is Pзаб over a near-constant — so a shape on one axis may
be another axis wearing a ratio.  That is a question for a fit with the others held fixed,
not for a binned panel; this module shows what each axis looks like on its own and nothing
more.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import results_dir
from analysis.workflows.field_sim.km import kaplan_meier, rmst as km_rmst
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk, esp_population

SLUG = "production_risk_freq_bins"

#: «Частота» in the Свод sheet runs from 2.0 to 236 Hz, which no ESP drive does; those rows
#: are transcription noise.  Same window every other module on this axis uses.
FREQ_BOUNDS = (30.0, 70.0)
#: Width of the plain fixed-width grid ("reset to uniform").
DEFAULT_WIDTH = 2.0
#: Target failures per bin for the adaptive builder, and the threshold below which a bin is
#: reported but flagged.  40 is the operator's rule: a bin that thin cannot carry a median.
TARGET_BIN_N = 40
#: Backwards-compatible alias.
MIN_BIN_N = TARGET_BIN_N
#: Same threshold on a **discrete** axis, where it no longer builds anything — a bin is one
#: catalog value and cannot be grown.  It only marks a point thin and gates the trend, and
#: almost no single Qном size carries 40 failures, so holding 40 there would blank the trend
#: on every panel and draw every marker hollow.
CATALOG_BIN_N = 10
#: The adaptive build's seed bin.  50–51 Hz is the grid frequency and the reference every
#: frequency layer in the workflow is pinned at, so the binning grows outward from the
#: operating point rather than from an arbitrary axis end.
ANCHOR = (50.0, 51.0)
#: The builder's quantum.  Edges land on whole Hz, so a bin is always 1, 2, 3 … Hz wide.
STEP_HZ = 1.0
#: Widest bin the **automatic** build will produce.  The target and this cap are in tension —
#: the sparse tails cannot reach 40 failures inside 5 Hz — and the cap wins: a bin that hits
#: it stops growing and is reported short rather than swallowing 16 Hz of axis.  A 16 Hz bin
#: averages runs that have nothing physical in common, which is a worse error than a thin one.
#: Hand edits are not bound by this: :func:`merge_bin` and :func:`move_edge` make any width.
MAX_BIN_WIDTH_HZ = 5.0
#: Two edges closer than this describe no runs and break the digitize.
MIN_BIN_WIDTH = 0.5
#: RMST horizon, held at the project-wide 730 d so the numbers are comparable to every fit.
RMST_HORIZON = 730.0

#: Statistic key → (column, label).  ``geomean`` is the recommended default centre for a
#: right-skewed life; ``mean`` is kept because it is what a plan is built from.
STATS = {
    "mean": "среднее",
    "median": "медиана",
    "geomean": "геом. среднее",
    "trimmed": "усечённое среднее",
}
KM_STATS = {"km_median": "медиана KM", "rmst": f"RMST(0,{RMST_HORIZON:.0f})"}


# ---------------------------------------------------------------------------
# The covariates this machinery can be pointed at
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Property:
    """One x-axis: where the column comes from, and how a grid is built on it.

    ``scale`` is the substantive field.  Frequency, Kpod and water cut are **additive**
    quantities — a 2 Hz step means the same thing at 40 Hz and at 60 — so their grids are
    linear and their edges are round numbers.  Rates and ratios (Ql, Qном, ГЖ) are
    **multiplicative**: they span two or three decades, the models fit them in logs, and a
    linear grid would spend most of its bins on a tail holding 2 % of the runs while lumping
    the crowded low end into one.  Those are built on ``log10`` and their ``step`` /
    ``max_width`` are therefore in **dex** — 0.05 dex is ×1.12, 0.30 dex is ×2.

    ``anchor`` is the seed bin the build grows outward from.  Where the workflow already has
    a reference for the covariate, that is what it is pinned to (θ_Ql at 250, θ_Qном at 200,
    θ_Kпод at 0.8, frequency at grid 50) so the grid agrees with the fitted layers rather
    than with this panel's own median, which would move every time the data does.
    """
    key: str
    column: str
    label: str
    unit: str
    source: str                 # "svod" — from the sheet; "warehouse" — needs the t0 join
    bounds: tuple               # plausible window; outside is junk, not data
    anchor: tuple               # seed bin (upper edge advisory — the build enforces one step)
    step: float                 # quantum: data units if linear, DEX if log
    max_width: float            # cap on a bin: data units if linear, DEX if log
    min_width: float            # closest two edges may sit: DATA units if linear, RATIO if log
    ui_step: float              # granularity of the edge slider, data units
    scale: str = "linear"
    fmt: str = "%.0f"
    #: Significant figures a log-axis edge is rounded to for readability.  **3, not 2.**
    #: Qном is a catalog of discrete sizes, so an edge that moves 2 % can step over a spike
    #: and take 30 runs with it — the printed counts stay truthful, but the build would miss
    #: the target it just worked to hit.  3 figures caps the move at 0.5 %.
    sig: int = 3
    #: The axis is a **catalog of discrete values**, not a continuum.  Its default grid is
    #: :func:`catalog_edges` — one bin per value actually present — instead of the n-target
    #: build, and the marker is drawn on the value rather than in the middle of a bracket.
    #: Widening a bin here does not average "nearby operating points"; it averages **different
    #: pumps**, and the shape a catalog axis has is the step from one size to the next.
    discrete: bool = False
    note: str = ""

    def min_gap(self, at: float) -> float:
        """Smallest gap allowed between two edges **near the value** ``at``.

        On a log axis a fixed absolute minimum is meaningless: 5 → 6.3 is a 26 % bin, a
        perfectly good one, yet only 1.3 units wide, while 1000 → 1001 is nothing at all
        and is 1 unit wide.  So ``min_width`` is read as a ratio there and converted to a
        local width here.
        """
        return abs(float(at)) * (self.min_width - 1.0) if self.scale == "log" \
            else self.min_width

    @property
    def list_min_width(self) -> float:
        """Gap enforced when a whole edge list is normalised at once.

        Zero on a log axis — one number cannot be right at both ends of three decades, and
        the real minimum is applied per edge by :meth:`min_gap` where an edit happens.
        Passing the data-unit ``min_width`` here would silently delete the low-end edges.
        """
        return 0.0 if self.scale == "log" else self.min_width

    @property
    def width_label(self) -> str:
        return "Макс. отношение границ бина, ×" if self.scale == "log" else \
            f"Макс. ширина бина, {self.unit}"

    def width_ui(self, width: float) -> float:
        """Cap as the operator types it: a ratio for log axes, a width for linear ones."""
        return float(10.0 ** width) if self.scale == "log" else float(width)

    def width_internal(self, shown: float) -> float:
        return float(np.log10(max(shown, 1.0000001))) if self.scale == "log" else float(shown)

    def axis_title(self) -> str:
        return f"{self.label}, {self.unit}"


#: Every covariate the app can bin on.  Adding one is a single entry here — nothing
#: downstream knows which axis it is looking at.
PROPERTIES: dict[str, Property] = {
    "freq": Property(
        "freq", "freq", "Рабочая частота", "Гц", "svod",
        bounds=FREQ_BOUNDS, anchor=(50.0, 51.0), step=1.0, max_width=5.0,
        min_width=0.5, ui_step=0.25, scale="linear", fmt="%.0f",
        note="Якорь 50–51 Гц — частота сети и точка отсчёта всех частотных слоёв. "
             "Вне 30–70 Гц «Частота» в Своде — опечатки (встречается 2 и 236)."),
    "ql": Property(
        "ql", "ql", "Дебит жидкости", "м³/сут", "svod",
        bounds=(5.0, 2000.0), anchor=(250.0, 280.0), step=0.05, max_width=0.30,
        min_width=1.02, ui_step=5.0, scale="log", fmt="%.0f",
        note="Логарифмическая сетка: два с половиной порядка, и слой θ_Ql везде в "
             "работе подгоняется по log Ql. Якорь 250 — пин θ_Ql."),
    "qnom": Property(
        "qnom", "qnom", "Номинальная производительность", "м³/сут", "svod",
        bounds=(20.0, 2000.0), anchor=(200.0, 224.0), step=0.05, max_width=0.30,
        min_width=1.02, ui_step=5.0, scale="log", fmt="%.0f", discrete=True,
        note="Паспортная подача насоса: выбирается при монтаже, поэтому не может быть "
             "загрязнена тем, чем кончился пуск. Якорь 200 — пин θ_Qном в Ya v2.1. "
             "Ось ДИСКРЕТНАЯ: сетка по умолчанию — один бин на один типоразмер, без "
             "правила «доращивать до n». Соседние размеры — разные насосы, и ступенька "
             "между ними и есть форма этой оси; правило n сливало бы её. Значения ближе "
             "2 % друг к другу (243/244, 460/461/464) сетка разделить не может и "
             "объединяет — это одна модель, записанная по-разному."),
    "kpod": Property(
        "kpod", "kpod", "Коэффициент подачи", "доля", "svod",
        bounds=(0.05, 1.8), anchor=(0.80, 0.85), step=0.05, max_width=0.30,
        min_width=0.02, ui_step=0.01, scale="linear", fmt="%.2f",
        note="Kпод = Ql / Qном по Своду. Якорь 0.8 — пин θ_Kпод. Это ОТНОШЕНИЕ двух "
             "других осей списка, так что читать его надо вместе с ними."),
    "glf": Property(
        "glf", "glf", "Газожидкостный фактор", "м³/м³", "warehouse",
        bounds=(1.0, 3000.0), anchor=(50.0, 63.0), step=0.05, max_width=0.40,
        min_width=1.02, ui_step=1.0, scale="log", fmt="%.0f",
        note="ГЖ = Qг / Qж по окну t0 из склада — в Своде такой колонки нет, поэтому "
             "потребуется сборка ковариат (долгая, кэшируется). Своей точки отсчёта у "
             "ГЖ нет: якорь 50 — просто круглое число рядом с медианой Ya."),
    "wcut": Property(
        "wcut", "wcut", "Обводнённость", "%", "warehouse",
        bounds=(0.0, 100.0), anchor=(40.0, 45.0), step=5.0, max_width=20.0,
        min_width=1.0, ui_step=1.0, scale="linear", fmt="%.0f",
        note="Из того же окна t0, что и ГЖ, — приезжает тем же соединением, поэтому "
             "показывается даром."),
    "pzab": Property(
        "pzab", "pzab", "Забойное давление", "атм", "pressure",
        bounds=(10.0, 400.0), anchor=(100.0, 112.0), step=0.05, max_width=0.30,
        min_width=1.02, ui_step=1.0, scale="log", fmt="%.0f",
        note="Среднее Pзаб по первым 30 рабочим суткам пуска (телеметрия `rzab`). "
             "Квантили Ya ложатся почти геометрически (37/68/95/129/259), поэтому шкала "
             "логарифмическая. Своей точки отсчёта нет: якорь 100 — круглое число рядом "
             "с медианой. ⚠ Сырой канал содержит −10⁵…10⁶ атм; всё вне 1–500 атм "
             "отбрасывается как физически невозможное (≈10 % суточных строк)."),
    "pzab_over_pbub": Property(
        "pzab_over_pbub", "pzab_over_pbub", "Pзаб / Pнас", "доля", "pressure",
        bounds=(0.05, 3.0), anchor=(1.0, 1.12), step=0.05, max_width=0.30,
        min_width=1.02, ui_step=0.01, scale="log", fmt="%.2f",
        note="Якорь ровно на 1.0 — давлении насыщения: ниже единицы газ выделяется уже "
             "в пласте. ⚠ ДВА ограничения. Pнас на месторождении почти константа (на Ya "
             "разброс между скважинами CV 3.7 %, 230–259 атм), поэтому В ПРЕДЕЛАХ ОДНОГО "
             "месторождения это Pзаб, поделённое на число: ранговая корреляция с Pзаб "
             "0.997, и кривые будут одной формы. И 98.6 % пусков Ya и так ниже единицы, "
             "так что порог почти не даёт контраста."),
}

#: Columns that only exist after the warehouse t0 join.
WAREHOUSE_COLS = ("glf", "qg", "wcut")
#: Columns that only exist after the pressure join.
PRESSURE_COLS = ("pzab", "pbub", "pzab_over_pbub")

#: Daily ``rzab`` outside this window is not a measurement.  The raw channel runs from
#: −105 668 to 1 249 937 atm; 90 % of operating-day rows sit inside the guard.
RZAB_PLAUSIBLE = (1.0, 500.0)
#: Window the pressure covariate is averaged over — the same first-30-operating-days rule
#: ``t0_covariates`` uses, and for the same reason: a run mean over a failing pump's last
#: days is already degradation, so regressing on it reads the failure backwards.
PRESSURE_WINDOW_OP_DAYS = 30


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
#: Свод columns every panel carries, so any of them can become the x-axis.
SVOD_COVARIATES = ("Частота", "Дебит жидк.", "Ном. Произв. м₃/сут")

#: Pseudo-field: every field pooled.  A real field code can never collide with it.
FLEET = "Весь фонд"
#: A field needs this many failures before it earns its own panel.  Below it a binned curve
#: is a handful of runs per bin and the grid builder cannot reach its target anywhere.
MIN_FIELD_FAILURES = 100

#: Populations that are a field **plus** an H₂S class — the same stratum keys the Vt models
#: are fitted on.  Vt only, and deliberately: «Кислый/Некислый» is written for Vt wells, and
#: both loaders stamp every non-Vt run ``nonsour``, so a "split" anywhere else would just
#: rename the whole field.  The class is the **v3.2 well-level relabel** (sour is a property
#: of the well's fluid, rolled onto all its runs) on both panels.
SOUR_SPLIT = {"Vt_sour": ("Vt", "sour"), "Vt_nonsour": ("Vt", "nonsour")}
#: Population key → what to show a человек.  Fields keep their own code.
POPULATION_LABELS = {"Vt_sour": "Vt · кислый", "Vt_nonsour": "Vt · некислый"}


def resolve_population(key: str) -> tuple[str, str | None]:
    """``population → (field, h2s_class | None)``; a plain field code passes through."""
    return SOUR_SPLIT.get(str(key), (str(key), None))


def population_label(key: str) -> str:
    return POPULATION_LABELS.get(str(key), str(key))


def population_mask(panel: pd.DataFrame, key: str) -> pd.Series:
    """Rows of ``panel`` belonging to population ``key`` — field, fleet or Vt ± sour."""
    field, h2s = resolve_population(key)
    keep = (pd.Series(True, index=panel.index) if field == FLEET
            else panel["field"] == field)
    if h2s is not None:
        keep = keep & (panel["h2s_class"].astype(str) == h2s)
    return keep


@lru_cache(maxsize=4)
def field_failure_counts(variant: str = "failures", mc_cohort: bool = True) -> pd.Series:
    """Rows per field in the fact panel, largest first.

    Мирнинский is counted **after** its 2024+ cohort filter, which is the standing rule for
    that field — so its number here is the number that may actually be reported, not the
    number of rows on the sheet.
    """
    from analysis.workflows.production_risk.svod_nno_decomposition import load_panel

    panel = load_panel(None, variant=variant, mc_cohort=mc_cohort)
    return panel.groupby("field").size().sort_values(ascending=False)


def selectable_fields(variant: str = "failures", min_n: int = MIN_FIELD_FAILURES,
                      mc_cohort: bool = True) -> list[str]:
    """``[FLEET, …fields clearing min_n…]`` — what a UI should offer as the population.

    The threshold gates which fields get their **own** panel; it does not gate who is in the
    fleet.  A field too thin to carry a curve of its own still belongs to the fleet it is
    part of, and leaving it out would make the pooled panel a different population from the
    one the workflow actually models.
    """
    c = field_failure_counts(variant, mc_cohort)
    return [FLEET, *c[c >= int(min_n)].index.tolist()]


@lru_cache(maxsize=4)
def population_counts(variant: str = "failures", mc_cohort: bool = True) -> pd.Series:
    """Rows per population — every field, plus the Vt sour/nonsour halves.

    A superset of :func:`field_failure_counts`, so it must **not** be summed: the split
    entries are the same runs counted a second time.  The fleet total comes from the
    field-only series.
    """
    from analysis.workflows.production_risk.svod_nno_decomposition import load_panel

    panel = load_panel(None, variant=variant, mc_cohort=mc_cohort)
    counts = field_failure_counts(variant, mc_cohort)
    extra = {k: int(population_mask(panel, k).sum()) for k in SOUR_SPLIT}
    return pd.concat([counts, pd.Series(extra, dtype="int64")])


def selectable_populations(variant: str = "failures", min_n: int = MIN_FIELD_FAILURES,
                           mc_cohort: bool = True) -> list[str]:
    """:func:`selectable_fields` with each split population inserted after its parent field.

    A split earns its own panel on the same rule as a field — enough failures to fill bins —
    and the halves are offered only when the parent field itself is on the list: a sour panel
    is a *subdivision* of Vt, and offering it while Vt is too thin to plot would promise more
    resolution than the population has.
    """
    fields = selectable_fields(variant, min_n, mc_cohort)
    c = population_counts(variant, mc_cohort)
    out = []
    for f in fields:
        out.append(f)
        out += [k for k, (fld, _) in SOUR_SPLIT.items()
                if fld == f and int(c.get(k, 0)) >= int(min_n)]
    return out


@lru_cache(maxsize=4)
def _panel_funnel(variant: str = "failures", mc_cohort: bool = True,
                  workbook: Path | None = None) -> pd.DataFrame:
    from analysis.workflows.production_risk.svod_nno_decomposition import selection_funnel

    return selection_funnel(workbook, variant=variant, mc_cohort=mc_cohort)


def population_funnel(field: str = "Ya", variant: str = "failures",
                      mc_cohort: bool = True,
                      workbook: Path | None = None) -> pd.DataFrame:
    """``[step, stage, kept, dropped]`` — how the Свод sheet shrinks to this population.

    Every panel in this module is a *filtered* sheet, and the filters are not small: on Ya
    the genuine-failure rule alone removes a third of the closed runs.  A bare "n = 630"
    invites the reader to compare it with a number they know from elsewhere and conclude the
    panel lost rows to something exotic (telemetry coverage, say — which for a Свод axis is
    never applied at all).  This is the audit trail: each row is what survived one named
    filter, and ``dropped`` is what that filter took.
    """
    raw = _panel_funnel(variant, mc_cohort, workbook)
    fld, h2s = resolve_population(field)
    keep = pd.Series(True, index=raw.index) if fld == FLEET else raw["field"] == fld
    if h2s is not None:
        keep = keep & (raw["h2s_class"].astype(str) == h2s)
    # every stage is listed even when the population empties out on an early one: a funnel
    # that just stops is indistinguishable from one that never ran
    steps = raw[["step", "stage"]].drop_duplicates().sort_values("step")
    kept = raw[keep].groupby("step")["kept"].sum()
    g = steps.assign(kept=steps["step"].map(kept).fillna(0).astype(int)).reset_index(drop=True)
    g["dropped"] = (g["kept"].shift(1) - g["kept"]).fillna(0).astype(int)
    return g


def funnel_table(field: str, prop: Property, prepared: pd.DataFrame, *,
                 variant: str = "failures", mc_cohort: bool = True,
                 workbook: Path | None = None) -> pd.DataFrame:
    """:func:`population_funnel` plus the two losses :func:`prepare` adds on this axis.

    ``prepared`` must be the **fact** panel of the same population and variant — the last
    sheet-side stage is what the axis stages then start from.  The censoring-aware panel is
    built by :func:`esp_population.build`, a different population with a different funnel.
    """
    g = population_funnel(field, variant, mc_cohort, workbook)
    a = prepared.attrs
    base = int(g["kept"].iloc[-1])
    lo, hi = (prop.fmt % prop.bounds[0], prop.fmt % prop.bounds[1])
    axis = pd.DataFrame({
        "step": [int(g["step"].iloc[-1]) + 1, int(g["step"].iloc[-1]) + 2],
        "stage": [f"есть значение «{prop.label}»",
                  f"значение в окне {lo}–{hi} {prop.unit} (вне окна — брак ввода)"],
        "kept": [base - int(a["n_missing"]), int(a["n_used"])]})
    axis["dropped"] = [int(a["n_missing"]), int(a["n_out_of_bounds"])]
    return pd.concat([g, axis], ignore_index=True)


@lru_cache(maxsize=1)
def warehouse_covariates() -> pd.DataFrame:
    """``(code, install) → glf / qg / wcut`` from the t0 window.

    Свод carries no gas column, so ГЖ can only come from the daily warehouse.  The join key
    is exact rather than fuzzy: :func:`esp_population.build` — which the covariate frame is
    assembled on — keeps the Свод row's own ``code`` and ``install``, so a Свод run and its
    covariate row are the *same* record, not a matched pair.

    Cached per process: assembling the frame reads the whole daily mart and takes minutes.
    """
    from analysis.workflows.production_risk import vt_ttf_covariates as V

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = V.build_frame()[0]
    keep = ["code", "install", *WAREHOUSE_COLS]
    return df[keep].drop_duplicates(["code", "install"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def pressure_covariates(window_op_days: int = PRESSURE_WINDOW_OP_DAYS) -> pd.DataFrame:
    """``(code, install) → pzab / pbub / pzab_over_pbub``, from telemetry + the run mart.

    ``Pзаб`` is a real per-day measurement (``proc__daily_merged.rzab``) averaged over the
    run's first ``window_op_days`` operating days, guarded to :data:`RZAB_PLAUSIBLE`.

    ``Pнас`` is **not** a measurement in the same sense.  ``mart__weibull_input.pbubble_atm``
    is close to a field label: on Ya it spans 230–259.5 atm, a between-well CV of 3.7 %, and
    it is not even stable within a well (233 of 372 Ya wells carry more than one value across
    their runs), so it is taken as the well's median here rather than trusted per run.

    ⚠ **Read the ratio with that in mind.**  Divided by something that barely varies, the
    ratio is Pзаб rescaled: on Ya the two rank runs at Spearman 0.997, so within one field
    their binned curves are the same curve.  What the ratio does add is the *threshold* at
    1.0 — below it gas breaks out in the formation — and even that is nearly contrast-free
    here, since 98.6 % of Ya runs sit below it.
    """
    import sqlite3

    from analysis.workflows.production_risk.esp_population import WAREHOUSE_DIR

    pop = esp_population.build(as_of=pd.Timestamp(C.SVOD_OPEN_ASOF))
    pop = pop[["code", "install", "end"]].dropna(subset=["code", "install"]).copy()
    pop["well_key"] = pop["code"].astype(str).str.casefold()

    con = sqlite3.connect(WAREHOUSE_DIR / "pump2.db")
    try:
        lo, hi = RZAB_PLAUSIBLE
        daily = pd.read_sql(
            "SELECT well_key, dt, rzab FROM proc__daily_merged "
            "WHERE qliq > 0 AND rzab BETWEEN ? AND ?", con, params=[lo, hi])
        pbub = pd.read_sql(
            "SELECT well_key, pbubble_atm FROM mart__weibull_input WHERE pbubble_atm > 0",
            con)
    finally:
        con.close()

    daily["dt"] = pd.to_datetime(daily["dt"], errors="coerce")
    daily = daily.dropna(subset=["dt"])
    by_well = {k: g.sort_values("dt") for k, g in daily.groupby("well_key", sort=False)}
    pbub["well_key"] = pbub["well_key"].astype(str).str.casefold()
    pbub_by_well = pbub.groupby("well_key")["pbubble_atm"].median()

    vals = []
    for wk, install, end in zip(pop["well_key"], pop["install"], pop["end"]):
        g = by_well.get(wk)
        if g is None:
            vals.append(np.nan)
            continue
        stop = pd.Timestamp(end) if pd.notna(end) else pd.Timestamp.max
        win = g[(g["dt"] >= pd.Timestamp(install)) & (g["dt"] <= stop)].head(window_op_days)
        vals.append(float(win["rzab"].mean()) if len(win) else np.nan)

    out = pop[["code", "install"]].copy()
    out["pzab"] = vals
    out["pbub"] = pop["well_key"].map(pbub_by_well).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pzab_over_pbub"] = out["pzab"] / out["pbub"].where(out["pbub"] > 0)
    return out.drop_duplicates(["code", "install"]).reset_index(drop=True)


#: Extra source → (columns it brings, loader).  ``"svod"`` needs no join.
ATTACHERS = {"warehouse": (WAREHOUSE_COLS, warehouse_covariates),
             "pressure": (PRESSURE_COLS, pressure_covariates)}


def attach_source(panel: pd.DataFrame, source: str) -> pd.DataFrame:
    """Left-join whatever a property's ``source`` needs.

    Runs the mart never saw keep NaN and drop out in :func:`prepare`, which reports how
    many — coverage on these axes is a real limit and is never silently absorbed.
    """
    if source == "svod":
        return panel
    cols, loader = ATTACHERS[source]
    if all(c in panel.columns for c in cols):
        return panel
    return panel.merge(loader(), on=["code", "install"], how="left")


def attach_warehouse(panel: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible alias for ``attach_source(panel, "warehouse")``."""
    return attach_source(panel, "warehouse")


def load_fact(field: str = "Ya", *, variant: str = "failures", mc_cohort: bool = True,
              source: str = "svod", workbook: Path | None = None) -> pd.DataFrame:
    """The fact panel: one row per closed Свод run with a reported ННО.

    ``variant="failures"`` is genuine failures only (ГТМ/ППР and no-signal pulls censored, so
    absent here); ``"all_closed"`` keeps every closed run.  Neither contains running pumps —
    that is what :func:`load_censored` is for.

    ``field`` is a population key: a field code, :data:`FLEET`, or one of
    :data:`SOUR_SPLIT`'s Vt halves.

    No covariate window is applied here: which column becomes the x-axis is the caller's
    choice, and each carries its own plausible range.  :func:`prepare` does that, per
    property, and reports what it dropped.
    """
    from analysis.workflows.production_risk.svod_nno_decomposition import load_panel

    panel = load_panel(workbook, variant=variant, mc_cohort=mc_cohort)
    g = panel[population_mask(panel, field)].rename(columns={"nno": "ttf"}).copy()
    g = attach_source(g, source)
    g.attrs["n_field"] = len(g)
    g.attrs["source"] = f"Свод · {population_label(field)} · {variant}"
    return g.reset_index(drop=True)


def load_censored(field: str = "Ya", *, as_of=None, mc_cohort: bool = True,
                  source: str = "svod", workbook: Path | None = None) -> pd.DataFrame:
    """The same wells with the **running pumps put back** — the censoring-aware population.

    Built through :func:`esp_population.build`, so it is the canonical population every fit in
    this workflow uses (open Свод rows aged to ``SVOD_OPEN_ASOF``, stale open rows dropped,
    Big-only closed runs included), with the sheet's own covariate columns carried along the
    Свод rows.

    The covariates are deliberately taken from the **same sheet columns** the fact panel uses,
    not from the telemetry run-means: the point of comparing these two panels is to change one
    thing — who is in the population — and swapping the covariate definition at the same time
    would confound it.  Big-only runs carry no sheet covariates and drop out in
    :func:`prepare`; that cost is reported rather than hidden.

    The H₂S class is the **v3.2 well-level relabel**, forced on for this panel regardless of
    ``config.SOUR_WELL_LEVEL_ALL_RUNS``.  That flag ships False, and with it off the Свод
    «Кислый/Некислый» flag — written only from the failure/workover DB — never reaches a
    *running* pump: Vt_sour then comes back with **zero open runs**, i.e. a censoring-aware
    panel with the censoring structurally removed, which is the failures-only panel wearing a
    KM curve.  The fact panel rolls the flag up to the well unconditionally
    (``svod_nno_decomposition.load_panel``) and the Vt models are fitted on the relabelled
    strata, so forcing it here is what makes the two panels and the layer models agree on what
    "sour" means.  ⚠ The roll-up is an OR: one false-positive sour workover row makes the whole
    well sour.
    """
    src = Path(workbook or crosswalk.resolve_prediction_workbook_path())
    svod = esp_population.load_svod_runs(src)
    sheet = pd.read_excel(src, sheet_name="Свод")
    num = {c: pd.to_numeric(sheet[c], errors="coerce").reindex(svod.index)
           for c in SVOD_COVARIATES}
    svod = svod.assign(freq=num["Частота"], ql=num["Дебит жидк."],
                       qnom=num["Ном. Произв. м₃/сут"])

    prev = C.SOUR_WELL_LEVEL_ALL_RUNS
    C.SOUR_WELL_LEVEL_ALL_RUNS = True
    try:
        pop = esp_population.build(as_of=as_of or pd.Timestamp(C.SVOD_OPEN_ASOF), svod=svod)
    finally:
        C.SOUR_WELL_LEVEL_ALL_RUNS = prev
    if mc_cohort:
        pop = esp_population.apply_mc_cohort(pop)
    keep = (pop["tte"] > 0) & population_mask(pop, field)
    g = pop[keep].rename(columns={"tte": "t"}).copy()
    g["kpod"] = np.where(g["qnom"] > 0, g["ql"] / g["qnom"], np.nan)
    g = attach_source(g, source)
    g.attrs["n_field"] = len(g)
    g.attrs["n_open"] = int(g["end"].isna().sum())
    g.attrs["source"] = f"популяция ESP · {population_label(field)} · с цензурированием"
    return g.reset_index(drop=True)


def prepare(panel: pd.DataFrame, prop: Property) -> pd.DataFrame:
    """Rows usable on ``prop``'s axis, with the covariate exposed as ``x``.

    Two separate losses are recorded in ``attrs`` and both matter: ``n_missing`` is runs the
    column is simply blank for (for ГЖ, everything the daily mart never covered), while
    ``n_out_of_bounds`` is runs whose value is present but impossible — 236 Hz, a 9000 m³/d
    nameplate.  The first is a coverage limit on the panel; the second is data quality.
    """
    d = panel.copy()
    if prop.column not in d.columns:
        raise KeyError(f"panel has no column {prop.column!r} for property {prop.key!r} — "
                       f"load it with with_warehouse=True?")
    x = pd.to_numeric(d[prop.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    d["x"] = x
    n_all = len(d)
    n_missing = int(x.isna().sum())
    d = d[x.notna() & x.between(*prop.bounds)]
    out = d.reset_index(drop=True)
    out.attrs = dict(panel.attrs)
    out.attrs |= {"property": prop.key, "n_before": n_all, "n_missing": n_missing,
                  "n_out_of_bounds": n_all - n_missing - len(out), "n_used": len(out)}
    return out


# ---------------------------------------------------------------------------
# Fitted hazard layers, removed as an AFT time-scale offset
# ---------------------------------------------------------------------------
#: Layer key → (label, the PROPERTIES axis it is the same variable as).
#: The second field is what lets the app notice you are about to flatten the very axis you
#: are looking at.
LAYER_AXIS = {"ql": ("θ_Ql", "ql"), "qnom": ("θ_Qном", "qnom"),
              "kpod": ("θ_Kпод", "kpod"), "freq": ("θ_частоты", "freq")}
#: Layer key → the panel column its θ is evaluated on.
LAYER_COLUMN = {"ql": "ql", "qnom": "qnom", "kpod": "kpod", "freq": "freq"}


@dataclass
class LayerModel:
    """A fitted model, reduced to what this panel needs: β and one θ per layer.

    The models themselves live in their own modules and are not touched here; this is the
    adapter that lets a binned panel ask them "what does your Ql layer say about this run".
    """
    key: str
    label: str
    field: str
    layers: tuple
    beta_of: object            # (panel) -> per-row β
    theta_of: object           # (panel, layer_key) -> per-row θ
    note: str = ""

    def theta(self, panel: pd.DataFrame, layer: str) -> np.ndarray:
        return np.asarray(self.theta_of(panel, layer), float)

    def beta(self, panel: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.beta_of(panel), float)


@lru_cache(maxsize=4)
def _ya_fitted(rate: str):
    """Ya k1/k2 hybrid, fitted once per process.  ``n_boot=0``: we want θ, not its band."""
    from analysis.workflows.production_risk import ya_k1k2_hybrid as YA
    return YA.run(rate=rate, n_boot=0, write=False).model


@lru_cache(maxsize=1)
def _vt_fitted():
    from analysis.workflows.production_risk import vt_v4 as V4
    return V4.run(write=False).model


def _num(panel: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(panel.get(col), errors="coerce").to_numpy(float)


def _ya_layer_model(key: str, rate: str, label: str, note: str) -> LayerModel:
    rate_key = "ql" if rate == "Ql" else "qnom"

    def theta_of(panel, layer):
        m = _ya_fitted(rate)
        x = _num(panel, LAYER_COLUMN[layer])
        if layer == "freq":
            th = m.theta_freq_hz(np.nan_to_num(x, nan=50.0))
        else:
            arm = {"ql": "Ql", "qnom": "Qnom", "kpod": "Kpod"}[layer]
            th = m.theta_at(arm, np.nan_to_num(x, nan=1.0))
        # a run whose covariate is unknown cannot be adjusted; θ = 1 leaves it alone
        return np.where(np.isfinite(x), np.asarray(th, float), 1.0)

    def beta_of(panel):
        return np.full(len(panel), float(_ya_fitted(rate).baseline.beta1))

    return LayerModel(key=key, label=label, field="Ya", layers=(rate_key, "kpod", "freq"),
                      beta_of=beta_of, theta_of=theta_of, note=note)


def _vt_layer_model() -> LayerModel:
    def theta_of(panel, layer):
        from analysis.workflows.production_risk import vt_v4 as V4
        x = _num(panel, LAYER_COLUMN[layer])
        safe = np.nan_to_num(x, nan=1.0)
        if layer == "freq":
            th = V4.theta_freq(np.nan_to_num(x, nan=50.0))
        elif layer == "kpod":
            th = V4.theta_kpod(safe)
        else:
            m = _vt_fitted()
            cls = panel.get("h2s_class", pd.Series(["nonsour"] * len(panel))).astype(str)
            th = np.ones(len(panel))
            for stratum, idx in pd.Series(range(len(panel))).groupby(cls.to_numpy()):
                loc = idx.to_numpy()
                th[loc] = m.theta_qnom(str(stratum), safe[loc])
        return np.where(np.isfinite(x), np.asarray(th, float), 1.0)

    def beta_of(panel):
        m = _vt_fitted()
        cls = panel.get("h2s_class", pd.Series(["nonsour"] * len(panel))).astype(str)
        return np.array([m.baseline(str(s))[0] for s in cls])

    return LayerModel(
        key="vt_v4", label="Vt v4 (θ_Qном)", field="Vt", layers=("qnom", "kpod", "freq"),
        beta_of=beta_of, theta_of=theta_of,
        note="Слой Qном подогнан на Vt, θ_частоты перенесён с Ya, θ_Kпод — накладка, по "
             "умолчанию ≡ 1 (измеренный ноль), так что её снятие ничего не меняет.")


#: Where the deployed per-field parameter blocks are written.
FIELD_V5_SLUG = "production_risk_field_v4"


@lru_cache(maxsize=1)
def _field_v5_blocks() -> tuple[dict, dict]:
    """The deployed ``TuneBlock`` / ``QnomBlock``: key → (β, …) and key → θ at the knots.

    Read from the newest ``field_v4`` run — the same two CSVs the calculators are wired from,
    so the viewer and the workbooks cannot disagree about what a field's Qnom layer is.
    """
    from analysis.paths import RESULTS_ROOT

    found = sorted((RESULTS_ROOT / FIELD_V5_SLUG).glob("*/tables"))
    if not found:
        raise FileNotFoundError(f"no {FIELD_V5_SLUG} results — run scripts/run/field_v4.py")
    tune = pd.read_csv(found[-1] / "deploy_tune_block.csv")
    qnom = pd.read_csv(found[-1] / "deploy_qnom_block.csv")
    qcols = [c for c in qnom.columns if c.startswith("q")]
    return ({r["key"]: float(r["beta"]) for _, r in tune.iterrows()},
            {r["key"]: (np.array([float(c[1:]) for c in qcols]),
                        np.array([float(r[c]) for c in qcols])) for _, r in qnom.iterrows()})


def field_v5_key(field, h2s) -> np.ndarray:
    """Per-run stratum key: ``Vt`` splits on H₂S, everything else is its own code.

    A field with no row of its own resolves to ``Fleet`` — the pooled fit — which is exactly
    what :func:`ModuleFailureV4.CM4_ResolveKey` does in the workbooks.
    """
    _, blocks = _field_v5_blocks()
    f = pd.Series(field).astype(str).to_numpy()
    cls = pd.Series(h2s).astype(str).to_numpy()
    out = np.where(f == "Vt", np.char.add("Vt_", cls), f)
    return np.array([k if k in blocks else "Fleet" for k in out])


def _field_v5_layer_model() -> LayerModel:
    """Every field's own θ_Qnom, plus the deployed operator layers for freq and Kpod.

    This is the one layer model that works on **«Весь фонд»** as well as on a single field.
    The reason is the pin: every field's θ_Qnom is pinned to 1 at the same Qnom 250, so runs
    from different fields are being rescaled toward *the same* reference pump — unlike the
    per-field baselines, which are pinned to their own η₀ and cannot be mixed on one panel.
    """
    from analysis.workflows.production_risk import pikpolka_sim as PS

    def theta_of(panel, layer):
        x = _num(panel, LAYER_COLUMN[layer])
        safe = np.nan_to_num(x, nan=1.0)
        if layer == "freq":
            th = np.array([PS.theta_freq(float(v), PS.CTRL_BASE)
                           for v in np.nan_to_num(x, nan=50.0)])
        elif layer == "kpod":
            th = np.array([PS.theta_kpod(float(v), PS.CTRL_BASE) for v in safe])
        else:
            _, blocks = _field_v5_blocks()
            keys = field_v5_key(panel.get("field", pd.Series([FLEET] * len(panel))),
                                panel.get("h2s_class", pd.Series(["nonsour"] * len(panel))))
            th = np.ones(len(panel))
            for key in np.unique(keys):
                loc = np.flatnonzero(keys == key)
                kn, tv = blocks[key]
                q = np.clip(safe[loc], kn[0], kn[-1])
                th[loc] = np.exp(np.interp(np.log(q), np.log(kn), np.log(tv)))
        return np.where(np.isfinite(x), np.asarray(th, float), 1.0)

    def beta_of(panel):
        betas, _ = _field_v5_blocks()
        keys = field_v5_key(panel.get("field", pd.Series([FLEET] * len(panel))),
                            panel.get("h2s_class", pd.Series(["nonsour"] * len(panel))))
        return np.array([betas.get(k, betas.get("Fleet", 1.0)) for k in keys])

    return LayerModel(
        key="field_v5", label="Пофондовая v5 (θ_Qном по участку)", field="*",
        layers=("qnom", "kpod", "freq"), beta_of=beta_of, theta_of=theta_of,
        note="θ_Qном подобран ОТДЕЛЬНО по каждому участку недр (Ya, Vt кисл./некисл., Az, "
             "Ic, Au, Mc), участок без своей строки уходит на Fleet — пул по всему фонду. "
             "Работает и на «Весь фонд»: все слои Qном пинятся к ОДНОМУ опорному насосу "
             "Qном 250, поэтому разные месторождения приводятся к общей точке. θ_частоты и "
             "θ_Кпод — сценарий «Базовый» из калькулятора, они ЗАДАНЫ, а не подобраны.")


#: Model key → builder.  Only fields with a fitted layer model appear here; on the rest the
#: app simply offers no adjustment rather than inventing one.
LAYER_MODELS = {
    "field_v5": _field_v5_layer_model,
    "ya_v2": lambda: _ya_layer_model(
        "ya_v2", "Ql", "Ya v2 (θ_Ql)",
        "Рейтинговый слой — ИЗМЕРЕННЫЙ дебит жидкости. Исходная версия модели."),
    "ya_v21": lambda: _ya_layer_model(
        "ya_v21", "Qnom", "Ya v2.1 (θ_Qном)",
        "Рейтинговый слой — ПАСПОРТНАЯ подача насоса. Насос выбирается при монтаже, поэтому "
        "его номинал не может быть загрязнён тем, чем кончился пуск; собственный отбор "
        "модели предпочитает эту версию (Qном перекрывает Ql)."),
    "vt_v4": _vt_layer_model,
}


# ---------------------------------------------------------------------------
# The empirical layer — fitted on whatever data is on screen
# ---------------------------------------------------------------------------
#: Minimum events before an empirical slope is worth estimating at all.
EMPIRICAL_MIN_EVENTS = 30


@dataclass(frozen=True)
class EmpiricalFit:
    """A Weibull-AFT slope of life on one covariate, fitted on the selected panel.

    ``log T = a[group] + c · log(x / ref) + σ·W`` — so ``c`` is the **elasticity of life**:
    ``c = −0.4`` means doubling the covariate costs ``1 − 2^-0.4 ≈ 24 %`` of life.  The
    adjustment that removes it is ``t · (x/ref)^(−c)``, the same power-law shape the
    workflow's own fitted θ_Ql turned out to have.

    Why an AFT and not a regression on the binned points: it is fitted on **runs**, it takes
    right-censoring (so it is usable on the censored population, where a plain regression is
    not), and its offset lands directly on the time scale with no β to divide by.

    ``groups`` gives each group its own intercept and keeps ONE shared slope.  On a pooled
    fleet panel that is the difference between a within-field slope and a slope that is
    partly the fields taking turns; with a single field it changes nothing.
    """
    column: str
    ref: float
    c: float                  # log-life per log-covariate (negative = shorter life)
    se: float
    shape: float              # Weibull shape of the AFT residual
    n: int
    events: int
    support: tuple
    n_groups: int = 1

    @property
    def per_doubling(self) -> float:
        """Share of life lost when the covariate doubles (negative = gained)."""
        return float(1.0 - 2.0 ** self.c)

    @property
    def z(self) -> float:
        return float(self.c / self.se) if self.se > 0 else np.nan

    def time_multiplier(self, x) -> np.ndarray:
        """``(x/ref)^(−c)``, clamped to the fitted support — never extrapolated.

        ``atleast_1d`` because a scalar argument would otherwise come back 0-d and raise on
        the ``[0]`` a caller writes — the same trap ``theta_freq_poly`` documents.
        """
        v = np.clip(np.atleast_1d(np.asarray(x, float)), *self.support)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (v / self.ref) ** (-self.c)
        return np.where(np.isfinite(out), out, 1.0)

    def __str__(self) -> str:
        d = self.per_doubling
        verb = "отнимает" if d > 0 else "добавляет"
        grp = f", свой уровень у каждого из {self.n_groups}" if self.n_groups > 1 else ""
        return (f"c = {self.c:+.3f} ± {self.se:.3f} (z = {self.z:+.1f}); удвоение "
                f"«{self.column}» {verb} {abs(d):.1%} срока; {self.events} отказов "
                f"из {self.n}{grp}")


def _inv_num_hessian(f, p, eps: float = 1e-4) -> np.ndarray:
    """Inverse of the central-difference Hessian of ``f`` at ``p`` — the covariance matrix.

    Falls back to a pseudo-inverse when the Hessian is singular (a group with no events, a
    covariate with no spread), so a degenerate fit reports a huge SE rather than raising.
    """
    p = np.asarray(p, float)
    n = p.size
    h = np.maximum(np.abs(p), 1.0) * eps
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            pp, pm, mp, mm = p.copy(), p.copy(), p.copy(), p.copy()
            pp[i] += h[i]; pp[j] += h[j]
            pm[i] += h[i]; pm[j] -= h[j]
            mp[i] -= h[i]; mp[j] += h[j]
            mm[i] -= h[i]; mm[j] -= h[j]
            H[i, j] = H[j, i] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h[i] * h[j])
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(H)


def fit_empirical_layer(x, t, event=None, *, column: str = "ql", ref: float | None = None,
                        groups=None, min_events: int = EMPIRICAL_MIN_EVENTS) -> EmpiricalFit:
    """Weibull AFT of life on ``log x``, right-censoring aware, one slope per call.

    Raises rather than returning a shaky number when there is too little to fit: a slope on
    a dozen events would be removed from the time scale as confidently as a good one, and
    nothing downstream could tell the difference.
    """
    from scipy.optimize import minimize

    x = np.asarray(x, float)
    t = np.asarray(t, float)
    ev = np.ones_like(t) if event is None else (np.asarray(event, float) == 1).astype(float)
    g = (np.zeros(len(x), int) if groups is None
         else pd.factorize(pd.Series(groups).astype(str))[0])
    ok = np.isfinite(x) & (x > 0) & np.isfinite(t) & (t > 0)
    x, t, ev, g = x[ok], t[ok], ev[ok], g[ok]
    if ev.sum() < min_events:
        raise ValueError(f"{int(ev.sum())} событий — меньше {min_events}, "
                         f"эмпирический слой не оценивается")
    ref = float(np.exp(np.mean(np.log(x)))) if ref is None else float(ref)
    u = np.log(x / ref)
    n_g = int(g.max()) + 1
    lt = np.log(t)

    def nll(p):
        a, c, lk = p[:n_g], p[n_g], p[n_g + 1]
        k = np.exp(lk)
        log_eta = a[g] + c * u
        z = k * (lt - log_eta)
        # log f = lk + z − log t − e^z ;  log S = −e^z
        return float(-np.sum(ev * (lk + z - lt) - np.exp(np.clip(z, -700, 700))))

    p0 = np.concatenate([np.full(n_g, float(np.mean(lt))), [0.0, 0.0]])
    res = minimize(nll, p0, method="BFGS")
    c, lk = float(res.x[n_g]), float(res.x[n_g + 1])
    # BFGS's own ``hess_inv`` is an approximation accumulated along the path, and it degrades
    # badly as parameters are added — on a 7-field fleet fit it inflated this SE tenfold.
    # The observed information is cheap here (≤ 9 parameters), so compute it directly.
    se = float(np.sqrt(max(_inv_num_hessian(nll, res.x)[n_g, n_g], 0.0)))
    return EmpiricalFit(column=column, ref=ref, c=c, se=se, shape=float(np.exp(lk)),
                        n=int(len(x)), events=int(ev.sum()),
                        support=(float(x.min()), float(x.max())), n_groups=n_g)


def empirical_layer_model(fit: EmpiricalFit) -> LayerModel:
    """Wrap an :class:`EmpiricalFit` so it removes through the same path as a fitted model.

    ``β ≡ 1`` here and "θ" carries the whole time-scale multiplier, so ``t · θ^(1/β)``
    degenerates to ``t · θ`` — the AFT offset the fit already is.  That keeps one code path
    for both kinds of layer instead of inventing a hazard shape the empirical fit never
    estimated.
    """
    key = fit.column

    def theta_of(panel, layer):
        v = pd.to_numeric(panel[LAYER_COLUMN[layer]], errors="coerce").to_numpy(float)
        return np.where(np.isfinite(v), fit.time_multiplier(v), 1.0)

    return LayerModel(key="empirical", label=f"эмпирический (по этим данным): {key}",
                      field="*", layers=(key,), beta_of=lambda p: np.ones(len(p)),
                      theta_of=theta_of, note=str(fit))


#: What a UI needs to render the selector **without paying for a fit**: label, field, layers.
#: Nothing here triggers model fitting; :func:`layer_model` does, and only when an adjustment
#: is actually asked for.
LAYER_MODEL_INFO = {
    "field_v5": {"label": "Пофондовая v5 (θ_Qном по участку)", "field": "*",
                 "layers": ("qnom", "kpod", "freq"),
                 "note": "θ_Qном подобран ОТДЕЛЬНО по каждому участку недр; участок без своей "
                         "строки уходит на Fleet (пул по всему фонду). Работает и на «Весь "
                         "фонд»: все слои Qном пинятся к одному опорному насосу Qном 250. "
                         "⚠ θ_частоты и θ_Кпод здесь — сценарий «Базовый» калькулятора: они "
                         "ЗАДАНЫ оператором, а не подобраны (при свободной подгонке частота — "
                         "ноль во всех фондах)."},
    "ya_v2": {"label": "Ya v2 (θ_Ql)", "field": "Ya",
              "layers": ("ql", "kpod", "freq"),
              "note": "Рейтинговый слой — ИЗМЕРЕННЫЙ дебит жидкости. Исходная версия."},
    "ya_v21": {"label": "Ya v2.1 (θ_Qном)", "field": "Ya",
               "layers": ("qnom", "kpod", "freq"),
               "note": "Рейтинговый слой — ПАСПОРТНАЯ подача. Насос выбирают при монтаже, "
                       "поэтому номинал не может быть загрязнён исходом пуска; отбор самой "
                       "модели предпочитает эту версию (Qном перекрывает Ql). ⚠ θ_Kпод "
                       "здесь плоский (измеренный ноль) — его снятие ничего не меняет."},
    "vt_v4": {"label": "Vt v4 (θ_Qном)", "field": "Vt",
              "layers": ("qnom", "kpod", "freq"),
              "note": "Слой Qном подогнан на Vt, θ_частоты перенесён с Ya, θ_Kпод — "
                      "накладка, по умолчанию ≡ 1."},
}


@lru_cache(maxsize=4)
def layer_model(key: str) -> LayerModel:
    """Fitted layer model by key.  The first call fits; afterwards it is cached."""
    if key not in LAYER_MODELS:
        raise KeyError(f"unknown layer model {key!r}; have {tuple(LAYER_MODELS)}")
    return LAYER_MODELS[key]()


def models_for_field(field: str) -> list[str]:
    """Keys of the layer models fitted on ``field`` — empty where none is wired.

    A :data:`SOUR_SPLIT` population resolves to its parent field: the Vt models are fitted
    with the sour/nonsour stratum inside them (θ per stratum, and ``beta_of`` reads
    ``h2s_class`` per run), so they apply unchanged to either half.

    ``field_v5`` is marked ``"*"`` and is offered everywhere, including :data:`FLEET`: it
    carries a θ_Qnom per stratum and they are all pinned at the same Qnom 250, so mixing
    fields on one panel does not mix reference points.
    """
    parent = resolve_population(field)[0]
    return [k for k, v in LAYER_MODEL_INFO.items()
            if v["field"] == parent or v["field"] == "*"]


def remove_layers(panel: pd.DataFrame, time_col: str, model: LayerModel,
                  layers) -> tuple[pd.Series, dict]:
    """``t_adj = t · Πθ ** (1/β)`` — the selected layers taken off the time scale.

    This is an **AFT offset, not a residual**.  A run held at a punishing rate carries θ > 1,
    so its observed life is scaled *up* to what it would have been at the model's reference;
    the result is still a life in days on the same axis as the raw panel, which is what makes
    adjusted and unadjusted curves directly comparable.  A subtracted residual would not be.

    Monotone in ``t`` and strictly positive, so it commutes with every order statistic —
    the median of the adjusted panel is the adjusted median.  It applies to censored rows
    exactly as to failures (a time-scale change moves both), so the KM panel stays valid.

    Runs whose covariate is missing get θ = 1 for that layer, i.e. are passed through
    unadjusted, and are counted in the returned meta.  Dropping them instead would change the
    population every time a layer is toggled, and then a curve that moved would not say
    whether the adjustment or the sample did it.
    """
    t = pd.to_numeric(panel[time_col], errors="coerce").to_numpy(float)
    layers = tuple(layers)
    theta = np.ones(len(panel))
    unadjusted = np.zeros(len(panel), bool)
    for lay in layers:
        if lay not in model.layers:
            raise KeyError(f"{model.key} has no layer {lay!r}; has {model.layers}")
        theta = theta * model.theta(panel, lay)
        unadjusted |= ~np.isfinite(_num(panel, LAYER_COLUMN[lay]))
    beta = model.beta(panel)
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = t * theta ** (1.0 / beta)
    meta = {"layers": layers, "model": model.key,
            "n_unadjusted": int(unadjusted.sum()) if layers else 0,
            "theta_median": float(np.nanmedian(theta)) if layers else 1.0,
            "beta_median": float(np.nanmedian(beta))}
    return pd.Series(np.where(np.isfinite(adj), adj, t), index=panel.index), meta


# ---------------------------------------------------------------------------
# Edge algebra — every operation returns a NEW list
# ---------------------------------------------------------------------------
def sanitize_edges(edges, min_width: float = MIN_BIN_WIDTH) -> list[float]:
    """Sorted, de-duplicated, nothing closer together than ``min_width``.

    Collapsing rather than raising is deliberate: this runs behind a UI where a dragged edge
    passes through its neighbour on the way somewhere sensible, and an exception there would
    be a worse answer than a merged bin.

    ``min_width`` is in the caller's own units, and there is no universal default that is
    safe: 0.5 is right for Hz and would collapse an entire Kpod grid, whose bins are 0.05
    wide.  Consumers that only need "sorted and unique" therefore pass **0**, and only the
    edge-editing operations impose a real minimum.
    """
    e = sorted(float(x) for x in edges if np.isfinite(x))
    if len(e) < 2:
        raise ValueError("need at least two edges")
    gap = max(float(min_width), 0.0)
    out = [e[0]]
    for x in e[1:]:
        if (x - out[-1] >= gap) if gap > 0 else (x > out[-1]):
            out.append(x)
    if len(out) < 2:                       # everything collapsed onto the first edge
        out.append(e[0] + (gap if gap > 0 else abs(e[0]) * 1e-6 + 1e-9))
    return out


def _round_sig(v, sig: int = 3) -> np.ndarray:
    """Round to ``sig`` significant figures — readable edges on a log axis.

    A 0.05 dex quantum puts adjacent edges 12 % apart while 3-figure rounding moves each by
    at most 0.5 %, so the order is safe and the bin counts barely move; see ``Property.sig``
    for why 2 figures is not enough on a catalog variable.
    """
    a = np.asarray(v, float)
    out = np.zeros_like(a)
    nz = a != 0
    mag = np.floor(np.log10(np.abs(a[nz])))
    out[nz] = np.round(a[nz] / 10.0 ** (mag - sig + 1)) * 10.0 ** (mag - sig + 1)
    return out


def uniform_edges(lo: float = FREQ_BOUNDS[0], hi: float = FREQ_BOUNDS[1],
                  width: float = DEFAULT_WIDTH, *, scale: str = "linear",
                  min_width: float = MIN_BIN_WIDTH, sig: int = 3) -> list[float]:
    """Even grid — 2 Hz by default, and what "reset" goes back to.

    On a log axis "even" means **equal ratio**, not equal width: ``width`` is then read as a
    multiplier, so 1.5 gives 100 / 150 / 225 / …  An equal-*width* grid on a rate axis puts
    nearly every run in the first bin, which is not a grid, it is a histogram of one.
    """
    if scale == "log":
        r = max(float(width), 1.0000001)
        n = max(int(round(np.log(hi / lo) / np.log(r))), 1)
        e = [float(lo) * r ** i for i in range(n + 1)]
        return sanitize_edges(_round_sig(e, sig), min_width=min_width)
    n = max(int(round((hi - lo) / width)), 1)
    return [float(lo + i * width) for i in range(n + 1)]


def equal_count_edges(x, k: int = 10, *, round_to: float = 0.5,
                      bounds: tuple[float, float] | None = None,
                      scale: str = "linear", min_width: float = MIN_BIN_WIDTH,
                      sig: int = 3) -> list[float]:
    """Quantile edges — every bin holds the same number of runs, so every point on the curve
    carries the same precision.

    The fixed-width grid's weakness is the opposite: its tail bins hold 1–3 runs and their
    means swing by hundreds of days.  Edges are rounded for readability (52.5, not 52.4713 —
    significant figures on a log axis, a fixed step on a linear one); duplicates after
    rounding collapse, so ``k`` is an upper bound.
    """
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    q = np.quantile(v, np.linspace(0, 1, k + 1))
    q = _round_sig(q, sig) if scale == "log" else (
        np.round(q / round_to) * round_to if round_to else q)
    lo, hi = bounds if bounds else (np.floor(v.min()), np.ceil(v.max()))
    q[0], q[-1] = min(q[0], lo), max(q[-1], hi)
    return sanitize_edges(q, min_width=min_width)


def catalog_values(x, *, scale: str = "log", tol: float = 1.02) -> list[list[float]]:
    """Distinct values of a **discrete** axis, with the inseparable ones grouped.

    ``tol`` is the axis's own smallest expressible bin — a ratio on a log axis, a width on a
    linear one.  Values closer than that cannot be given separate bins by any grid, so they
    are grouped here **explicitly** rather than being silently collapsed later by
    :func:`sanitize_edges`.  On Qном that folds the same pump recorded three ways
    (243 / 244, 460 / 461 / 464) into one size and leaves the real neighbours apart.

    Returns one list of raw values per group, ascending.
    """
    v = np.asarray(x, float)
    v = np.sort(v[np.isfinite(v) & ((v > 0) | (scale != "log"))])
    if v.size == 0:
        return []
    groups: list[list[float]] = [[float(v[0])]]
    for a in v[1:]:
        first = groups[-1][0]
        close = (a / first <= float(tol)) if scale == "log" else (a - first <= float(tol))
        if close:
            groups[-1].append(float(a))
        else:
            groups.append([float(a)])
    return groups


def catalog_edges(x, *, bounds: tuple[float, float] | None = None, scale: str = "log",
                  min_width: float = 0.0, tol: float = 1.02) -> list[float]:
    """One bin per value present — the grid for an axis that **is** a catalog.

    Cuts sit halfway between neighbouring sizes (geometrically on a log axis), so a bin
    contains one nameplate and nothing else, and no bin is grown to reach a target count: on a
    catalog axis widening does not average nearby operating points, it averages *different
    pumps*, and the step from one size to the next is the shape being looked at.  Thin bins
    are therefore expected and are the honest picture — they stay marked thin, and the trend
    still ignores them.

    Edges are **not** rounded to significant figures: a rounded cut can step over a size and
    take all its runs into the neighbouring bin, which is exactly what this grid exists to
    prevent.  The outer edges are mirrored out from the first and last sizes and then widened
    to ``bounds`` if needed, so no run falls outside the grid.
    """
    groups = catalog_values(x, scale=scale, tol=tol)
    if not groups:
        return list(bounds) if bounds else [0.0, 1.0]
    reps = [float(np.median(g)) for g in groups]
    if len(reps) == 1:
        r = reps[0]
        e = [r / 1.05, r * 1.05] if scale == "log" else [r - 0.5, r + 0.5]
    else:
        cuts = ([float(np.sqrt(a * b)) for a, b in zip(reps, reps[1:])] if scale == "log"
                else [0.5 * (a + b) for a, b in zip(reps, reps[1:])])
        first = (reps[0] ** 2 / cuts[0]) if scale == "log" else (2 * reps[0] - cuts[0])
        last = (reps[-1] ** 2 / cuts[-1]) if scale == "log" else (2 * reps[-1] - cuts[-1])
        e = [first, *cuts, last]
    # the grid must cover every run it was built from, and the axis window if one is given
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    pad = 0.999 if scale == "log" else 1.0
    e[0] = min(e[0], float(v.min()) * pad - (0.0 if scale == "log" else 0.001))
    e[-1] = max(e[-1], float(v.max()) / pad + (0.0 if scale == "log" else 0.001))
    if bounds:
        e[0], e[-1] = min(e[0], float(bounds[0])), max(e[-1], float(bounds[1]))
    return sanitize_edges(e, min_width=min_width)


def observed_centres(tab: pd.DataFrame, x, edges) -> pd.DataFrame:
    """Move ``f_mid`` onto the data: the **median value actually present** in each bin.

    On a catalog axis the marker then sits on the nameplate (200, not the 201.6 middle of the
    bracket around it), which is the only position that reads correctly when the reader knows
    the axis is discrete.  The bracket itself is unchanged and still drawn by the count panel,
    and the original grid midpoint is kept as ``f_mid_grid``.

    Everything downstream — the trend, the ratio mode, the hover — reads ``f_mid``, so this
    keeps points, curve and counts on one x.
    """
    out = tab.copy()
    v = np.asarray(x, float)
    idx = bin_index(v, list(edges))
    out["f_mid_grid"] = out["f_mid"]
    mids = out["f_mid"].to_numpy(float).copy()
    for i in range(len(mids)):
        got = v[(idx == i) & np.isfinite(v)]
        if got.size:
            mids[i] = float(np.median(got))
    out["f_mid"] = mids
    out.attrs = dict(tab.attrs)
    return out


def adaptive_edges(x, *, anchor: tuple[float, float] = ANCHOR, min_n: int = TARGET_BIN_N,
                   bounds: tuple[float, float] = FREQ_BOUNDS, step: float = STEP_HZ,
                   max_width: float = MAX_BIN_WIDTH_HZ, merge_tails: bool = True,
                   scale: str = "linear", min_width: float | None = None,
                   sig: int = 3) -> list[float]:
    """The default build, on either a linear or a logarithmic axis.

    On ``scale="log"`` the whole construction happens in ``log10`` — the seed, the quantum,
    the cap and the sweep — and the edges are mapped back and rounded to ``sig`` significant
    figures.  So ``step`` and ``max_width`` are in **dex** there: a bin is at least ×10^step
    wide and at most ×10^max_width.  Everything the docstring below says about "1 Hz" reads
    as "one quantum"; nothing else about the rule changes.
    """
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log', got {scale!r}")
    mw = MIN_BIN_WIDTH if min_width is None else float(min_width)
    if scale == "linear":
        return sanitize_edges(_adaptive_core(x, anchor, min_n, bounds, step, max_width,
                                             merge_tails), min_width=mw)
    v = np.asarray(x, float)
    v = np.log10(v[np.isfinite(v) & (v > 0)])
    lg = np.log10
    e = _adaptive_core(v, (lg(anchor[0]), lg(anchor[1])), min_n,
                       (lg(bounds[0]), lg(bounds[1])), step, max_width, merge_tails)
    # min_width=0 here on purpose: the core already enforced spacing in log space, and a
    # DATA-unit minimum would delete the low-end edges (5 → 6.3 is a 26 % bin, 1.3 units).
    return sanitize_edges(_round_sig(10.0 ** np.asarray(e, float), sig), min_width=0.0)


def _adaptive_core(x, anchor: tuple[float, float], min_n: int,
                   bounds: tuple[float, float], step: float, max_width: float,
                   merge_tails: bool) -> list[float]:
    """The build rule itself, in whatever coordinate it is handed (Hz, dex, fraction …).

    The rule, in order — frequency in brackets as the worked example:

    1. **Seed** at ``anchor`` [50–51 Hz], one quantum wide.  If even that holds fewer than
       ``min_n`` it widens, alternating right and left so the seed stays centred on the
       operating point.
    2. **Grow outward** from the seed's upper edge: take one quantum; while the bin is short
       of ``min_n``, there is axis left **and it is still narrower than ``max_width``**,
       extend its far edge by another quantum.  So a bin is one quantum wide where the data
       is dense [1 Hz] and up to the cap out in the tails [5 Hz].
    3. **Then the same leftward** from the seed's lower edge.
    4. **Tails**: the outermost bin on each side is whatever is left over.  ``merge_tails``
       folds it into its inner neighbour — but only when the result still fits the cap.

    **``max_width`` outranks ``min_n``.**  Sparse tails cannot reach the target inside the
    cap, so bins out there come back short and are drawn flagged.  That is the intended
    trade: an over-wide bin averages runs with nothing physical in common and hides the very
    structure the binning exists to find, whereas a thin bin merely says "unknown here" — and
    says it in the right place.  Hand edits are not bound by the cap.

    Growing from the middle outward — rather than sweeping from the low end up — is what
    makes the grid stable: the crowded middle is cut at the same places regardless of what
    the sparse tails do, whereas a one-way sweep lets one extra run at the far end shift
    every edge above it.
    """
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    lo_b, hi_b = float(bounds[0]), float(bounds[1])
    cap = max(float(max_width), float(step))
    tol = 1e-9

    def count(a: float, b: float) -> int:
        # half-open bins, except that the very top edge is closed (matches ``bin_index``)
        return int(np.sum((v >= a) & (v <= b)) if b >= hi_b else np.sum((v >= a) & (v < b)))

    a_lo = float(np.clip(anchor[0], lo_b, hi_b))
    a_hi = float(np.clip(max(anchor[1], a_lo + step), lo_b, hi_b))
    grow_right = True
    while (count(a_lo, a_hi) < min_n and (a_lo > lo_b or a_hi < hi_b)
           and (a_hi - a_lo) + step <= cap + tol):
        if grow_right and a_hi < hi_b:
            a_hi = min(a_hi + step, hi_b)
        elif a_lo > lo_b:
            a_lo = max(a_lo - step, lo_b)
        else:
            a_hi = min(a_hi + step, hi_b)
        grow_right = not grow_right

    def sweep(start: float, limit: float, sign: int) -> list[float]:
        """Edges from ``start`` out to ``limit``; ``sign`` is +1 right, −1 left."""
        cuts, cur = [], start
        while (cur < limit) if sign > 0 else (cur > limit):
            nxt = cur + sign * step
            nxt = min(nxt, limit) if sign > 0 else max(nxt, limit)
            while (count(*sorted((cur, nxt))) < min_n
                   and ((nxt < limit) if sign > 0 else (nxt > limit))
                   and abs(nxt + sign * step - cur) <= cap + tol):
                nxt = min(nxt + sign * step, limit) if sign > 0 else max(nxt + sign * step, limit)
            cuts.append(nxt)
            cur = nxt
        # the leftover outermost bin is usually short; fold it inward, but never past the cap
        if merge_tails and len(cuts) > 1:
            inner_start = cuts[-3] if len(cuts) > 2 else start
            if (count(*sorted((cuts[-2], cuts[-1]))) < min_n
                    and abs(cuts[-1] - inner_start) <= cap + tol):
                cuts.pop(-2)
        return cuts

    right = sweep(a_hi, hi_b, +1)
    left = sweep(a_lo, lo_b, -1)
    # min_width in THIS coordinate — half a quantum.  The data-unit default would be
    # nonsense in log space, where the whole axis is only three units wide.
    e = sanitize_edges(sorted(left + [a_lo, a_hi] + right), min_width=step * 0.5)
    # The cap can leave an EMPTY sliver at an end (the last step reaches the bound before the
    # next bin is due, and folding it inward would break the cap).  A bin describing no runs
    # is never worth drawing, so the outer edge is pulled in to the data instead.
    while len(e) > 2 and count(e[0], e[1]) == 0:
        e = e[1:]
    while len(e) > 2 and count(e[-2], e[-1]) == 0:
        e = e[:-1]
    return e


def merge_bin(edges, i: int) -> list[float]:
    """Merge bin ``i`` with bin ``i+1`` by deleting the edge between them."""
    e = list(map(float, edges))
    if not 0 <= i < len(e) - 2:
        raise IndexError(f"no bin {i} to merge with its successor ({len(e) - 1} bins)")
    return e[:i + 1] + e[i + 2:]


def split_bin(edges, i: int, at: float | None = None, *,
              min_width: float = MIN_BIN_WIDTH, scale: str = "linear") -> list[float]:
    """Split bin ``i`` at ``at`` by inserting an edge.

    The default cut is the **geometric** midpoint on a log axis and the arithmetic one on a
    linear axis: halving 100–400 at 250 leaves two bins that are nothing alike in the terms
    the axis is measured in, whereas 200 halves it evenly in ratio.
    """
    e = list(map(float, edges))
    if not 0 <= i < len(e) - 1:
        raise IndexError(f"no bin {i} ({len(e) - 1} bins)")
    if at is not None:
        cut = float(at)
    elif scale == "log" and e[i] > 0:
        cut = float(np.sqrt(e[i] * e[i + 1]))
    else:
        cut = 0.5 * (e[i] + e[i + 1])
    if not e[i] < cut < e[i + 1]:
        raise ValueError(f"cut {cut} is outside bin {i} = [{e[i]}, {e[i + 1]})")
    return sanitize_edges(e[:i + 1] + [cut] + e[i + 1:], min_width=min_width)


def move_edge(edges, i: int, to: float, *, min_width: float = MIN_BIN_WIDTH) -> list[float]:
    """Move edge ``i`` to ``to``, clamped so it cannot cross its neighbours.

    Clamping rather than reordering keeps the *identity* of every bin stable while a slider
    is dragged — reordering would silently renumber the bins under the operator's hand.
    """
    e = list(map(float, edges))
    if not 0 <= i < len(e):
        raise IndexError(f"no edge {i} ({len(e)} edges)")
    lo = e[i - 1] + min_width if i > 0 else -np.inf
    hi = e[i + 1] - min_width if i < len(e) - 1 else np.inf
    e[i] = float(np.clip(float(to), lo, hi))
    return e


def merge_thin_bins(edges, x, min_n: int = MIN_BIN_N, *,
                    min_width: float = MIN_BIN_WIDTH) -> list[float]:
    """Repeatedly fold the thinnest under-populated bin into its smaller neighbour.

    Folding into the *smaller* neighbour rather than the nearer one keeps counts even; the
    loop stops when every bin clears ``min_n`` or only one bin is left.  Empty bins go first,
    which is what makes the sparse tails of the fixed grid collapse the way one would by hand.
    """
    e = sanitize_edges(edges, min_width=min_width)
    while True:
        n = bin_counts(x, e)
        if len(n) <= 1 or n.min() >= min_n:
            return e
        i = int(np.argmin(n))
        # merge with whichever neighbour is smaller; at an end there is only one choice
        if i == 0:
            e = merge_bin(e, 0)
        elif i == len(n) - 1:
            e = merge_bin(e, i - 1)
        else:
            e = merge_bin(e, i if n[i + 1] <= n[i - 1] else i - 1)


def bin_index(x, edges) -> np.ndarray:
    """Bin number per observation; ``-1`` for anything outside ``[edges[0], edges[-1])``."""
    e = np.asarray(edges, float)
    v = np.asarray(x, float)
    idx = np.digitize(v, e) - 1
    # digitize puts a value exactly on the top edge in a phantom bin past the last one; the
    # convention everywhere else in this workflow is half-open bins with a CLOSED top end.
    idx = np.where(v == e[-1], len(e) - 2, idx)
    return np.where((idx < 0) | (idx > len(e) - 2) | ~np.isfinite(v), -1, idx)


def bin_counts(x, edges) -> np.ndarray:
    idx = bin_index(x, edges)
    return np.bincount(idx[idx >= 0], minlength=len(edges) - 1)


# ---------------------------------------------------------------------------
# Per-bin statistics
# ---------------------------------------------------------------------------
def _trimmed_mean(y: np.ndarray, trim: float) -> float:
    """Symmetric trimmed mean; ``trim`` is the share removed from EACH tail."""
    if trim <= 0 or len(y) < 3:
        return float(np.mean(y))
    k = int(np.floor(len(y) * float(trim)))
    if 2 * k >= len(y):
        return float(np.median(y))
    return float(np.mean(np.sort(y)[k:len(y) - k]))


def _centres(y: np.ndarray, trim: float) -> dict:
    ly = np.log(y[y > 0])
    return {
        "mean": float(np.mean(y)),
        "median": float(np.median(y)),
        "geomean": float(np.exp(np.mean(ly))) if len(ly) else np.nan,
        "trimmed": _trimmed_mean(y, trim),
    }


def bin_stats(x, y, edges, *, trim: float = 0.1, n_boot: int = 400, seed: int = 20260728,
              ci: float = 95.0) -> pd.DataFrame:
    """Per-bin centres, spread, shape diagnostics and bootstrap intervals.

    Columns worth knowing:

    * ``mean`` / ``median`` / ``geomean`` / ``trimmed`` — the four centres, each with
      ``*_lo`` / ``*_hi`` percentile-bootstrap bounds (resampled **within** the bin, so the
      interval answers "how much would this bin's number move on another draw of these runs",
      which is the question when comparing two adjacent bins);
    * ``sigma_log`` — sd of log t.  This is the spread that generates the mean/median gap;
      when it is roughly constant across bins the two curves differ by a constant factor and
      any *shape* disagreement between them is noise;
    * ``mean_over_median`` and ``p90_over_p50`` — the gap itself, read two ways;
    * ``skew_log`` — skewness on the log scale.  Near 0 means "lognormal, so the geometric
      mean is the right centre"; strongly negative means a clump of early deaths pulling the
      log-scale left tail, i.e. a **mixture**, which no single centre summarises honestly;
    * ``share_under_90d`` — the early-death share, the direct read on that mixture.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    e = sanitize_edges(edges, min_width=0.0)
    idx = bin_index(x, e)
    rng = np.random.default_rng(seed)
    half = (100.0 - float(ci)) / 2.0

    rows = []
    for i in range(len(e) - 1):
        yy = y[(idx == i) & np.isfinite(y)]
        if yy.size == 0:
            rows.append({"bin": i, "f_lo": e[i], "f_hi": e[i + 1],
                         "f_mid": 0.5 * (e[i] + e[i + 1]), "n": 0})
            continue
        ly = np.log(yy[yy > 0])
        cen = _centres(yy, trim)
        row = {"bin": i, "f_lo": e[i], "f_hi": e[i + 1], "f_mid": 0.5 * (e[i] + e[i + 1]),
               "n": int(yy.size), **cen,
               "sd": float(np.std(yy, ddof=1)) if yy.size > 1 else np.nan,
               "se": float(np.std(yy, ddof=1) / np.sqrt(yy.size)) if yy.size > 1 else np.nan,
               "sigma_log": float(np.std(ly, ddof=1)) if ly.size > 1 else np.nan,
               "skew_log": _skew(ly),
               "p10": float(np.percentile(yy, 10)), "p25": float(np.percentile(yy, 25)),
               "p75": float(np.percentile(yy, 75)), "p90": float(np.percentile(yy, 90)),
               "min": float(yy.min()), "max": float(yy.max()),
               "share_under_90d": float(np.mean(yy < 90.0))}
        row["mean_over_median"] = row["mean"] / row["median"] if row["median"] > 0 else np.nan
        row["p90_over_p50"] = row["p90"] / row["median"] if row["median"] > 0 else np.nan

        if n_boot > 0 and yy.size > 1:
            draws = {k: np.empty(n_boot) for k in STATS}
            for b in range(n_boot):
                s = yy[rng.integers(0, yy.size, yy.size)]
                for k, v in _centres(s, trim).items():
                    draws[k][b] = v
            for k, v in draws.items():
                row[f"{k}_lo"] = float(np.nanpercentile(v, half))
                row[f"{k}_hi"] = float(np.nanpercentile(v, 100.0 - half))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_outside"] = int((idx < 0).sum())
    out.attrs["trim"] = trim
    return out


def composition(x, labels, edges, *, top: int = 4) -> pd.DataFrame:
    """Share of each label per bin — the guard a **pooled** panel cannot be read without.

    Fields differ in life by a factor of two or more, so on a fleet panel a bin's average is
    partly a field-mix average.  If the mix slides along the axis, the curve can show a shape
    that no field has: Simpson's paradox, and on a binned panel it is invisible unless the
    mix is drawn.  Read this alongside the fleet curve — where the mix is flat, the shape is
    the covariate; where it swings, suspect the mix first.

    Labels beyond ``top`` (by total count) fold into «прочие» rather than becoming their own
    thin series, which would be unreadable and would imply precision that is not there.
    """
    e = sanitize_edges(edges, min_width=0.0)
    idx = bin_index(np.asarray(x, float), e)
    lab = pd.Series(np.asarray(labels, dtype=object)).astype(str)
    keep = lab.value_counts().index[:int(top)].tolist()
    lab = lab.where(lab.isin(keep), "прочие")
    rows = []
    for i in range(len(e) - 1):
        m = idx == i
        n = int(m.sum())
        if not n:
            continue
        counts = lab[m].value_counts()
        for name in [*keep, "прочие"]:
            k = int(counts.get(name, 0))
            if k or name in keep:
                rows.append({"bin": i, "f_lo": e[i], "f_hi": e[i + 1],
                             "f_mid": 0.5 * (e[i] + e[i + 1]), "label": name,
                             "n": k, "share": k / n, "n_bin": n})
    return pd.DataFrame(rows)


def _skew(v: np.ndarray) -> float:
    if v.size < 3:
        return np.nan
    s = np.std(v, ddof=0)
    return float(np.mean((v - v.mean()) ** 3) / s ** 3) if s > 0 else np.nan


# ---------------------------------------------------------------------------
# Censoring-aware per-bin statistics
# ---------------------------------------------------------------------------
def _km_summary(t: np.ndarray, ev: np.ndarray, horizon: float) -> tuple[float, float]:
    """``(KM median, RMST(0, horizon))``.  Median is NaN when the curve never reaches 0.5."""
    times, surv, _ = kaplan_meier(t, ev)
    below = np.flatnonzero(surv <= 0.5)
    med = float(times[below[0]]) if below.size else np.nan
    return med, float(km_rmst(times, surv, horizon))


def km_bin_stats(x, t, event, edges, *, horizon: float = RMST_HORIZON,
                 n_boot: int = 400, seed: int = 20260728, ci: float = 95.0) -> pd.DataFrame:
    """Per-bin **KM median** and **RMST(0, horizon)** — the same bins, censoring handled.

    This is the panel the fact panel cannot be: ННО only exists for a pulled pump, so a raw
    mean or median of it conditions on having failed.  Here every run enters, a still-running
    pump contributing exposure without an event, and both summaries are read off the KM curve.

    RMST is reported alongside the median because they fail differently: the KM median is
    undefined in a bin whose curve never crosses 0.5 (long-lived, heavily censored bins — the
    good ones), while RMST(0, 730) is always defined and is the project's standing estimand.
    A bin with **little follow-up past the horizon** flatters RMST, so ``surv_at_horizon`` and
    ``max_t`` are emitted to show how much of the area is extrapolated flat.
    """
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    ev = np.asarray(event, float)
    e = sanitize_edges(edges, min_width=0.0)
    idx = bin_index(x, e)
    rng = np.random.default_rng(seed)
    half = (100.0 - float(ci)) / 2.0

    rows = []
    for i in range(len(e) - 1):
        m = (idx == i) & np.isfinite(t) & (t > 0)
        tt, ee = t[m], ev[m]
        base = {"bin": i, "f_lo": e[i], "f_hi": e[i + 1], "f_mid": 0.5 * (e[i] + e[i + 1]),
                "n": int(tt.size), "events": int((ee == 1).sum()),
                "censored": int((ee != 1).sum())}
        if tt.size == 0:
            rows.append(base)
            continue
        med, r = _km_summary(tt, ee, horizon)
        times, surv, _ = kaplan_meier(tt, ee)
        base |= {"km_median": med, "rmst": r,
                 "surv_at_horizon": float(surv[times <= horizon][-1]),
                 "max_t": float(tt.max()),
                 "naive_median": float(np.median(tt[ee == 1])) if (ee == 1).any() else np.nan,
                 "naive_mean": float(np.mean(tt[ee == 1])) if (ee == 1).any() else np.nan}
        if n_boot > 0 and tt.size > 2:
            bm, br = np.empty(n_boot), np.empty(n_boot)
            for b in range(n_boot):
                j = rng.integers(0, tt.size, tt.size)
                bm[b], br[b] = _km_summary(tt[j], ee[j], horizon)
            base |= {"km_median_lo": float(np.nanpercentile(bm, half)),
                     "km_median_hi": float(np.nanpercentile(bm, 100.0 - half)),
                     "rmst_lo": float(np.nanpercentile(br, half)),
                     "rmst_hi": float(np.nanpercentile(br, 100.0 - half))}
        rows.append(base)
    out = pd.DataFrame(rows)
    out.attrs["horizon"] = horizon
    return out


def normalize_to_bin(tab: pd.DataFrame, cols, ref_x: float, *,
                     bounds: tuple[float, float] = (-np.inf, np.inf)) -> pd.DataFrame:
    """Rescale each statistic to a **multiple of its own value in the bin holding** ``ref_x``.

    ``Y(ref_x) = 1`` by construction and every other bin reads as a share of it, which is the
    only way two panels on different time scales — days vs adjusted days, fact vs KM, one
    field vs another — can be compared by *shape*.  Bootstrap bounds are divided by the same
    number so the ribbon stays around its own curve.

    Each statistic is divided by **its own** reference value, not by a shared one: the mean
    and the median differ by a factor in every bin, and a common divisor would fold that
    level difference into the shape being compared.

    ⚠ This is a division by one noisy number.  The reference bin's own sampling error is
    *not* propagated — it enters every point identically, so the ribbon here understates the
    uncertainty of a comparison against the reference and the curve pivots bodily if that bin
    is thin.  Read it for shape; read the unscaled panel for level.

    Statistics the reference bin cannot supply (an empty bin, a KM median that never crosses
    0.5) are blanked rather than silently rescaled by something else, and named in
    ``attrs["unnormalised"]``.
    """
    if "f_lo" not in tab.columns or tab.empty:
        raise ValueError("таблица бинов пуста — нормировать не на что")
    edges = [float(v) for v in tab["f_lo"]] + [float(tab["f_hi"].iloc[-1])]
    i = int(bin_index([float(ref_x)], edges)[0])
    if i < 0:
        raise ValueError(f"опорное значение {ref_x:g} вне сетки "
                         f"{edges[0]:g}–{edges[-1]:g}")
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (lo <= float(ref_x) <= hi):
        raise ValueError(f"опорное значение {ref_x:g} вне окна характеристики "
                         f"{lo:g}–{hi:g}")

    out = tab.copy()
    row = tab.iloc[i]
    refs, unusable = {}, []
    for c in cols:
        targets = [c] + [c + s for s in ("_lo", "_hi") if c + s in out.columns]
        v = float(row[c]) if c in tab.columns and pd.notna(row[c]) else np.nan
        if not np.isfinite(v) or v <= 0:
            unusable.append(c)
            for t in targets:
                out[t] = np.nan
            continue
        refs[c] = v
        for t in targets:
            out[t] = out[t] / v
    out.attrs = dict(tab.attrs)
    out.attrs |= {"ref_bin": i, "ref_x": float(ref_x),
                  "ref_lo": edges[i], "ref_hi": edges[i + 1],
                  "ref_values": refs, "unnormalised": unusable}
    return out


# ---------------------------------------------------------------------------
# Trend through the reliable bins
# ---------------------------------------------------------------------------
#: Form key → label.  Every form is fitted **in log life**, which is what keeps the curve
#: positive everywhere and matches how every frequency layer in this workflow is
#: parameterised (``ya_freq_empirical.theta_freq_poly``).  A plain polynomial in days can
#: cross zero just outside the data and would then be predicting a negative life.
TREND_FORMS = {
    "poly1": "лог-линейный",
    "poly2": "парабола в log (n=3)",
    "poly3": "кубика в log (n=4)",
    "rational": "рациональная (насыщающаяся)",
}
#: Frequency the trend's ``u = f − ref`` is centred on.  50 Hz, as everywhere else.
TREND_REF = 50.0


@dataclass(frozen=True)
class TrendFit:
    """A weighted fit through the bins that carry enough failures to be worth fitting."""
    form: str
    stat: str
    params: tuple
    weighted_r2: float
    rmse_log: float
    n_bins: int
    n_runs: int
    support: tuple
    opt_hz: float
    opt_value: float
    ref: float = TREND_REF
    scale: str = "linear"
    #: True when the maximum sits at an end of the support — i.e. the curve is monotone over
    #: the fitted range and there is no interior optimum.  For the rate axes this is the
    #: normal case (more throughput, shorter life) and "оптимум" would over-claim.
    opt_at_edge: bool = False

    def _u(self, f) -> np.ndarray:
        v = np.atleast_1d(np.asarray(f, float))
        if self.scale == "log":
            return np.log10(np.maximum(v, 1e-12)) - np.log10(self.ref)
        return v - self.ref

    def predict(self, f) -> np.ndarray:
        u = self._u(f)
        p = np.asarray(self.params, float)
        if self.form == "rational":
            a, a1, a2, b2 = p
            return np.exp(a) * (1.0 + a1 * u + a2 * u ** 2) / (1.0 + b2 * u ** 2)
        return np.exp(sum(c * u ** k for k, c in enumerate(p)))

    def __str__(self) -> str:
        where = "максимум на краю" if self.opt_at_edge else "оптимум"
        return (f"{TREND_FORMS[self.form]} по {self.n_bins} бинам "
                f"({self.n_runs} отказов): R²={self.weighted_r2:.2f}, "
                f"{where} {self.opt_hz:.4g}")


def _trend_weights(d: pd.DataFrame, count_col: str = "n") -> np.ndarray:
    """Inverse-variance weights for a bin's statistic on the **log** scale.

    ``var(log(stat)) ≈ sigma_log² / n`` up to a constant that is the same for every bin of a
    given statistic, so ``w = n / sigma_log²`` is the right relative weight and the constant
    cancels.  Weighting by ``n`` alone would over-trust a bin that is both large and wildly
    dispersed — exactly the 59–70 Hz mixture bins.  Where ``sigma_log`` is unavailable (the
    KM table has no per-run spread) it falls back to the count.
    """
    n = d[count_col].to_numpy(float)
    s = (d["sigma_log"].to_numpy(float) if "sigma_log" in d.columns
         else np.full(len(d), np.nan))
    w = np.where(np.isfinite(s) & (s > 0), n / np.maximum(s, 1e-6) ** 2, n)
    return w / w.mean() if w.sum() > 0 else np.ones(len(d))


def fit_trend(tab: pd.DataFrame, *, stat: str = "median", form: str = "poly2",
              min_n: int = TARGET_BIN_N, ref: float = TREND_REF,
              count_col: str = "n", scale: str = "linear") -> TrendFit | None:
    """Weighted trend through the bins holding at least ``min_n`` failures.

    Thin bins are **excluded, not down-weighted**.  A 3-run bin's median is not a noisy
    measurement of the curve — it is barely a measurement at all, and letting it in with a
    small weight still lets it drag a cubic's tail. ``min_n`` is the same knob that flags a
    bin thin in the chart, so what you see hollow is what the trend ignored.

    The fit is on **log of the statistic**, weighted by ``n / sigma_log²``
    (:func:`_trend_weights`).  Returns ``None`` when fewer bins survive than the form has
    parameters — an exactly-determined "fit" would report R² = 1 and mean nothing.

    ``support`` is the covariate range actually fitted.  Nothing here is valid outside it,
    and a cubic in particular runs away fast; the caller should not draw past it.

    ``scale`` must match the axis the bins were built on.  A parabola in **raw** Ql, on an
    axis running 5 → 2000, is not a curve through the data — it is a straight line with a
    rounding error, and it puts its vertex wherever the last decade happens to sit.  On a
    log axis the fit uses ``u = log10(x) − log10(ref)``, which is the same coordinate the
    workflow's own rate layers are estimated in, so the shape is comparable with them.
    """
    if form not in TREND_FORMS:
        raise ValueError(f"form must be one of {tuple(TREND_FORMS)}, got {form!r}")
    d = tab[(tab[count_col] >= int(min_n)) & tab[stat].notna()
            & (tab[stat] > 0)].sort_values("f_mid")
    n_par = {"poly1": 2, "poly2": 3, "poly3": 4, "rational": 4}[form]
    if len(d) <= n_par:
        return None

    xm = d["f_mid"].to_numpy(float)
    u = (np.log10(np.maximum(xm, 1e-12)) - np.log10(ref)) if scale == "log" \
        else xm - float(ref)
    y = np.log(d[stat].to_numpy(float))
    w = _trend_weights(d, count_col)
    sw = np.sqrt(w)

    deg = {"poly1": 1, "poly2": 2, "poly3": 3}.get(form)
    if deg is not None:
        X = np.column_stack([u ** k for k in range(deg + 1)])
        params, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        pred = X @ params
    else:
        from scipy.optimize import least_squares

        # Start from the parabola: the rational is a saturating re-parameterisation of the
        # same shape, so the quadratic fit is both a good init and the honest fallback.
        X = np.column_stack([np.ones_like(u), u, u * u])
        q, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        p0 = np.array([q[0], q[1], q[2], 1e-3])

        def resid(p):
            a, a1, a2, b2 = p
            num = 1.0 + a1 * u + a2 * u ** 2
            # the numerator must stay positive on the fitted range for the log to exist;
            # a floor rather than a hard constraint keeps the solver on a smooth surface
            return sw * (a + np.log(np.maximum(num, 1e-9))
                         - np.log1p(b2 * u ** 2) - y)

        try:
            sol = least_squares(resid, p0, bounds=([-np.inf, -np.inf, -np.inf, 0.0],
                                                   [np.inf, np.inf, np.inf, 1.0]),
                                max_nfev=4000)
            params = sol.x
            pred = y - resid(params) / np.where(sw > 0, sw, 1.0)
        except Exception:                                        # noqa: BLE001
            return fit_trend(tab, stat=stat, form="poly2", min_n=min_n, ref=ref,
                             count_col=count_col, scale=scale)

    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    lo, hi = float(d["f_lo"].min()), float(d["f_hi"].max())
    fit = TrendFit(form=form, stat=stat, params=tuple(map(float, params)),
                   weighted_r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                   rmse_log=float(np.sqrt(np.average((y - pred) ** 2, weights=w))),
                   n_bins=int(len(d)), n_runs=int(d[count_col].sum()), support=(lo, hi),
                   opt_hz=np.nan, opt_value=np.nan, ref=float(ref), scale=scale)
    grid = np.geomspace(max(lo, 1e-9), hi, 601) if scale == "log" else np.linspace(lo, hi, 601)
    vals = fit.predict(grid)
    i = int(np.argmax(vals))
    return dc_replace(fit, opt_hz=float(grid[i]), opt_value=float(vals[i]),
                      opt_at_edge=bool(i <= 2 or i >= len(grid) - 3))


# ---------------------------------------------------------------------------
# Is the shape real?  One test, run per statistic.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlatnessTest:
    stat: str
    spread: float          # observed n-weighted sd of the per-bin statistic
    p_value: float
    n_perm: int
    n_bins: int

    def __str__(self) -> str:
        return f"{self.stat}: разброс {self.spread:.1f} д, p = {self.p_value:.3f}"


def permutation_flatness(x, y, edges, *, stat: str = "median", n_perm: int = 1000,
                         seed: int = 20260728, min_n: int = MIN_BIN_N,
                         trim: float = 0.1) -> FlatnessTest:
    """Does the binned curve depart from flat by more than chance?

    The frequency labels are shuffled against the lives, the curve is recomputed, and the
    n-weighted spread of the per-bin statistic is compared with the observed one.  Because the
    binning is held fixed and only the labels move, this is a test of *shape*, not of the
    edges — and re-cutting the bins re-runs it, which is exactly how a shape claim should be
    stress-tested.

    Run it per statistic.  "The mean says one thing and the median another" is only a finding
    if at least one of them beats this test; on thin bins, usually neither does.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    e = sanitize_edges(edges, min_width=0.0)
    idx = bin_index(x, e)

    def spread(ix: np.ndarray) -> float:
        vals, wts = [], []
        for i in range(len(e) - 1):
            yy = y[ix == i]
            if yy.size < min_n:
                continue
            vals.append(_centres(yy, trim)[stat])
            wts.append(yy.size)
        if len(vals) < 2:
            return np.nan
        v, w = np.asarray(vals, float), np.asarray(wts, float)
        mu = np.average(v, weights=w)
        return float(np.sqrt(np.average((v - mu) ** 2, weights=w)))

    obs = spread(idx)
    rng = np.random.default_rng(seed)
    null = np.array([spread(idx[rng.permutation(idx.size)]) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    # +1 in both parts: the observed configuration is itself one of the possible draws, so a
    # p-value can never be exactly 0 — with 1000 permutations the floor is 1/1001.
    p = float((np.sum(null >= obs) + 1) / (null.size + 1)) if null.size else np.nan
    return FlatnessTest(stat=stat, spread=obs, p_value=p, n_perm=int(null.size),
                        n_bins=int(sum(bin_counts(x, e) >= min_n)))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save(edges, fact: pd.DataFrame | None = None, km: pd.DataFrame | None = None, *,
         label: str = "custom", run_date: str | None = None) -> Path:
    """Write the current binning and its tables to ``results/<SLUG>/<date>/``.

    The edge list goes out as its own one-row csv so a binning an operator liked can be
    pasted back into the app (or a script) verbatim — a figure alone would not be reproducible.
    """
    out = results_dir(SLUG, run_date)
    e = sanitize_edges(edges, min_width=0.0)
    pd.DataFrame([{"label": label, "n_bins": len(e) - 1,
                   "edges": "|".join(f"{v:g}" for v in e)}]).to_csv(
        out / "tables" / f"edges_{label}.csv", index=False, encoding="utf-8-sig")
    if fact is not None:
        fact.to_csv(out / "tables" / f"bins_{label}.csv", index=False, encoding="utf-8-sig")
    if km is not None:
        km.to_csv(out / "tables" / f"bins_km_{label}.csv", index=False, encoding="utf-8-sig")
    return out


def parse_edges(text: str) -> list[float]:
    """Read an edge list back from ``"30|32|34"`` or ``"30, 32, 34"``."""
    parts = [p for p in str(text).replace("|", ",").replace(";", ",").split(",") if p.strip()]
    return sanitize_edges([float(p) for p in parts])
