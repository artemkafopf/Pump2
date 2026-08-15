"""Двуязычный (RU + EN) отчёт о пересборке Свода.

Один HTML с переключателем языка: блоки размечены ``data-l="ru"`` / ``data-l="en"``,
переключатель прячет неактивный. Оглавление у каждого языка своё и ведёт на свои
якоря — общее оглавление на переключённом языке ведёт в никуда.

Отчёт отвечает ровно на приёмочные вопросы: сколько строк добавилось, у скольких
появился диагноз, у скольких изменилась группа причины — и всё это построчно, со
ссылкой на CSV, а не одним итогом.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from .diff import SvodDiff

#: Приёмочные пороги из брифа переноса.
#:
#: ⚠⚠ 3 929 пусков / 2 078 отказов считаются НЕ по строкам регистра (их 3 430), а по
#: популяции модели — ``v7_imex_nodes.population()``, которая склеивает регистр с
#: паспортом оборудования и несёт провенанс big_pre / big_post / svod. Спутать эти две
#: единицы наблюдения — классическая ошибка проекта, поэтому здесь считается только то,
#: что видно в самом регистре, а популяционные числа проверяются отдельным прогоном и
#: подставляются в ``BuildStats.population``.
ACCEPTANCE = {
    "population_runs": 3929,
    "population_failures": 2078,
    "node_coverage_2018": 0.84,
    "node_coverage_all": 0.66,
}

_CSS = """
:root{--bg:#fff;--fg:#1a1d21;--mut:#5b6570;--line:#e3e7ec;--accent:#1f6feb;
      --ok:#1a7f37;--warn:#9a6700;--bad:#b91c1c;--chip:#f2f5f9}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e9ee;--mut:#98a2b3;
      --line:#232a33;--accent:#5aa2ff;--ok:#4ac26b;--warn:#d4a72c;--bad:#f47174;--chip:#171c23}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:2rem 1.25rem 5rem}
h1{font-size:1.7rem;margin:.2em 0 .1em}
h2{font-size:1.22rem;margin:2.2rem 0 .6rem;padding-top:.6rem;border-top:1px solid var(--line)}
h3{font-size:1.02rem;margin:1.4rem 0 .4rem}
p,li{color:var(--fg)}
.sub{color:var(--mut);margin:.1rem 0 1.4rem}
.switch{position:sticky;top:0;z-index:5;background:var(--bg);padding:.7rem 0;
        border-bottom:1px solid var(--line);margin-bottom:1.2rem}
.switch button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--chip);
        color:var(--fg);border-radius:999px;padding:.3rem .95rem;margin-right:.4rem}
.switch button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
table{border-collapse:collapse;width:100%;margin:.6rem 0 1rem;font-size:.92rem}
th,td{border-bottom:1px solid var(--line);padding:.42rem .6rem;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600;white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.7rem;margin:1rem 0}
.card{border:1px solid var(--line);border-radius:10px;padding:.75rem .9rem;background:var(--chip)}
.card .k{color:var(--mut);font-size:.8rem;text-transform:uppercase;letter-spacing:.03em}
.card .v{font-size:1.5rem;font-variant-numeric:tabular-nums;margin-top:.15rem}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
code{background:var(--chip);padding:.1em .35em;border-radius:4px;font-size:.9em}
ul.toc{list-style:none;padding-left:0}
ul.toc li{margin:.15rem 0}
"""

_JS = """
(function(){
  function apply(lang){
    document.querySelectorAll('[data-l]').forEach(function(el){
      el.hidden = el.getAttribute('data-l') !== lang;
    });
    document.querySelectorAll('.switch button').forEach(function(b){
      b.setAttribute('aria-pressed', String(b.dataset.lang === lang));
    });
    document.documentElement.lang = lang;
  }
  document.querySelectorAll('.switch button').forEach(function(b){
    b.addEventListener('click', function(){ apply(b.dataset.lang); });
  });
  apply('ru');
})();
"""


@dataclass
class BuildStats:
    """Что получилось в сборке — то, что проверяет приёмка."""

    rows: int
    failures: int
    running: int
    node_coverage_all: float
    node_coverage_2018: float
    cause_groups: Mapping[str, int]
    disputed: int
    frac_runs: int
    frac_before_mount: int
    sources: Mapping[str, str]
    #: Числа с популяции модели (``v7_imex_nodes.population()``), если её прогоняли:
    #: ``population_runs``, ``population_failures``, ``node_coverage_all``,
    #: ``node_coverage_2018``. Регистр их измерить не может — у него другая единица
    #: наблюдения.
    population: Optional[Mapping[str, float]] = None


def collect_stats(register_path: Path | str, sources: Mapping[str, str]) -> BuildStats:
    """Measure the produced register — read back from disk, not from memory.

    Reading the file that was actually written is the only measurement that also
    proves the write succeeded; counting the in-memory frame would report numbers
    for a workbook that may never have reached the disk intact.
    """
    from ..config import SVOD_RUNNING_COLUMN, SVOD_SHEET_NAME
    from .causes import CAUSE_GROUP_COLUMN, DISPUTED_COLUMN, NODE_CATEGORY_COLUMN

    frame = pd.read_excel(register_path, sheet_name=SVOD_SHEET_NAME, header=0)
    mount = pd.to_datetime(frame.get("Дата монтажа"), errors="coerce")
    flag = pd.to_numeric(frame.get("Флаг отказа"), errors="coerce")
    running = pd.to_numeric(frame.get(SVOD_RUNNING_COLUMN), errors="coerce").fillna(0)
    categorized = frame[NODE_CATEGORY_COLUMN].notna() if NODE_CATEGORY_COLUMN in frame else pd.Series(False, index=frame.index)

    failures_mask = flag == 1
    cohort_2018 = mount >= pd.Timestamp("2018-01-01")

    def _coverage(mask: pd.Series) -> float:
        subset = failures_mask & mask
        return float(categorized[subset].mean()) if subset.any() else 0.0

    groups = (
        frame[CAUSE_GROUP_COLUMN].value_counts().to_dict()
        if CAUSE_GROUP_COLUMN in frame else {}
    )
    return BuildStats(
        rows=len(frame),
        failures=int(failures_mask.sum()),
        running=int(running.sum()),
        node_coverage_all=_coverage(pd.Series(True, index=frame.index)),
        node_coverage_2018=_coverage(cohort_2018.fillna(False)),
        cause_groups=groups,
        disputed=int(pd.to_numeric(frame.get(DISPUTED_COLUMN), errors="coerce").fillna(0).sum()),
        frac_runs=int(pd.to_numeric(frame.get("ГРП"), errors="coerce").fillna(0).sum()),
        frac_before_mount=int(pd.to_numeric(frame.get("ГРП до монтажа"), errors="coerce").fillna(0).sum()),
        sources=dict(sources),
    )


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _table(frame: pd.DataFrame, limit: int = 25) -> str:
    if frame is None or frame.empty:
        return '<p class="sub">—</p>'
    shown = frame.head(limit)
    head = "".join(f"<th>{_esc(c)}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False)
    )
    more = (
        f'<p class="sub">показаны первые {limit} из {len(frame)} — полный список в CSV</p>'
        if len(frame) > limit else ""
    )
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}'


def _cards(items: Sequence[tuple[str, str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v {cls}">{_esc(v)}</div></div>'
        for k, v, cls in items
    ) + "</div>"


def _verdict(actual: float, expected: float, *, at_least: bool = False) -> str:
    if at_least:
        return "ok" if actual >= expected else "bad"
    return "ok" if actual == expected else "warn"


def _acceptance_rows(stats: BuildStats) -> list[tuple[str, str, str, str, str]]:
    """(критерий RU, критерий EN, ожидание, факт, класс).

    Регистровые строки считаются здесь; популяционные — только если их подставили
    отдельным прогоном (``BuildStats.population``), иначе они честно помечаются
    «не проверено», а не подменяются похожим числом из регистра.
    """
    rows = [
        ("Строк в регистре (одна на пуск)", "Rows in the register (one per run)",
         "—", str(stats.rows), ""),
        ("Отказов в регистре (Флаг отказа = 1)", "Failures in the register (flag = 1)",
         "—", str(stats.failures), ""),
        ("Работающих (ПУСТАЯ дата окончания)", "Running (EMPTY stop date)",
         "—", str(stats.running), ""),
    ]
    population = stats.population or {}
    for key, ru, en in (
        ("population_runs", "Пусков в популяции модели", "Runs in the model population"),
        ("population_failures", "Отказов в популяции модели", "Failures in the model population"),
    ):
        expected = ACCEPTANCE[key]
        actual = population.get(key)
        rows.append((
            ru, en, str(expected),
            "не проверено" if actual is None else str(actual),
            "" if actual is None else _verdict(actual, expected),
        ))
    for key, ru, en in (
        ("node_coverage_2018", "Покрытие категорией узла, монтажи 2018+", "Node-category coverage, 2018+ mounts"),
        ("node_coverage_all", "Покрытие категорией узла, вся популяция", "Node-category coverage, full population"),
    ):
        expected = ACCEPTANCE[key]
        actual = population.get(key)
        rows.append((
            ru, en, f"≥ {expected:.0%}",
            "не проверено" if actual is None else f"{actual:.1%}",
            "" if actual is None else _verdict(actual, expected, at_least=True),
        ))
    return rows


def _acceptance_table(stats: BuildStats, lang: str) -> str:
    criterion = "Критерий" if lang == "ru" else "Criterion"
    expected = "Ожидание" if lang == "ru" else "Expected"
    actual = "Факт" if lang == "ru" else "Actual"
    rows = "".join(
        f"<tr><td>{_esc(ru if lang == 'ru' else en)}</td>"
        f"<td class='n'>{_esc(exp)}</td>"
        f"<td class='n {cls}'>{_esc(got)}</td></tr>"
        for ru, en, exp, got, cls in _acceptance_rows(stats)
    )
    return (
        f"<div class='wrap'><table><thead><tr><th>{criterion}</th>"
        f"<th class='n'>{expected}</th><th class='n'>{actual}</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _summary_table(summary: Mapping[str, object], labels: Mapping[str, str], lang: str) -> str:
    metric = "Показатель" if lang == "ru" else "Metric"
    value = "Значение" if lang == "ru" else "Value"
    rows = "".join(
        f"<tr><td>{_esc(labels.get(key, key) if lang == 'en' else key)}</td>"
        f"<td class='n'>{_esc(val)}</td></tr>"
        for key, val in summary.items()
    )
    return (
        f"<div class='wrap'><table><thead><tr><th>{metric}</th>"
        f"<th class='n'>{value}</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


SUMMARY_EN = {
    "строк в новом регистре": "rows in the new register",
    "строк в прежнем регистре": "rows in the previous register",
    "добавилось пусков": "runs added",
    "исчезло пусков": "runs removed",
    "совпало по ключу": "matched on the key",
    "появился диагноз": "diagnosis gained",
    "пропал диагноз": "diagnosis lost",
    "изменилась группа причины": "cause group changed",
    "изменений в отслеживаемых полях": "tracked-field changes",
}

SECTIONS = [
    ("sources", "Источники и что изменилось", "Sources and what changed"),
    ("acceptance", "Приёмка", "Acceptance"),
    ("causes", "Разметка причин", "Cause taxonomy"),
    ("diff", "Диф против прежнего Свода", "Diff against the previous register"),
    ("fixes", "Исправленные дефекты", "Defects fixed"),
]


def _toc(lang: str) -> str:
    items = "".join(
        f'<li><a href="#{key}-{lang}">{_esc(ru if lang == "ru" else en)}</a></li>'
        for key, ru, en in SECTIONS
    )
    return f'<ul class="toc">{items}</ul>'


def render_report(
    stats: BuildStats,
    diff: Optional[SvodDiff],
    fixes: Sequence[tuple[str, str]],
    *,
    built_at: str,
) -> str:
    """Return the complete bilingual HTML document."""
    cards = _cards([
        ("Пусков / Runs", f"{stats.rows}", ""),
        ("Отказов / Failures", f"{stats.failures}", ""),
        ("Работает / Running", f"{stats.running}", ""),
        ("Категория узла / Node category", f"{stats.node_coverage_all:.0%}", ""),
        ("Спорная зона / Disputed", f"{stats.disputed}", ""),
        ("ГРП до монтажа / Frac before mount", f"{stats.frac_before_mount}", ""),
    ])

    groups = pd.DataFrame(
        [{"группа": k, "строк": v} for k, v in stats.cause_groups.items()]
    )
    sources = pd.DataFrame([{"источник": k, "путь": v} for k, v in stats.sources.items()])

    def fixes_list(lang: str) -> str:
        items = []
        for entry in fixes:
            title, description = (entry[0], entry[1]) if lang == "ru" else (entry[2], entry[3])
            items.append(f"<li><b>{_esc(title)}</b> — {_esc(description)}</li>")
        return "".join(items)

    def block(lang: str) -> str:
        ru = lang == "ru"
        diff_body = (
            _summary_table(diff.summary, SUMMARY_EN, lang)
            + ("<h3>Появился диагноз</h3>" if ru else "<h3>Diagnosis gained</h3>")
            + _table(diff.gained_diagnosis)
            + ("<h3>Изменилась группа причины</h3>" if ru else "<h3>Cause group changed</h3>")
            + _table(diff.changed_group)
            + ("<h3>Добавленные пуски</h3>" if ru else "<h3>Runs added</h3>")
            + _table(diff.added)
            if diff is not None
            else ('<p class="sub">Диф не строился.</p>' if ru else '<p class="sub">No diff was produced.</p>')
        )
        return f"""
<div data-l="{lang}">
  <h1>{'Пересборка Свода ЭЦН' if ru else 'ESP register rebuild'}</h1>
  <p class="sub">{'Собрано' if ru else 'Built'} {_esc(built_at)} · <code>analysis.ingest.svod</code></p>
  {cards}
  {_toc(lang)}

  <h2 id="sources-{lang}">{'Источники и что изменилось' if ru else 'Sources and what changed'}</h2>
  <p>{
    'Регистр собирается из сырых выгрузок в этом же репозитории — обработка входа и '
    'расчёт больше не расходятся. Все пути идут через <code>analysis.paths</code>.'
    if ru else
    'The register is assembled from the raw exports inside this repository, so input '
    'processing and modelling can no longer drift apart. Every path resolves through '
    '<code>analysis.paths</code>.'
  }</p>
  {_table(sources, limit=20)}

  <h2 id="acceptance-{lang}">{'Приёмка' if ru else 'Acceptance'}</h2>
  {_acceptance_table(stats, lang)}
  <p class="sub">{
    '⚠⚠ Единица наблюдения разная. Строк в регистре 3 430; приёмочные 3 929 / 2 078 — '
    'это ПОПУЛЯЦИЯ МОДЕЛИ (<code>v7_imex_nodes.population()</code>), которая склеивает '
    'регистр с паспортом оборудования. Сравнивать их напрямую нельзя.'
    if ru else
    '⚠⚠ Two different units of observation. The register holds 3 430 rows; the 3 929 / '
    '2 078 acceptance figures come from the MODEL POPULATION '
    '(<code>v7_imex_nodes.population()</code>), which joins the register to the '
    'equipment passport. They are not directly comparable.'
  }</p>
  <p class="sub">{
    '⚠⚠ «Работает» считается по ПУСТОЙ дате окончания. Флаг отказа = 0 не означает '
    '«работает»: туда же попадают подъёмы по ГТМ и ППР — их 1 703.'
    if ru else
    '⚠⚠ "Running" is decided by an EMPTY stop date. A zero failure flag does not mean '
    '"running": planned ГТМ/ППР pulls carry a zero too — 1 703 of them.'
  }</p>

  <h2 id="causes-{lang}">{'Разметка причин' if ru else 'Cause taxonomy'}</h2>
  <p>{
    'Критерий заказчика: в эксплуатацию уходит всё, на что нельзя повлиять на начальном '
    'этапе. «Механическое повреждение кабеля» перенесено в монтаж; «необеспечен приток» и '
    '«влияние газа» — в эксплуатацию. Граница «брак &lt;узел&gt;» против «брак монтажа» '
    'коммерческая, поэтому вместо жёсткого отнесения стоит флаг <code>спорная_зона</code>.'
    if ru else
    "The customer's criterion: anything that cannot be influenced up front counts as "
    'operation. "Mechanical cable damage" moved to assembly; "no inflow" and "gas influence" '
    'moved to operation. The "component defect" vs "installation defect" boundary is '
    'commercial, so instead of a hard assignment those rows carry a <code>спорная_зона</code> flag.'
  }</p>
  {_table(groups, limit=10)}

  <h2 id="diff-{lang}">{'Диф против прежнего Свода' if ru else 'Diff against the previous register'}</h2>
  <p class="sub">{
    '⚠⚠ Стыковка только по (скважина, дата монтажа). Позиционный <code>run</code> у разных '
    'кадров разный.'
    if ru else
    '⚠⚠ Joined on (well, mount date) only. The positional <code>run</code> index differs '
    'between frames.'
  }</p>
  {diff_body}

  <h2 id="fixes-{lang}">{'Исправленные дефекты' if ru else 'Defects fixed'}</h2>
  <ul>{fixes_list(lang)}</ul>
</div>"""

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Пересборка Свода ЭЦН / ESP register rebuild</title>
<style>{_CSS}</style></head>
<body><main>
<div class="switch"><button data-lang="ru" aria-pressed="true">Русский</button><button data-lang="en" aria-pressed="false">English</button></div>
{block('ru')}
{block('en')}
</main><script>{_JS}</script></body></html>"""


def write_report(path: Path | str, *args, **kwargs) -> Path:
    """Render and write the bilingual report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(*args, **kwargs), encoding="utf-8")
    return path


#: Дефекты, найденные и исправленные при переносе: (заголовок RU, текст RU, EN, EN).
DEFECTS_FIXED = [
    (
        "Сборка шла не на дату ПДК",
        "Основное правило: регистр строится НА ПОСЛЕДНЮЮ ДАТУ ПДК. Отказы приходят "
        "только из ПДК, живые пуски — из паспорта оборудования, который обновляется "
        "отдельно; правее последней записи ПДК экспозиция копится, а событие записать "
        "нечем. ПДК кончается 2026-07-12, паспорт — 2026-06-05, а цензурирование шло по "
        "сегодняшнему дню. В паспорте колонки ННО нет вовсе, поэтому все 556 живых "
        "пусков брали календарный возраст на сегодня и несли 18 904 суток бесплатной "
        "выживаемости (6 % их наработки, медиана 385 → 351) — смещение бьёт в раннюю "
        "полосу. Теперь: пуск, смонтированный после горизонта, выбрасывается; живой "
        "цензурируется горизонтом; пуск, закрывшийся за горизонтом, остаётся "
        "цензурированным, а не исчезает из регистра вместе со всей прожитой жизнью. "
        "Переопределяется --as-of.",
        "The build was not cut at the PDK date",
        "The governing rule: the register is built AS OF THE LAST PDK DATE. Failures "
        "come only from PDK, live runs from the equipment passport which is refreshed "
        "separately; past PDK's last record exposure accrues but no event can be "
        "recorded. PDK ends 2026-07-12, the passport 2026-06-05, yet censoring used "
        "today's date. The passport carries no runtime column at all, so all 556 live "
        "runs took calendar age to today and carried 18 904 days of free survival (6 % "
        "of their runtime, median 385 → 351) — a bias aimed straight at the early band. "
        "Now: a run mounted after the horizon is dropped; a live run is censored at the "
        "horizon; a run closing past the horizon stays censored instead of vanishing "
        "from the register with its whole observed life. Override with --as-of.",
    ),
    (
        "Многолетние пуски схлопывались в отказы на 0–2 суток",
        "Предпочтение подтверждённому отказу при схлопывании дублей ПДК было "
        "ФИЛЬТРОМ: если в группе (скважина, дата монтажа) нашёлся отказ, все "
        "остальные строки выбрасывались. Пока «Прочие» с заполненным узлом считались "
        "неоднозначными, это почти не срабатывало; портированная правка сделала их "
        "подтверждённым отказом — и происшествие в день спуска стало вытеснять "
        "плановый подъём годы спустя. Ya_622, монтаж 2019-07-30: пуск на 1359 суток "
        "превращался в отказ на 0 суток. Направление ошибки худшее из возможных — она "
        "фабрикует события в полосе 0–5 суток, где решается вопрос о детской "
        "смертности. Предпочтение отказу стало тай-брейком среди строк с ОДНОЙ датой "
        "остановки; дата решает первой.",
        "Multi-year runs collapsed into 0–2 day failures",
        "Preferring a confirmed failure when collapsing duplicate PDK rows was a "
        "FILTER: if any row in a (well, mount date) group was a failure, every other "
        "row was discarded. While «Прочие» with a diagnosed component counted as "
        "ambiguous this rarely fired; the ported fix made those rows confirmed "
        "failures, and a run-in-day incident began displacing a planned pull years "
        "later. Ya_622, mounted 2019-07-30: a 1359-day run became a 0-day failure. The "
        "error runs in the worst possible direction — it manufactures events in the "
        "0–5 day band, where the infant-mortality question is decided. The failure "
        "preference is now a tie-break among rows sharing ONE stop date; the date "
        "decides first.",
    ),
    (
        "Мёртвые дубликаты процессоров",
        "В db_builder модуль enrich нёс вторую копию пяти процессоров (ПДК, Big, "
        "ТехРежим, ОПЗ, лаборатория) со старой логикой. Тесты импортировали именно "
        "её, то есть проверяли код, который в проде не выполнялся. Дубликаты удалены, "
        "тесты переставлены на живые реализации.",
        "Dead duplicates of five processors",
        "In db_builder the enrich module carried a second copy of five processors "
        "(PDK, Big, TechRegime, OPZ, lab) running older logic. The tests imported that "
        "copy — so they were checking code that never executed in production. The "
        "duplicates are gone and the tests now point at the live implementations.",
    ),
    (
        "Дебит жидкости молча пропадал из телеметрии",
        "Выгрузка от 2026-08-15 добавила блок ОЗНА, и «Дебит жидкости (ОЗНА)» стал "
        "вторым кандидатом на подстроку «дебитжидкости». Резолвер принимает только "
        "единственное совпадение — Qliq_m3d вышел пустым во всех 1 414 192 строках "
        "(в июньской сборке было заполнено 1 168 492). Добавлен точный алиас на ШАХ; "
        "ничья при разрешении имени теперь печатается вслух.",
        "Liquid rate vanished from telemetry, silently",
        "The 2026-08-15 export added an ОЗНА block, making «Дебит жидкости (ОЗНА)» a "
        "second candidate for the substring «дебитжидкости». The resolver accepts only "
        "a unique match, so Qliq_m3d came out empty across all 1 414 192 rows (the "
        "June build had 1 168 492 filled). An exact alias now pins the ШАХ column, and "
        "a resolution tie is reported out loud.",
    ),
    (
        "Пластовое давление под именем давления насыщения",
        "«Рпл» стоял в алиасах Pbubble_atm. На телеметрии дефект был латентным "
        "(колонка пуста во всех строках), но на источнике ТехРежима пластовое поехало "
        "бы как давление насыщения вместе со всеми производными pressure_ratio_*. "
        "Разведено на Pbubble_atm и P_reservoir_atm.",
        "Reservoir pressure carried under the bubble-point name",
        "«Рпл» sat among the Pbubble_atm aliases. On telemetry the defect was latent "
        "(the column is empty in every row), but on a TechRegime source reservoir "
        "pressure would have travelled as bubble-point pressure together with every "
        "pressure_ratio_* derived from it. Split into Pbubble_atm and P_reservoir_atm.",
    ),
    (
        "Сборка телеметрии падала на нефизичном числе",
        "«Газовый фактор (ОЗНА), м3/т» содержит 13 значений до 2.3·10²⁴ — счётчик "
        "делит на околонулевой дебит. SQLite INTEGER 64-битный, вся загрузка падала. "
        "Значения вне диапазона сохраняются текстом без потери цифр.",
        "The telemetry build died on a non-physical number",
        "«Газовый фактор (ОЗНА), м3/т» holds 13 values up to 2.3e24 — the meter "
        "dividing by a near-zero rate. SQLite's INTEGER is 64-bit, so the whole load "
        "aborted. Out-of-range values are now stored as exact decimal text.",
    ),
    (
        "ТехРежим терял 15 колонок из 16 на legacy-ветке",
        "На ветке месячных папок Excel (без sqlite) zero-safe колонки пропускались в "
        "пользу усреднения по окну истории, которого у legacy-кадра нет — оставался "
        "заполненным только «Тип ствола скв». В проде ветка не выбиралась, потому что "
        "рядом лежит techregime.sqlite.",
        "TechRegime lost 15 of 16 columns on the legacy path",
        "On the monthly-Excel-folder path (no sqlite beside the export) the zero-safe "
        "columns were skipped in favour of a history-window average that a legacy "
        "frame has no dates for — only «Тип ствола скв» stayed populated. Production "
        "never took that path because techregime.sqlite sits next to the export.",
    ),
    (
        "Excel-серийник в колонке даты",
        "Векторный разбор дат читал голое число как наносекунды от эпохи (44000 → "
        "1970 г.). Числовые ячейки направлены в проверяющий диапазон parse_date.",
        "Excel serial in a date column",
        "Vectorized date parsing read a bare number as nanoseconds since the epoch "
        "(44000 → 1970). Numeric cells are routed to the range-checking parse_date.",
    ),
    (
        "Потеря лога при падении сборки",
        "stdout был блочно буферизован, и упавшая сборка не оставляла ни строки "
        "прогресса — только трейсбек. Включена построчная буферизация.",
        "The log was lost when a build died",
        "stdout was block-buffered, so a build that failed left no progress lines at "
        "all — only the traceback. Line buffering is now enabled.",
    ),
    (
        "Свободный текст ПДК и «Прочие» с диагнозом",
        "Портированы правки из db_builder: «Примечание» и «Осложнения при ТКРС» "
        "доезжают до Свода; «Прочие» с заполненным узлом больше не вылетает как "
        "неоднозначное; при равных дате и наработке выигрывает строка С диагнозом.",
        "PDK free text and «Прочие» rows that carry a diagnosis",
        "Ported from db_builder: «Примечание» and «Осложнения при ТКРС» now reach the "
        "register; «Прочие» with a diagnosed component no longer drops out as "
        "ambiguous; on equal date and runtime the diagnosed row wins.",
    ),
]

__all__ = [
    "ACCEPTANCE",
    "DEFECTS_FIXED",
    "BuildStats",
    "collect_stats",
    "render_report",
    "write_report",
]
