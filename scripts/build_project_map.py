#!/usr/bin/env python
"""Generate ``docs/PROJECT_MAP.md`` — an annotated, always-current map of the repo.

Why this exists
---------------
Claude Code loads ``CLAUDE.md`` into context every session, and CLAUDE.md pulls
this map in with ``@docs/PROJECT_MAP.md``.  The tree is derived from
``git ls-files`` and the per-module notes are read out of the modules' own
docstrings, so the map cannot drift from the code the way a hand-written tree
does.  Curated notes below cover only what no docstring can express — which
directories are legacy, and where new code is supposed to go.

Output is deterministic: no timestamp, no ordering by mtime.  Re-running when
nothing structural changed rewrites an identical file, so ``git status`` stays
quiet and ``--check`` is meaningful.

Usage
-----
    python scripts/build_project_map.py             # write docs/PROJECT_MAP.md
    python scripts/build_project_map.py --stdout    # print instead of writing
    python scripts/build_project_map.py --check     # exit 1 if the file is stale

A ``SessionStart`` hook in ``.claude/settings.json`` runs this at the start of
every Claude Code session so the imported map is current.

This script deliberately does not use ``analysis.paths.results_dir()``: the
output is a committed document, not an analysis result.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "PROJECT_MAP.md"

# Directories shown in the tree.  Depth is counted in path separators, so
# ``backend/analysis/workflows`` is depth 3.
MAX_TREE_DEPTH = 3

# Roots allowed one extra level, because their children are the unit of work.
# Without this the workflow packages — the things most analyses live in — would
# be collapsed into a single ``workflows/ [57]`` line.
DEEP_ROOTS: tuple[str, ...] = ("backend/analysis/workflows", "backend/analysis/models")

# Longest docstring summary kept in the map, in characters.
SUMMARY_WIDTH = 88

# --------------------------------------------------------------------------
# Curated tables — the only hand-maintained part.  Everything else is derived.
# --------------------------------------------------------------------------

# Notes for directories.  A missing entry is fine: the directory still appears
# in the tree with its file count, just without prose.  New unannotated
# directories are listed at the bottom of the map so drift is visible.
DIR_NOTES: dict[str, str] = {
    ".claude": "Claude Code project config — settings.json, skills/",
    ".vscode": "editor config",
    "agents": "agent instructions and per-analysis briefs",
    "agents/analyses": "one brief per analysis line — handoffs, prompts, plans",
    "analysis": "LEGACY — migrating to backend/analysis/workflows/; do not add files here",
    "analysis/vt_failure": "LEGACY — moved to backend/analysis/workflows/vt_failure/",
    "backend": "FastAPI service, the analysis library, and the test suite",
    "backend/analysis": "the reusable computation library — all new heavy logic lands here",
    "backend/analysis/common": "plotting, Cox reporting, shared modelling config",
    "backend/analysis/data": "loaders, label hygiene, failure-mode classification, run covariates",
    "backend/analysis/features": "derived features and stress transforms",
    "backend/analysis/models": "estimators — model code, no I/O",
    "backend/analysis/models/survival": "the survival estimators: Weibull, Cox variants, CIF, landmark",
    "backend/analysis/workflows": "end-to-end analyses; each package exposes a single entrypoint",
    "backend/analysis/workflows/chemistry_cox": "chemistry Cox block",
    "backend/analysis/workflows/esp_survival": "ESP survival stack — mixture fits, RUL, VBA bundle export",
    "backend/analysis/workflows/production_risk": "production-risk forecast feeding the ПП plan",
    "backend/analysis/workflows/vt_failure": "Vt failure analysis — the phased survival study",
    "backend/app": "FastAPI service",
    "backend/app/api": "HTTP routes",
    "backend/app/core": "service config",
    "backend/app/db": "SQLAlchemy models and session",
    "backend/app/schemas": "pydantic request/response schemas",
    "backend/app/services": "service logic — analysis, reporting, repair forecast, LLM client",
    "backend/tests": "pytest suite — run with `cd backend && python -m pytest tests/ -v`",
    "docs": "committed documentation",
    "docs/decisions": "decision records",
    "docs/methodology": "method write-ups",
    "docs/notes": "working notes and findings",
    "docs/presentations": "report and deck sources",
    "frontend": "Vue 3 client for the FastAPI service",
    "frontend/src": "app source",
    "frontend/src/assets": "logos and images",
    "frontend/src/components": "single-file Vue components — one per panel/module",
    "frontend/src/services": "API client",
    "frontend/src/styles": "global styles",
    "scripts": "thin CLI wrappers — heavy logic belongs in backend/analysis/",
    "scripts/deploy": "launcher and exe builds",
    "scripts/features": "feature-table builds",
    "scripts/ingest": "raw to interim ingestion",
    "scripts/marts": "mart builds",
    "scripts/processing": "interim processing",
    "scripts/quality": "data-quality checks",
    "scripts/run": "workflow entrypoints — phase_<block>_<step>.py mirrors the analysis phases",
    "streamlit_apps": "interactive explorers — Свод panels and model fitters",
    "vba": "Excel VBA modules for the shipped calculator (ПикПолка / NPV_УВЧ)",
}

# Directories marked as off-limits for new code, rendered with a warning marker.
LEGACY_DIRS: set[str] = {"analysis", "analysis/vt_failure"}

# Packages expanded to per-module docstring lines.  Keep this list short: every
# entry costs context in every session.  Packages whose filenames already carry
# the ordering (phase1.py, phase2.py, ...) are covered by GROUPED_NOTES instead.
DETAIL_DIRS: list[str] = [
    "backend/analysis/models/survival",
    "backend/analysis/workflows/production_risk",
    "backend/analysis/data",
    "backend/analysis/features",
    "backend/analysis/common",
]

# Packages summarised in one line rather than expanded file by file.
GROUPED_NOTES: dict[str, str] = {
    "backend/analysis/workflows/vt_failure": (
        "phase1..phase11 (+8b/8c/8d) run in order from `run.py`; `data.py` merges mart + "
        "failure categories, `config.py` holds the shared constants"
    ),
    "backend/analysis/workflows/esp_survival": (
        "phase0..phase6 run in order from `run.py`; `data_mart.py` builds the frame, "
        "`vba_bundle.py` exports the Excel model bundle"
    ),
    "backend/analysis/workflows/chemistry_cox": "single step — `phase_chem.py`",
}

# Untracked-but-real directories.  These are gitignored, so `git ls-files`
# cannot see them, but an agent still needs to know they exist and what for.
UNTRACKED_DIRS: list[tuple[str, str]] = [
    ("results/", "canonical output root — every new analysis writes here via `results_dir(slug)`"),
    ("data/", "data layer: raw / interim / marts / warehouse, addressed through `analysis.paths`"),
    ("analysis_outputs/", "LEGACY output, read-only — new output goes to results/"),
    ("archive/", "retired code and outputs"),
    ("temp/", "scratch — use results/ instead"),
    ("catboost_info/", "CatBoost training scratch"),
    ("build/", "PyInstaller build scratch"),
    ("dist/", "built launcher artefacts"),
]

# "If you are doing X, start at Y."  Curated because it encodes intent, not
# structure.  Paths are verified at build time and dropped if they disappear.
START_HERE: list[tuple[str, str]] = [
    ("any path or filename", "`backend/analysis/paths.py` — the single source of truth"),
    ("any analysis output", "`results_dir(\"<workflow>_<step>_<variant>\")` from `analysis.paths`"),
    ("a new survival estimator", "`backend/analysis/models/survival/`"),
    ("a new end-to-end analysis", "`backend/analysis/workflows/<name>/` + a wrapper in `scripts/run/`"),
    ("the production-risk forecast", "`backend/analysis/workflows/production_risk/run.py`"),
    ("the shipped Excel calculator", "`vba/` and `scripts/deploy/`"),
    ("agent conventions", "`agents/AGENTS.md`"),
]


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------


def tracked_files() -> list[str]:
    """Every git-tracked path, POSIX-separated.  Honours .gitignore for free."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return sorted(p for p in out.stdout.splitlines() if p.strip())


def _headline(doc: str) -> str:
    """The docstring's opening sentence, re-joined if the author wrapped it.

    A first line ending mid-clause — a trailing comma, or an open paren that
    never closes — reads as truncated in the map, so pull in following lines
    until the clause is whole (bounded, so a runaway docstring can't take over).
    """
    lines = [line.strip() for line in doc.strip().splitlines()]
    headline = lines[0]
    for follow in lines[1:3]:
        balanced = headline.count("(") == headline.count(")")
        if not follow or (balanced and not headline.endswith((",", "—", "-"))):
            break
        headline = f"{headline} {follow}"
    return headline


def module_summary(rel_path: str) -> tuple[str, str, bool]:
    """Docstring headline of a Python module: display form, raw form, is-shim.

    The raw form is returned untruncated because the shim section parses the
    re-export target out of it; truncating first would clip the target.
    Returns empty strings for unparseable or undocumented modules rather than
    raising — a syntax error in one file must not break the whole map.
    """
    path = REPO_ROOT / rel_path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return "", "", False
    doc = ast.get_docstring(tree)
    if not doc:
        return "", "", False
    raw = _headline(doc)
    is_shim = raw.lower().startswith(("backward-compat", "backward compat"))
    display = raw if len(raw) <= SUMMARY_WIDTH else raw[: SUMMARY_WIDTH - 1].rstrip() + "…"
    return display, raw, is_shim


def build_dir_index(files: list[str]) -> dict[str, int]:
    """Map every shown directory to its recursive tracked-file count."""
    counts: dict[str, int] = defaultdict(int)
    for path in files:
        parts = path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            directory = "/".join(parts[:depth])
            limit = MAX_TREE_DEPTH + 1 if directory.startswith(DEEP_ROOTS) else MAX_TREE_DEPTH
            if depth > limit:
                break
            counts[directory] += 1
    return dict(counts)


def render_tree(counts: dict[str, int], root_files: list[str]) -> list[str]:
    lines: list[str] = []
    for directory in sorted(counts):
        depth = directory.count("/")
        indent = "  " * depth
        name = directory.split("/")[-1]
        note = DIR_NOTES.get(directory, "")
        marker = " ⛔" if directory in LEGACY_DIRS else ""
        head = f"{indent}{name}/{marker}"
        tail = f"  [{counts[directory]}]"
        lines.append(f"{head}{tail}" + (f" — {note}" if note else ""))
    if root_files:
        lines.append("")
        lines.append("Repo root: " + ", ".join(f"`{f}`" for f in root_files))
    return lines


def render_detail(files: list[str], directory: str) -> list[str]:
    """Per-module docstring lines for one package."""
    members = [
        f
        for f in files
        if f.startswith(directory + "/")
        and f.endswith(".py")
        and "/" not in f[len(directory) + 1 :]
        and not f.endswith("__init__.py")
    ]
    if not members:
        return []
    lines = [f"**`{directory}/`**" + (f" — {DIR_NOTES[directory]}" if directory in DIR_NOTES else ""), ""]
    for member in members:
        summary, _, _ = module_summary(member)
        stem = member.rsplit("/", 1)[-1]
        lines.append(f"- `{stem}` — {summary}" if summary else f"- `{stem}` — *(no module docstring)*")
    lines.append("")
    return lines


def render_shims(files: list[str]) -> list[str]:
    """Collapse backward-compat shims into one warning block."""
    shims: list[tuple[str, str]] = []
    for path in files:
        if not path.endswith(".py"):
            continue
        _, raw, is_shim = module_summary(path)
        if is_shim:
            target = raw.split("use ", 1)[-1].split(" directly", 1)[0]
            target = target.split("import from ", 1)[-1].split(" directly", 1)[0]
            shims.append((path, target.strip().rstrip(".")))
    if not shims:
        return []
    lines = [
        f"{len(shims)} modules are import shims that re-export from elsewhere. Grep will "
        "match them; **import the canonical path instead**, and never add code to them.",
        "",
    ]
    for path, target in sorted(shims):
        lines.append(f"- `{path}` → `{target}`")
    lines.append("")
    return lines


def render_start_here() -> list[str]:
    lines = ["| If you are touching… | Start at |", "|---|---|"]
    for topic, target in START_HERE:
        lines.append(f"| {topic} | {target} |")
    lines.append("")
    return lines


def unannotated(counts: dict[str, int]) -> list[str]:
    missing = [d for d in sorted(counts) if d not in DIR_NOTES]
    if not missing:
        return []
    return [
        "These directories have no curated note yet — add one to `DIR_NOTES` in "
        "`scripts/build_project_map.py`:",
        "",
        ", ".join(f"`{d}`" for d in missing),
        "",
    ]


def build_document() -> str:
    files = tracked_files()
    counts = build_dir_index(files)
    root_files = [f for f in files if "/" not in f and not f.startswith(".")]
    n_py = sum(1 for f in files if f.endswith(".py"))

    out: list[str] = []
    add = out.append

    add("# Pump2 — project map")
    add("")
    add(
        "<!-- GENERATED by scripts/build_project_map.py — do not edit by hand. "
        "Structure comes from `git ls-files`; module notes come from the modules' own "
        "docstrings. Regenerate with `python scripts/build_project_map.py`. -->"
    )
    add("")
    add(f"{len(files)} tracked files, {n_py} Python. Counts in `[brackets]` are tracked files per directory.")
    add("")

    add("## Start here")
    add("")
    out.extend(render_start_here())

    add("## Layout")
    add("")
    add("```text")
    out.extend(render_tree(counts, root_files))
    add("```")
    add("")

    add("## Key packages")
    add("")
    for directory in DETAIL_DIRS:
        out.extend(render_detail(files, directory))
    for directory, note in sorted(GROUPED_NOTES.items()):
        if any(f.startswith(directory + "/") for f in files):
            add(f"**`{directory}/`** — {note}")
            add("")

    shim_block = render_shims(files)
    if shim_block:
        add("## ⚠ Import shims — not real modules")
        add("")
        out.extend(shim_block)

    add("## Not in git")
    add("")
    for name, note in UNTRACKED_DIRS:
        add(f"- `{name}` — {note}")
    add("")

    gap = unannotated(counts)
    if gap:
        add("## Unannotated")
        add("")
        out.extend(gap)

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    parser.add_argument("--check", action="store_true", help="exit 1 if the written map is stale")
    args = parser.parse_args(argv)

    document = build_document()

    if args.stdout:
        sys.stdout.write(document)
        return 0

    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is missing", file=sys.stderr)
            return 1
        if OUTPUT_PATH.read_text(encoding="utf-8") != document:
            print(
                f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is stale — run "
                "`python scripts/build_project_map.py`",
                file=sys.stderr,
            )
            return 1
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Skip the write when nothing changed so the file mtime — and git — stay quiet.
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text(encoding="utf-8") == document:
        return 0
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
