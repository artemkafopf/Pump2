"""CLI wrapper — ESP survival analysis.

Usage:
    python scripts/run/esp_survival.py [phases]

    phases: comma-separated subset, e.g. "0,1,2,3" or "all" (default)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.workflows.esp_survival.run import main

if __name__ == "__main__":
    phases_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(phases_arg)
