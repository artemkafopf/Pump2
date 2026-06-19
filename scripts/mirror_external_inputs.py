from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.input_paths import (  # noqa: E402
    DEFAULT_PRESENTATION_EXTERNAL,
    DEFAULT_V03_ALL_EXTERNAL,
    DEFAULT_V03_FAILURES_EXTERNAL,
    LOCAL_INPUT_DIR,
)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy external Excel and presentation inputs into the project.")
    parser.add_argument("--target-dir", default=str(LOCAL_INPUT_DIR), help="Destination directory for mirrored inputs.")
    parser.add_argument("--v03-all-source", default=str(DEFAULT_V03_ALL_EXTERNAL), help="Source V03_all workbook path.")
    parser.add_argument("--v03-failures-source", default=str(DEFAULT_V03_FAILURES_EXTERNAL), help="Source V03_failures workbook path.")
    parser.add_argument("--presentation-source", default=str(DEFAULT_PRESENTATION_EXTERNAL), help="Source presentation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = Path(args.target_dir)
    copy_file(Path(args.v03_all_source), target_dir / "Отказы свод с анализом_БДА_V03_all.xlsx")
    copy_file(Path(args.v03_failures_source), target_dir / "Отказы свод с анализом_БДА_V03_failures.xlsx")
    copy_file(Path(args.presentation_source), target_dir / "Отказность Аналитика(1).pptx")
    print(f"inputs -> {target_dir}")


if __name__ == "__main__":
    main()
