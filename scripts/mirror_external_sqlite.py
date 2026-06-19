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

from analysis.sqlite_paths import (  # noqa: E402
    DEFAULT_TECHREGIME_EXTERNAL,
    DEFAULT_TELEMETRY_EXTERNAL,
    LOCAL_SQLITE_DIR,
)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy external telemetry and techregime SQLite files into the project.")
    parser.add_argument("--target-dir", default=str(LOCAL_SQLITE_DIR), help="Destination directory for mirrored SQLite files.")
    parser.add_argument("--telemetry-source", default=str(DEFAULT_TELEMETRY_EXTERNAL), help="Source telemetry.sqlite path.")
    parser.add_argument("--techregime-source", default=str(DEFAULT_TECHREGIME_EXTERNAL), help="Source techregime.sqlite path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = Path(args.target_dir)
    telemetry_target = target_dir / "telemetry.sqlite"
    techregime_target = target_dir / "techregime.sqlite"

    copy_file(Path(args.telemetry_source), telemetry_target)
    copy_file(Path(args.techregime_source), techregime_target)

    print(f"telemetry -> {telemetry_target}")
    print(f"techregime -> {techregime_target}")


if __name__ == "__main__":
    main()
