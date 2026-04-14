"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  main.py
=============================================================================
CLI entry point for Module 4 combined multi-user EDA.

Launch:
  python main.py                            (from module_4_eda/)
  python main.py --source <path_to_M3_output>
=============================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure module directory is on path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from analyser import CombinedEDA
from config import MODULE_VERSION, MODULE_LABEL


def main():
    parser = argparse.ArgumentParser(
        description=f"Module 4 EDA v{MODULE_VERSION} — Combined Multi-User Analysis"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to M3 split output (containing train/user_NNN/ folders)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (auto-numbered if not specified)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  MODULE {MODULE_LABEL} v{MODULE_VERSION} — Exploratory Data Analysis")
    print(f"  Combined multi-user analysis with thesis-quality Word report")
    print(f"{'=' * 70}\n")

    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"  ERROR: Source path does not exist: {source_path}")
            sys.exit(1)

        # Discover user folders under train/
        train_dir = source_path / "train"
        if not train_dir.exists():
            train_dir = source_path  # try directly

        user_folders = {}
        for d in sorted(train_dir.iterdir()):
            if d.is_dir() and d.name.startswith("user_"):
                user_folders[d.name] = d

        if not user_folders:
            print(f"  ERROR: No user_NNN folders found in {train_dir}")
            sys.exit(1)

        print(f"  Found {len(user_folders)} training user folders")
        eda = CombinedEDA(verbose=True)
        results = eda.run(
            source={"folders": user_folders},
            output_dir=args.output,
        )
    else:
        print("  No --source provided. Use with pipeline or provide M3 output path.")
        print(f"\n  Usage: python main.py --source <path_to_M3_split_output>")
        sys.exit(0)

    print(f"\n  Report: {results.report_path}")
    print(f"  Output: {results.output_dir}")
    print(f"  Done!\n")


if __name__ == "__main__":
    main()
