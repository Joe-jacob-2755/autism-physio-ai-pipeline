"""
=============================================================================
MODULE 3B - DATA SPLITTING  |  main.py  (v1.0.0)
=============================================================================
Interactive CLI entry point for user-level train/val/test data splitting.

This module sits between M3 (Preprocessing) and M4 (Feature Engineering).
It splits data by user_id to prevent data leakage from participant-specific
physiological patterns.

Launch from repo root:
  Windows:    run_data_splitting.bat
  Mac/Linux:  ./run_data_splitting.sh

Or directly:
  cd module_3_data_splitting
  python main.py
  python main.py --source ../module_5_preprocessing/outputs/M5_v1.0.0_run_001
  python main.py --sources dir1 dir2 dir3 --mode three_way --train 0.6
  python main.py --sources dir1 dir2 --mode two_way --train 0.8 --test 0.2
=============================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
M5_DIR = REPO_ROOT / "module_5_preprocessing"

sys.path.insert(0, str(MODULE_DIR))

from config import (
    MODULE_VERSION, MODULE_LABEL,
    DEFAULT_SPLIT_MODE, DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO, DEFAULT_TRAIN_RATIO_2WAY, DEFAULT_TEST_RATIO_2WAY,
    DEFAULT_SEED,
)
from splitter import DataSplitter
from exporter import SplitExporter

WIDTH = 64


def _line():
    print("─" * WIDTH)


def _header(t):
    print()
    print("═" * WIDTH)
    print(f"  {t}")
    print("═" * WIDTH)


def _section(t):
    print(f"\n  {'─' * (WIDTH - 4)}\n  {t}\n  {'─' * (WIDTH - 4)}")


def _ask(prompt, default=""):
    return input(f"\n  {prompt} [{default}]: ").strip() or str(default)


def _ask_yn(prompt, default=True):
    raw = input(f"\n  {prompt} ({'Y/n' if default else 'y/N'}): ").strip().lower()
    return default if raw == "" else raw in ("y", "yes")


def _ask_float(prompt, default):
    raw = input(f"\n  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid number, using default: {default}")
        return default


# ── Source browsing ───────────────────────────────────────────────────────

def _browse_sources() -> list:
    """Browse for M3 output folders to split."""
    _section("Select M3 data source(s)")
    candidates = []

    m5_out = M5_DIR / "outputs"
    if m5_out.exists():
        for d in sorted(m5_out.iterdir()):
            if d.is_dir():
                feat_raw = d / "features_raw"
                cleaned = d / "cleaned_signals"
                if feat_raw.exists() or cleaned.exists():
                    csv_count = len(list(d.rglob("*.csv")))
                    candidates.append(
                        (d, f"M3: {d.name}  ({csv_count} CSVs)"))

    if not candidates:
        print("  No M3 output folders found. Enter path(s) manually.")
        raw = input("\n  Enter folder path(s), comma-separated: ").strip()
        if not raw:
            return []
        return [Path(p.strip()) for p in raw.split(",") if Path(p.strip()).exists()]

    print()
    for i, (p, lbl) in enumerate(candidates, 1):
        print(f"    {i}.  {lbl}")
    print(f"    {len(candidates) + 1}.  Enter path manually")

    print("\n  Select one or more (comma-separated numbers, e.g. 1,2,3)")
    raw = input("  Selection [1]: ").strip() or "1"

    selected = []
    for token in raw.split(","):
        token = token.strip()
        try:
            idx = int(token) - 1
            if idx == len(candidates):
                p = Path(input("  Enter path: ").strip())
                if p.exists():
                    selected.append(p)
            elif 0 <= idx < len(candidates):
                selected.append(candidates[idx][0])
        except ValueError:
            p = Path(token)
            if p.exists():
                selected.append(p)
            else:
                print(f"  Invalid: {token}")

    return selected


# ── Interactive mode ──────────────────────────────────────────────────────

def _interactive_mode():
    """Full interactive guided session."""
    os.system("cls" if os.name == "nt" else "clear")
    print()
    print("  " + "=" * (WIDTH - 4))
    print(f"  |    AUTISM PHYSIO-AI PIPELINE" + " " * (WIDTH - 38) + "|")
    print(f"  |    Module 3B - Data Splitting" + " " * (WIDTH - 39) + "|")
    print(f"  |    v{MODULE_VERSION}" + " " * (WIDTH - 14) + "|")
    print("  " + "=" * (WIDTH - 4))

    # ── Source ────────────────────────────────────────────────────────
    sources = _browse_sources()
    if not sources:
        print("  No valid sources selected. Exiting.")
        return

    # ── Split mode ────────────────────────────────────────────────────
    _section("Split configuration")
    print("""
  Split modes:
    1. Three-way  (train / validation / test)  [default]
       Default: 60% / 20% / 20%

    2. Two-way    (train / test)
       Default: 80% / 20%
""")
    mode_idx = int(_ask("Select mode", "1"))
    split_mode = "two_way" if mode_idx == 2 else "three_way"

    # ── Ratios ────────────────────────────────────────────────────────
    _section("Split ratios")
    if split_mode == "three_way":
        train_r = _ask_float("Train ratio", DEFAULT_TRAIN_RATIO)
        val_r = _ask_float("Validation ratio", DEFAULT_VAL_RATIO)
        test_r = _ask_float("Test ratio", DEFAULT_TEST_RATIO)
    else:
        train_r = _ask_float("Train ratio", DEFAULT_TRAIN_RATIO_2WAY)
        val_r = 0.0
        test_r = _ask_float("Test ratio", DEFAULT_TEST_RATIO_2WAY)

    total = train_r + val_r + test_r
    if abs(total - 1.0) > 0.01:
        print(f"\n  WARNING: Ratios sum to {total:.2f}, not 1.0. "
              "Adjusting proportionally.")
        train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    # ── Seed ──────────────────────────────────────────────────────────
    seed = int(_ask("Random seed", str(DEFAULT_SEED)))

    # ── Stratification ────────────────────────────────────────────────
    stratify = _ask_yn("Stratify by autism_severity + verbal_status?",
                       default=True)

    # ── Summary ───────────────────────────────────────────────────────
    _section("Split Parameters Summary")
    print(f"""
  Sources        : {len(sources)} M3 folder(s)
  Split mode     : {split_mode}
  Train ratio    : {train_r:.0%}
  Val ratio      : {val_r:.0%}
  Test ratio     : {test_r:.0%}
  Seed           : {seed}
  Stratified     : {stratify}
""")
    if not _ask_yn("Start splitting?", default=True):
        print("  Cancelled.")
        return

    # ── Run ───────────────────────────────────────────────────────────
    _header("Running data split")

    splitter = DataSplitter(
        split_mode=split_mode,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        stratify=stratify,
        seed=seed,
        verbose=True,
    )

    # Load data from M3 outputs
    feature_dfs, combined_df, user_demo_df = splitter.load_from_m5_outputs(
        sources)

    if user_demo_df.empty:
        print("\n  ERROR: No user demographics found. Cannot split.")
        return

    # Split users
    split_result = splitter.split_users(user_demo_df)

    # Split data
    split_result = splitter.split_data(split_result, feature_dfs, combined_df)

    # Export
    exporter = SplitExporter()
    saved = exporter.export_all(split_result)

    # Done
    _header("Split complete")
    print(f"""
  Output folder  : {exporter.run_dir}
  Train users    : {len(split_result.train_users)}
  Val users      : {len(split_result.val_users)}
  Test users     : {len(split_result.test_users)}
  Files written  : {len(saved)}

  Next step: Run Module 4 (Feature Engineering) on the train split.
  Test data is saved separately — do NOT use it until final evaluation.
""")


# ── CLI mode ──────────────────────────────────────────────────────────────

def _cli_mode():
    """Argument-driven mode."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            f"Module 3B - Data Splitting (User-level train/val/test)  "
            f"v{MODULE_VERSION}"
        ),
    )
    parser.add_argument(
        "--source", default=None,
        help="Single M3 output folder path")
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Multiple M3 output folder paths (one per user)")
    parser.add_argument(
        "--mode", default=DEFAULT_SPLIT_MODE,
        choices=["two_way", "three_way"],
        help=f"Split mode (default: {DEFAULT_SPLIT_MODE})")
    parser.add_argument(
        "--train", type=float, default=None,
        help="Train ratio (e.g. 0.6)")
    parser.add_argument(
        "--val", type=float, default=None,
        help="Validation ratio (e.g. 0.2, ignored in two_way)")
    parser.add_argument(
        "--test", type=float, default=None,
        help="Test ratio (e.g. 0.2)")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--no_stratify", action="store_true",
        help="Disable stratified splitting")
    parser.add_argument(
        "--output_dir", default=None,
        help="Custom output root directory")

    args = parser.parse_args()

    # Resolve sources
    if args.sources:
        source_dirs = [Path(s) for s in args.sources]
    elif args.source:
        source_dirs = [Path(args.source)]
    else:
        parser.error("--source or --sources is required in CLI mode")

    for s in source_dirs:
        if not s.exists():
            parser.error(f"Source not found: {s}")

    # Build splitter
    splitter = DataSplitter(
        split_mode=args.mode,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        stratify=not args.no_stratify,
        seed=args.seed,
        verbose=True,
    )

    # Load, split, export
    feature_dfs, combined_df, user_demo_df = splitter.load_from_m5_outputs(
        source_dirs)

    if user_demo_df.empty:
        print("ERROR: No user demographics found. Cannot split.")
        sys.exit(1)

    split_result = splitter.split_users(user_demo_df)
    split_result = splitter.split_data(split_result, feature_dfs, combined_df)

    exporter = SplitExporter(
        output_root=args.output_dir if args.output_dir else None)
    saved = exporter.export_all(split_result)

    print(f"\nOutput: {exporter.run_dir}")
    print(f"Files:  {len(saved)}")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 1:
        _interactive_mode()
    else:
        _cli_mode()
