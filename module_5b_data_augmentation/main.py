"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  main.py
=============================================================================
CLI entry point for standalone data augmentation.

Usage:
  python main.py --source <M5_output_dir> [options]
  python main.py  # interactive mode

Options:
  --source PATH       Path to M5 preprocessing output (contains user_NNN/)
  --output PATH       Output directory (default: auto-numbered)
  --strategy STR      equalize | median | minimum_ratio (default: median)
  --device STR        cuda | cpu (default: cpu)
  --seed INT          Random seed (default: 42)
  --auto              Non-interactive mode (use config thresholds)
  --detect_only       Only run imbalance detection, don't augment
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure module directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODULE_LABEL, MODULE_VERSION, OUTPUT_ROOT
from augmenter import DataAugmenter
from imbalance_detector import ImbalanceDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _next_run_folder() -> Path:
    """Create auto-numbered output folder."""
    import re
    root = Path(__file__).parent / OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{MODULE_LABEL}_v{MODULE_VERSION}_run_(\d+)$")
    existing = [
        int(m.group(1))
        for d in root.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    n = (max(existing) + 1) if existing else 1
    folder = root / f"{MODULE_LABEL}_v{MODULE_VERSION}_run_{n:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _load_demographics(source_dir: Path) -> dict:
    """Load demographics from M5 preprocessing metadata files."""
    import json
    demographics = {}
    for user_dir in sorted(source_dir.iterdir()):
        if not user_dir.is_dir() or not user_dir.name.startswith("user_"):
            continue
        meta_path = user_dir / "preprocessing_metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            source_meta = meta.get("source_meta", {})
            profile = source_meta.get("user_profile", {})
            demographics[user_dir.name] = {
                "autism_severity": profile.get("autism_severity", "Medium"),
                "verbal_status": profile.get("verbal_status", "Verbal"),
            }
        except Exception:
            pass
    return demographics


def run_detect_only(source: Path) -> None:
    """Run imbalance detection only and print report."""
    detector = ImbalanceDetector(source)
    report = detector.analyse()

    print(f"\n{'=' * 60}")
    print(f"  Imbalance Detection Report")
    print(f"{'=' * 60}")
    print(report.summary_table().to_string(index=False))
    print(f"\n  IR = {report.imbalance_ratio}  |  Severity: {report.severity}")
    print(f"  Recommendation: {report.reason}")
    print(f"{'=' * 60}\n")


def run_augmentation(
    source: Path,
    output: Path,
    strategy: str = "median",
    device: str = "cpu",
    seed: int = 42,
    interactive: bool = True,
) -> None:
    """Run full augmentation pipeline."""
    demographics = _load_demographics(source)

    augmenter = DataAugmenter(
        source_dir=source,
        output_dir=output,
        demographics=demographics,
        strategy=strategy,
        interactive=interactive,
        device=device,
        seed=seed,
    )

    result = augmenter.run()

    if result.skipped:
        print(f"\n  Augmentation skipped: {result.skip_reason}")
    else:
        total = sum(
            len(segs)
            for by_emo in result.accepted_segments.values()
            for segs in by_emo.values()
        )
        print(f"\n  Augmentation complete!")
        print(f"  Synthetic segments generated: {total}")
        print(f"  Output: {result.output_dir}")
        print(f"  Time: {result.elapsed_s:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description=f"{MODULE_LABEL} v{MODULE_VERSION} — Data Augmentation"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to M5 preprocessing output directory"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: auto-numbered)"
    )
    parser.add_argument(
        "--strategy", type=str, default="median",
        choices=["equalize", "median", "minimum_ratio"],
        help="Augmentation strategy (default: median)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for TGAN training (default: cpu)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auto", action="store_true",
        help="Non-interactive mode (use config thresholds)"
    )
    parser.add_argument(
        "--detect_only", action="store_true",
        help="Only run imbalance detection"
    )

    args = parser.parse_args()

    # Source directory
    if args.source:
        source = Path(args.source)
    else:
        print(f"\n  [{MODULE_LABEL}] Data Augmentation Module v{MODULE_VERSION}")
        print("  Enter path to M5 preprocessing output:")
        source = Path(input("  > ").strip())

    if not source.exists():
        print(f"  ERROR: Source directory not found: {source}")
        sys.exit(1)

    if args.detect_only:
        run_detect_only(source)
        return

    output = Path(args.output) if args.output else _next_run_folder()

    run_augmentation(
        source=source,
        output=output,
        strategy=args.strategy,
        device=args.device,
        seed=args.seed,
        interactive=not args.auto,
    )


if __name__ == "__main__":
    main()
