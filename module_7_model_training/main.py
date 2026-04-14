"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  main.py
=============================================================================
CLI + interactive entry point for model training pipeline.
=============================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure module directory is on path for local imports
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import MODULE_VERSION, RANDOM_SEED
from trainer import ModelTrainer
from exporter import ModelExporter
from report_generator import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Module 7 — Model Training v{MODULE_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source data/combined_features.csv
  python main.py --source data/ --seed 42 --skip_temporal
  python main.py --source data/ --user_dependent --per_user_dir data/per_user/
        """,
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to M6 combined features CSV or directory")
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory (default: auto-numbered)")
    parser.add_argument(
        "--skip_temporal", action="store_true",
        help="Skip temporal models (BiLSTM, CNN, TPP, TFT)")
    parser.add_argument(
        "--skip_unsupervised", action="store_true",
        help="Skip unsupervised models")
    parser.add_argument(
        "--user_dependent", action="store_true",
        help="Train user-dependent models after global selection")
    parser.add_argument(
        "--per_user_dir", type=str, default=None,
        help="Directory with per-user CSVs for user-dependent training")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output")
    return parser.parse_args()


def _generate_word_report(output_dir: Path, results: list,
                          verbose: bool = True):
    """Generate the Word (.docx) report. Fails gracefully if python-docx missing."""
    try:
        rg = ReportGenerator(verbose=verbose)
        docx_path = rg.generate(output_dir, results)
        if verbose:
            print(f"  Word report: {docx_path}")
    except ImportError:
        if verbose:
            print("  [WARN] python-docx not installed — skipping Word report")
    except Exception as exc:
        if verbose:
            print(f"  [WARN] Word report generation failed: {exc}")


def run_interactive():
    """Interactive menu for model training."""
    print("=" * 60)
    print(f"  MODULE 7 — MODEL TRAINING v{MODULE_VERSION}")
    print("=" * 60)
    print()

    # Source
    source = input("Source CSV/directory path: ").strip()
    if not source:
        print("No source provided. Exiting.")
        return

    source_path = Path(source)
    if not source_path.exists():
        print(f"Path not found: {source_path}")
        return

    # Seed
    seed_input = input(f"Random seed [{RANDOM_SEED}]: ").strip()
    seed = int(seed_input) if seed_input else RANDOM_SEED

    # Options
    skip_temporal = input("Skip temporal models? [y/N]: ").strip().lower() == "y"
    skip_unsupervised = input("Skip unsupervised models? [y/N]: ").strip().lower() == "y"
    user_dep = input("Train user-dependent models? [y/N]: ").strip().lower() == "y"

    per_user_dir = None
    if user_dep:
        per_user = input("Per-user data directory: ").strip()
        if per_user:
            per_user_dir = Path(per_user)

    print()
    trainer = ModelTrainer(source_path, seed=seed, verbose=True)
    output_dir = trainer.run(
        skip_temporal=skip_temporal,
        skip_unsupervised=skip_unsupervised,
        user_dependent=user_dep,
        per_user_dir=per_user_dir,
    )

    # Export HTML report
    exporter = ModelExporter(verbose=True)
    comparison_dir = output_dir / "comparison"
    if comparison_dir.exists():
        exporter.export_comparison_html(
            None, trainer.results, comparison_dir)
        exporter.export_radar_chart(trainer.results, comparison_dir)

    # Generate Word report
    _generate_word_report(output_dir, trainer.results, verbose=True)

    print(f"\nDone. Results in: {output_dir}")


def run_from_cli(args):
    """Run from command-line arguments."""
    if args.source is None:
        print("Error: --source is required")
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"Error: source not found: {source}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    per_user_dir = Path(args.per_user_dir) if args.per_user_dir else None

    trainer = ModelTrainer(
        source, seed=args.seed, verbose=not args.quiet,
        output_dir=output_dir)
    output_dir = trainer.run(
        skip_temporal=args.skip_temporal,
        skip_unsupervised=args.skip_unsupervised,
        user_dependent=args.user_dependent,
        per_user_dir=per_user_dir,
    )

    # Export HTML report
    exporter = ModelExporter(verbose=not args.quiet)
    comparison_dir = output_dir / "comparison"
    if comparison_dir.exists():
        exporter.export_comparison_html(
            None, trainer.results, comparison_dir)
        exporter.export_radar_chart(trainer.results, comparison_dir)

    # Generate Word report
    _generate_word_report(output_dir, trainer.results, verbose=not args.quiet)


def run_module_7(source: str | Path, output_dir: Path | None = None,
                 seed: int = RANDOM_SEED, verbose: bool = True,
                 skip_temporal: bool = False,
                 skip_unsupervised: bool = False,
                 user_dependent: bool = False,
                 per_user_dir: Path | None = None,
                 ) -> Path:
    """
    Programmatic entry point for pipeline integration.

    Returns
    -------
    output_dir : Path to output directory with all results
    """
    trainer = ModelTrainer(
        source, seed=seed, verbose=verbose, output_dir=output_dir)
    out = trainer.run(
        skip_temporal=skip_temporal,
        skip_unsupervised=skip_unsupervised,
        user_dependent=user_dependent,
        per_user_dir=per_user_dir,
    )

    # Export reports
    exporter = ModelExporter(verbose=verbose)
    comparison_dir = out / "comparison"
    if comparison_dir.exists():
        exporter.export_comparison_html(
            None, trainer.results, comparison_dir)
        exporter.export_radar_chart(trainer.results, comparison_dir)

    # Generate Word report
    _generate_word_report(out, trainer.results, verbose=verbose)

    return out


if __name__ == "__main__":
    args = parse_args()
    if args.source:
        run_from_cli(args)
    else:
        run_interactive()
