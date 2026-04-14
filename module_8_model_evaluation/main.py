"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  main.py
=============================================================================
CLI entry point and 11-phase orchestrator for held-out test set evaluation.

Usage (standalone):
    python module_8_model_evaluation/main.py \
        --m7_dir outputs/run_NNN/module_7_model_training \
        --features outputs/run_NNN/module_6_.../combined_features_all_users_split.csv \
        --output_dir outputs/run_NNN/module_8_model_evaluation \
        --seed 42

Usage (pipeline):
    Called by run_pipeline_auto.py via run_module_8()
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Ensure module directory is on path for local imports
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import MODULE_VERSION, MODULE_LABEL, RANDOM_SEED


class ModelEvaluationModule:
    """Module 8 -- Model Evaluation on Held-Out Test Set."""

    def __init__(self, verbose: bool = True, seed: int = RANDOM_SEED):
        self.verbose = verbose
        self.seed = seed

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{MODULE_LABEL}] {msg}")

    def run(
        self,
        m7_dir: Path,
        features_csv: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Execute the full 11-phase evaluation pipeline.

        Parameters
        ----------
        m7_dir : Path to M7 output (contains supervised/, training_metadata.json)
        features_csv : Path to M6 combined_features_all_users_split.csv
        output_dir : Path to write M8 outputs

        Returns
        -------
        Dict with summary results and paths
        """
        t_start = time.time()
        m7_dir = Path(m7_dir)
        features_csv = Path(features_csv)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        self._log(f"Module 8 v{MODULE_VERSION} -- Model Evaluation")
        self._log(f"  M7 dir:      {m7_dir}")
        self._log(f"  Features:    {features_csv}")
        self._log(f"  Output:      {output_dir}")
        self._log(f"  Seed:        {self.seed}")

        # ─── Phase 1: Load data + models ──────────────────────────────
        self._log("Phase 1/11: Loading data and models...")
        from data_loader import M8DataLoader
        from model_loader import ModelLoader

        loader = M8DataLoader(verbose=self.verbose)
        test_data = loader.load_test_data(features_csv)
        m7_meta = None
        try:
            m7_meta = loader.load_m7_metadata(m7_dir)
        except FileNotFoundError:
            self._log("  WARNING: M7 metadata not found -- continuing without it")

        val_metrics = loader.load_per_model_val_metrics(m7_dir)
        split_dists = loader.load_train_val_summary(features_csv)

        model_loader = ModelLoader(verbose=self.verbose)
        models = model_loader.load_all(m7_dir, supervised_only=True)

        if not models:
            self._log("ERROR: No models loaded -- cannot proceed")
            return {"error": "No models loaded"}

        self._log(f"  Loaded {test_data.n_samples} test samples, "
                  f"{len(models)} models")

        # ─── Phase 2: Test-set evaluation ─────────────────────────────
        self._log("Phase 2/11: Evaluating on test set...")
        from test_evaluator import TestEvaluator

        evaluator = TestEvaluator(verbose=self.verbose)
        eval_results = evaluator.evaluate_all(
            models, test_data.X, test_data.y,
            test_data.class_names, val_metrics,
        )
        comparison_df = evaluator.build_comparison_table(eval_results)

        # ─── Phase 2b: Forecast evaluation (BiLSTM Forecaster) ────────
        forecast_eval_result = None
        forecast_models = {k: v for k, v in models.items()
                          if v.data_format == "forecast"}
        if forecast_models:
            self._log("Phase 2b: Evaluating forecaster models...")
            from forecast_evaluator import ForecastEvaluator
            fc_eval = ForecastEvaluator(verbose=self.verbose, seed=self.seed)
            forecast_eval_result = fc_eval.evaluate_all(
                forecast_models, m7_dir, features_csv,
            )
            # Add forecast result to eval_results for reporting
            if forecast_eval_result:
                eval_results.extend(forecast_eval_result)
        else:
            self._log("Phase 2b: No forecaster models found, skipping")

        # ─── Phase 3: Generalisation gap ──────────────────────────────
        self._log("Phase 3/11: Analysing generalisation gap...")
        from generalization_analyzer import GeneralizationAnalyzer

        gen_analyzer = GeneralizationAnalyzer(verbose=self.verbose)
        gen_results = gen_analyzer.analyse_all(eval_results, split_dists)
        gap_table = gen_analyzer.build_gap_table(gen_results)
        gap_matrix = gen_analyzer.build_gap_matrix(gen_results)

        # ─── Phase 4: Calibration ─────────────────────────────────────
        self._log("Phase 4/11: Analysing calibration...")
        from calibration_analyzer import CalibrationAnalyzer

        cal_analyzer = CalibrationAnalyzer(verbose=self.verbose)
        cal_results = cal_analyzer.analyse_all(
            eval_results, test_data.y, test_data.class_names
        )

        # ─── Phase 5: Clinical utility ────────────────────────────────
        self._log("Phase 5/11: Clinical utility analysis...")
        from clinical_analyzer import ClinicalAnalyzer

        clin_analyzer = ClinicalAnalyzer(verbose=self.verbose)
        clinical_results = clin_analyzer.analyse_all(
            eval_results, test_data.y, test_data.class_names
        )

        # ─── Phase 6: Demographic evaluation ──────────────────────────
        self._log("Phase 6/11: Demographic evaluation...")
        from demographic_evaluator import DemographicEvaluator

        demo_evaluator = DemographicEvaluator(verbose=self.verbose)
        demo_results = demo_evaluator.analyse_all(
            eval_results, test_data.y, test_data.demographics
        )

        # ─── Phase 7: Robustness testing ──────────────────────────────
        self._log("Phase 7/11: Robustness testing (top-N models)...")
        from robustness_tester import RobustnessTester

        rob_tester = RobustnessTester(seed=self.seed, verbose=self.verbose)
        rob_results = rob_tester.test_all(
            eval_results, models, test_data.X, test_data.y,
            test_data.feature_names,
        )

        # ─── Phase 8: Statistical comparison ──────────────────────────
        self._log("Phase 8/11: Statistical model comparison...")
        from statistical_comparator import StatisticalComparator

        stat_comp = StatisticalComparator(verbose=self.verbose)
        stat_comparison = stat_comp.compare_all(
            eval_results, test_data.y, test_data.class_names
        )

        # ─── Phase 9: Error analysis ──────────────────────────────────
        self._log("Phase 9/11: Error analysis...")
        from error_analyzer import ErrorAnalyzer

        err_analyzer = ErrorAnalyzer(verbose=self.verbose)
        error_results = err_analyzer.analyse_all(
            eval_results, test_data.y, test_data.class_names,
            test_data.X, test_data.feature_names, test_data.user_ids,
        )

        # ─── Phase 10: Inference profiling ────────────────────────────
        self._log("Phase 10/11: Inference profiling...")
        from inference_profiler import InferenceProfiler

        profiler = InferenceProfiler(verbose=self.verbose)
        prof_results = profiler.profile_all(
            eval_results, models, test_data.X,
        )

        # ─── Phase 11: Visualisation + Report ─────────────────────────
        self._log("Phase 11/11: Generating visualisations and report...")
        from demographic_evaluator import DemographicEvaluator as DE
        import visualizer as viz

        plot_paths = self._generate_all_plots(
            eval_results, test_data, comparison_df,
            gen_results, gap_table, gap_matrix,
            cal_results, clinical_results, demo_results,
            rob_results, stat_comparison, error_results,
            prof_results, plots_dir,
        )

        # Word report
        from report_generator import M8ReportGenerator
        report_gen = M8ReportGenerator(verbose=self.verbose)

        subgroup_df = pd.DataFrame()
        equity_df = pd.DataFrame()
        if demo_results:
            de = DemographicEvaluator(verbose=False)
            subgroup_df = de.build_subgroup_table(demo_results)
            equity_df = de.build_equity_summary(demo_results)

        report_path = report_gen.build(
            output_dir=output_dir,
            eval_results=eval_results,
            comparison_df=comparison_df,
            gen_results=gen_results,
            gap_table=gap_table,
            gap_matrix=gap_matrix,
            cal_results=cal_results,
            clinical_results=clinical_results,
            demo_results=demo_results,
            rob_results=rob_results,
            stat_comparison=stat_comparison,
            error_results=error_results,
            prof_results=prof_results,
            test_data=test_data,
            m7_meta=m7_meta,
            split_dists=split_dists,
            plot_paths=plot_paths,
        )

        # ─── Save metadata + CSV outputs ──────────────────────────────
        total_time = time.time() - t_start
        self._log(f"Total time: {total_time:.1f}s")

        # Save comparison CSV
        if not comparison_df.empty:
            comparison_df.to_csv(output_dir / "test_comparison.csv", index=False)

        # Save metadata
        best = next((r for r in eval_results if not r.skipped), None)
        metadata = {
            "module": MODULE_LABEL,
            "version": MODULE_VERSION,
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "m7_dir": str(m7_dir),
            "features_csv": str(features_csv),
            "n_test_samples": test_data.n_samples,
            "n_classes": test_data.n_classes,
            "class_names": test_data.class_names,
            "n_models_evaluated": len([r for r in eval_results if not r.skipped]),
            "n_models_skipped": len([r for r in eval_results if r.skipped]),
            "best_model": best.model_name if best else None,
            "best_test_f1": best.test_metrics.get("f1_weighted", 0) if best else 0,
            "total_time_s": total_time,
            "report_path": str(report_path),
        }
        with open(output_dir / "evaluation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self._log(f"DONE -- Best model: {metadata['best_model']} "
                  f"(F1={metadata['best_test_f1']:.4f})")

        return metadata

    def _generate_all_plots(
        self,
        eval_results, test_data, comparison_df,
        gen_results, gap_table, gap_matrix,
        cal_results, clinical_results, demo_results,
        rob_results, stat_comparison, error_results,
        prof_results, plots_dir,
    ) -> Dict[str, Path]:
        """Generate all 32 charts and return path dict."""
        import visualizer as viz
        from demographic_evaluator import DemographicEvaluator

        paths = {}

        # Section 3
        paths["v01"] = viz.v01_confusion_matrices(eval_results, test_data.class_names, plots_dir)
        paths["v02"] = viz.v02_roc_curves(eval_results, test_data.y, test_data.class_names, plots_dir)
        paths["v03"] = viz.v03_precision_recall_curves(eval_results, test_data.y, test_data.class_names, plots_dir)
        paths["v04"] = viz.v04_metric_comparison_bar(comparison_df, plots_dir)
        paths["v05"] = viz.v05_per_class_f1_heatmap(eval_results, test_data.class_names, plots_dir)

        # Section 4
        paths["v06"] = viz.v06_train_val_test_comparison(gap_table, plots_dir)
        paths["v07"] = viz.v07_overfitting_heatmap(gap_matrix, plots_dir)
        paths["v08"] = viz.v08_generalization_waterfall(gen_results, plots_dir)

        # Section 5
        paths["v09"] = viz.v09_reliability_diagrams(cal_results, plots_dir)
        paths["v10"] = viz.v10_ece_comparison(cal_results, plots_dir)
        paths["v11"] = viz.v11_confidence_histogram(eval_results, plots_dir)

        # Section 6
        paths["v12"] = viz.v12_hp_fnr_comparison(clinical_results, plots_dir)
        paths["v13"] = viz.v13_clinical_cost_heatmap(clinical_results, test_data.class_names, plots_dir)
        paths["v14"] = viz.v14_nns_bar(clinical_results, plots_dir)
        paths["v15"] = viz.v15_sensitivity_specificity(clinical_results, test_data.class_names, plots_dir)

        # Section 7+8
        de = DemographicEvaluator(verbose=False)
        subgroup_df = de.build_subgroup_table(demo_results) if demo_results else pd.DataFrame()
        equity_df = de.build_equity_summary(demo_results) if demo_results else pd.DataFrame()

        paths["v16"] = viz.v16_f1_by_severity(subgroup_df, plots_dir)
        paths["v17"] = viz.v17_f1_by_verbal(subgroup_df, plots_dir)
        paths["v18"] = viz.v18_equity_radar(demo_results, plots_dir)
        paths["v19"] = viz.v19_disparity_heatmap(equity_df, plots_dir)

        # Section 9
        paths["v20"] = viz.v20_noise_degradation(rob_results, plots_dir)
        paths["v21"] = viz.v21_channel_dropout(rob_results, plots_dir)
        paths["v22"] = viz.v22_missing_feature_impact(rob_results, plots_dir)
        paths["v23"] = viz.v23_robustness_ranking(rob_results, plots_dir)

        # Section 10
        from statistical_comparator import StatisticalComparator
        model_names = [r.model_name for r in eval_results if not r.skipped]
        sc = StatisticalComparator(verbose=False)
        mcnemar_matrix = sc.build_mcnemar_matrix(stat_comparison, model_names)
        paths["v24"] = viz.v24_mcnemar_heatmap(mcnemar_matrix, plots_dir)
        paths["v25"] = viz.v25_delong_forest(stat_comparison, plots_dir)
        paths["v26"] = viz.v26_cochran_q(stat_comparison, plots_dir)

        # Section 11
        paths["v27"] = viz.v27_confused_pairs(error_results, plots_dir)
        paths["v28"] = viz.v28_per_user_error(error_results, plots_dir)
        paths["v29"] = viz.v29_misclassification_flow(error_results, test_data.class_names, plots_dir)
        paths["v30"] = viz.v30_failure_tsne(eval_results, test_data.y, test_data.X, test_data.class_names, plots_dir)

        # Section 12
        paths["v31"] = viz.v31_latency_comparison(prof_results, plots_dir)
        paths["v32"] = viz.v32_pareto_frontier(eval_results, prof_results, plots_dir)

        self._log(f"  Generated {len(paths)} charts")
        return paths


# Need pandas for the module
import pandas as pd


def run_module_8(
    m7_dir: str,
    features_csv: str,
    output_dir: str,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Convenience function for pipeline integration."""
    module = ModelEvaluationModule(verbose=verbose, seed=seed)
    return module.run(
        m7_dir=Path(m7_dir),
        features_csv=Path(features_csv),
        output_dir=Path(output_dir),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Module 8 -- Model Evaluation on Held-Out Test Set"
    )
    parser.add_argument("--m7_dir", required=True,
                        help="Path to M7 output directory")
    parser.add_argument("--features", required=True,
                        help="Path to M6 combined_features_all_users_split.csv")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: auto-numbered)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Auto-number output dir if not specified
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        import re
        out_root = _THIS_DIR / "outputs"
        out_root.mkdir(exist_ok=True)
        pattern = re.compile(rf"M8_v{re.escape(MODULE_VERSION)}_run_(\d+)")
        existing = []
        if out_root.exists():
            for d in out_root.iterdir():
                m = pattern.match(d.name)
                if m:
                    existing.append(int(m.group(1)))
        next_num = (max(existing) + 1) if existing else 1
        out = out_root / f"M8_v{MODULE_VERSION}_run_{next_num:03d}"

    result = run_module_8(
        m7_dir=args.m7_dir,
        features_csv=args.features,
        output_dir=str(out),
        seed=args.seed,
        verbose=not args.quiet,
    )

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
