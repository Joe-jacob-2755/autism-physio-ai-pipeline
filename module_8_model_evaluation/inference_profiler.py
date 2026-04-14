"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  inference_profiler.py
=============================================================================
Profile inference latency, throughput, and memory footprint per model.
Results inform deployment feasibility (real-time wearable vs. batch server).
=============================================================================
"""
from __future__ import annotations

import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import PROFILE_N_RUNS, PROFILE_BATCH_SIZES

log = logging.getLogger(__name__)


@dataclass
class LatencyProfile:
    """Latency profile for one batch size."""
    batch_size: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    throughput_samples_per_s: float


@dataclass
class InferenceProfile:
    """Full inference profile for one model."""
    model_name: str
    framework: str
    model_size_kb: float
    latency_profiles: List[LatencyProfile] = field(default_factory=list)
    single_sample_ms: float = 0.0
    # Deployment feasibility
    meets_realtime: bool = False   # <100ms per sample for wearable
    meets_batch: bool = False      # <1s per 64-sample batch


class InferenceProfiler:
    """Profile inference performance of trained models."""

    def __init__(
        self,
        n_runs: int = PROFILE_N_RUNS,
        batch_sizes: Optional[List[int]] = None,
        verbose: bool = True,
    ):
        self.n_runs = n_runs
        self.batch_sizes = batch_sizes or PROFILE_BATCH_SIZES
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Prof] {msg}")

    def profile_all(
        self,
        eval_results: list,
        models: dict,
        X_test: np.ndarray,
    ) -> Dict[str, InferenceProfile]:
        """
        Profile inference for all non-skipped models.

        Parameters
        ----------
        eval_results : List[EvalResult]
        models : dict of {name: LoadedModel}
        X_test : (n_test, n_features)

        Returns
        -------
        Dict[model_name, InferenceProfile]
        """
        results: Dict[str, InferenceProfile] = {}

        for er in eval_results:
            if er.skipped or er.model_name not in models:
                continue

            model = models[er.model_name]
            ip = self._profile_one(model, X_test)
            results[er.model_name] = ip

            self._log(
                f"  {er.model_name}: single={ip.single_sample_ms:.2f}ms, "
                f"realtime={'YES' if ip.meets_realtime else 'NO'}"
            )

        return results

    def _profile_one(
        self, model, X_test: np.ndarray
    ) -> InferenceProfile:
        """Profile a single model."""
        ip = InferenceProfile(
            model_name=model.name,
            framework=model.framework,
            model_size_kb=model.model_size_kb,
        )

        for batch_size in self.batch_sizes:
            lp = self._profile_batch(model, X_test, batch_size)
            ip.latency_profiles.append(lp)

        # Single sample latency (batch_size=1)
        single = next(
            (lp for lp in ip.latency_profiles if lp.batch_size == 1), None
        )
        if single:
            ip.single_sample_ms = single.median_ms

        # Deployment feasibility
        ip.meets_realtime = ip.single_sample_ms < 100.0  # <100ms
        batch_64 = next(
            (lp for lp in ip.latency_profiles if lp.batch_size == 64), None
        )
        if batch_64:
            ip.meets_batch = batch_64.median_ms < 1000.0  # <1s

        return ip

    def _profile_batch(
        self, model, X_test: np.ndarray, batch_size: int
    ) -> LatencyProfile:
        """Profile inference latency at a given batch size."""
        n_test = X_test.shape[0]
        batch = X_test[:min(batch_size, n_test)]

        # Warm-up run
        try:
            model.predict(batch)
        except Exception:
            pass

        # Timed runs
        gc.disable()
        latencies = []
        for _ in range(self.n_runs):
            t0 = time.perf_counter()
            try:
                model.predict(batch)
            except Exception:
                break
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
        gc.enable()

        if not latencies:
            return LatencyProfile(
                batch_size=batch_size,
                mean_ms=float("inf"),
                median_ms=float("inf"),
                p95_ms=float("inf"),
                p99_ms=float("inf"),
                std_ms=0.0,
                throughput_samples_per_s=0.0,
            )

        arr = np.array(latencies)
        median_ms = float(np.median(arr))
        throughput = (batch_size / (median_ms / 1000)) if median_ms > 0 else 0.0

        return LatencyProfile(
            batch_size=batch_size,
            mean_ms=float(np.mean(arr)),
            median_ms=median_ms,
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            std_ms=float(np.std(arr)),
            throughput_samples_per_s=throughput,
        )

    def build_latency_table(
        self, results: Dict[str, InferenceProfile]
    ) -> pd.DataFrame:
        """Build latency comparison table."""
        rows = []
        for name, ip in results.items():
            for lp in ip.latency_profiles:
                rows.append({
                    "model": name,
                    "framework": ip.framework,
                    "batch_size": lp.batch_size,
                    "mean_ms": lp.mean_ms,
                    "median_ms": lp.median_ms,
                    "p95_ms": lp.p95_ms,
                    "throughput": lp.throughput_samples_per_s,
                })
        return pd.DataFrame(rows)

    def build_deployment_table(
        self, results: Dict[str, InferenceProfile]
    ) -> pd.DataFrame:
        """Build deployment feasibility table."""
        rows = []
        for name, ip in results.items():
            rows.append({
                "model": name,
                "framework": ip.framework,
                "model_size_kb": ip.model_size_kb,
                "single_sample_ms": ip.single_sample_ms,
                "meets_realtime": ip.meets_realtime,
                "meets_batch": ip.meets_batch,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("single_sample_ms", ascending=True)
        return df
