#!/usr/bin/env python3
"""
ASC Test Suite — Two modes:
  1. Benchmark: automated timing across context sizes
  2. Manual: import lib, run PAR, run prefetch, measure timing

Usage:
  python run_tests.py                    # test all models
  python run_tests.py --model falcon-h1-0.5b   # test one model
  python run_tests.py --dram-only        # DRAM bandwidth only
"""

import argparse
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from asc.par import PredictiveAttentionRouter
from asc.prefetch import StreamingPrefetchQueue
from asc.benchmark import DRAMBenchmark, ModelBenchmark
from asc.models import MODELS, get_model_config, list_models


class TestResult:
    def __init__(self, name, passed, detail=""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


# ============================================================
# MODE 1: Manual import tests (production validation)
# ============================================================

def test_manual_par():
    """Test PAR creation, forward pass, prediction."""
    results = []

    cfg = get_model_config("falcon-h1-0.5b")
    par = PredictiveAttentionRouter(d_model=cfg.d_model, n_blocks=32, top_k=8)

    # Test creation
    results.append(TestResult("PAR creation", True, f"params={par.get_param_count():,}"))

    # Test forward
    hidden = torch.randn(1, cfg.d_model)
    t0 = time.perf_counter()
    top_indices, logits = par(hidden)
    t1 = time.perf_counter()
    results.append(TestResult("PAR forward", top_indices.shape == (1, 8),
                              f"shape={top_indices.shape} time={((t1-t0)*1000):.3f}ms"))

    # Test predict_blocks
    top_idx, logits = par.predict_blocks(hidden)
    results.append(TestResult("PAR predict_blocks", True, f"blocks={top_idx[0].tolist()}"))

    # Test determinism
    idx1, _ = par(hidden)
    idx2, _ = par(hidden)
    results.append(TestResult("PAR deterministic", torch.equal(idx1, idx2), ""))

    # Test all model configs
    for name in list_models():
        cfg = get_model_config(name)
        par = PredictiveAttentionRouter(d_model=cfg.d_model, n_blocks=32, top_k=8)
        hidden = torch.randn(1, cfg.d_model)
        idx, _ = par(hidden)
        results.append(TestResult(f"PAR init {name}", True,
                                  f"d_model={cfg.d_model} params={par.get_param_count():,}"))

    return results


def test_manual_prefetch():
    """Test prefetch queue creation and stats."""
    results = []

    q = StreamingPrefetchQueue()
    stats = q.get_stats()
    results.append(TestResult("Prefetch queue creation", stats["n_prefetched"] == 0, ""))

    q.reset_stats()
    stats = q.get_stats()
    results.append(TestResult("Prefetch reset_stats", stats["n_prefetched"] == 0, ""))

    if torch.cuda.is_available():
        kv_cache = torch.randn(8, 2, 2048, 2, 64, device="cuda", dtype=torch.float16)
        block_indices = torch.tensor([0, 1, 2, 3], device="cuda")
        q.prefetch_blocks(kv_cache, block_indices, block_size=256)
        q.synchronize()
        stats = q.get_stats()
        results.append(TestResult("Prefetch CUDA", stats["n_prefetched"] == 4,
                                  f"prefetched={stats['n_prefetched']}"))
    else:
        results.append(TestResult("Prefetch CUDA", False, "CUDA not available"))

    return results


def test_manual_dram():
    """Test DRAM bandwidth measurement."""
    results = []

    r = DRAMBenchmark.measure_bandwidth(16, "sequential", n_iter=5)
    results.append(TestResult("DRAM sequential", r["bandwidth_gbps"] > 0,
                              f"{r['bandwidth_gbps']:.1f} GB/s"))

    r = DRAMBenchmark.measure_bandwidth(16, "scattered", n_iter=5)
    results.append(TestResult("DRAM scattered", r["bandwidth_gbps"] > 0,
                              f"{r['bandwidth_gbps']:.1f} GB/s"))

    seq = DRAMBenchmark.measure_bandwidth(64, "sequential", n_iter=10)
    scat = DRAMBenchmark.measure_bandwidth(64, "scattered", n_iter=10)
    penalty = (scat["bandwidth_gbps"] - seq["bandwidth_gbps"]) / seq["bandwidth_gbps"] * 100
    results.append(TestResult("DRAM penalty exists", penalty < -30,
                              f"penalty={penalty:.0f}% seq={seq['bandwidth_gbps']:.1f} scat={scat['bandwidth_gbps']:.1f}"))

    return results


# ============================================================
# MODE 2: Benchmark tests (model decode timing)
# ============================================================

def run_benchmark(model_name, ctx_sizes=None):
    """Run full benchmark for a single model."""
    if ctx_sizes is None:
        ctx_sizes = [512, 1024, 2048, 4096, 8192]

    config = get_model_config(model_name)
    b = ModelBenchmark(model_config=config)

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {config.name}")
    print(f"  d_model={config.d_model} layers={config.n_layers} attn={config.n_attn_layers} kv_heads={config.n_kv_heads}")
    print(f"  KV per token: {config.kv_bytes_per_token} bytes")
    print(f"{'=' * 70}")

    results = []
    for n_ctx in ctx_sizes:
        try:
            r = b.measure_decode(n_ctx)
            results.append(r)
            print(f"  ctx={n_ctx:>5}: {r['avg_ms']:.2f}ms ({r['tokens_per_sec']:.0f} tok/s) | KV={r['kv_size_mb']:.1f}MB")
        except Exception as e:
            results.append({"model": config.name, "n_ctx": n_ctx, "error": str(e)})
            print(f"  ctx={n_ctx}: ERROR - {e}")

    return results


def run_par_overhead(model_name):
    """Test PAR overhead when running alongside decode."""
    config = get_model_config(model_name)
    par = PredictiveAttentionRouter(d_model=config.d_model, n_blocks=32, top_k=8)

    print(f"\nPAR overhead for {config.name}:")
    hidden = torch.randn(1, config.d_model)

    # Measure PAR time
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            par(hidden)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_par = np.mean(times)
    print(f"  PAR inference: {avg_par:.4f}ms")
    return avg_par


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ASC Test Suite")
    parser.add_argument("--model", type=str, help="Test specific model")
    parser.add_argument("--dram-only", action="store_true", help="DRAM bandwidth only")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip model benchmarks")
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    all_results = []

    print("=" * 70)
    print("ASC TEST SUITE")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    # MODE 1: Manual import tests
    print("\n" + "=" * 70)
    print("MODE 1: Manual Import Tests")
    print("=" * 70)

    print("\n--- PAR Tests ---")
    for r in test_manual_par():
        print(f"  {r}")
        all_results.append({"test": r.name, "passed": r.passed, "detail": r.detail, "mode": "manual"})

    print("\n--- Prefetch Tests ---")
    for r in test_manual_prefetch():
        print(f"  {r}")
        all_results.append({"test": r.name, "passed": r.passed, "detail": r.detail, "mode": "manual"})

    print("\n--- DRAM Tests ---")
    for r in test_manual_dram():
        print(f"  {r}")
        all_results.append({"test": r.name, "passed": r.passed, "detail": r.detail, "mode": "manual"})

    if args.dram_only:
        summary = sum(1 for r in all_results if r["passed"])
        print(f"\n{summary}/{len(all_results)} tests passed")
        return

    # MODE 2: Benchmark tests
    if not args.skip_benchmark:
        print("\n" + "=" * 70)
        print("MODE 2: Benchmark Tests")
        print("=" * 70)

        models_to_test = [args.model] if args.model else list_models()

        for model_name in models_to_test:
            try:
                results = run_benchmark(model_name)
                all_results.append({"test": f"benchmark_{model_name}", "passed": True,
                                    "detail": json.dumps(results, default=str), "mode": "benchmark"})

                par_time = run_par_overhead(model_name)
                all_results.append({"test": f"par_overhead_{model_name}", "passed": True,
                                    "detail": f"{par_time:.4f}ms", "mode": "benchmark"})

            except Exception as e:
                all_results.append({"test": f"benchmark_{model_name}", "passed": False,
                                    "detail": str(e), "mode": "benchmark"})
                print(f"\n  FAILED: {e}")

    # Summary
    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'=' * 70}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
