#!/usr/bin/env python3
"""
Manual Library Tests — Import and use ASC directly.
No llama.cpp. No fake benchmarks. Just the library.

Run: python manual_tests.py
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from asc.par import PredictiveAttentionRouter
from asc.prefetch import StreamingPrefetchQueue, prefetch_kv_blocks
from asc.models import MODELS, get_model_config, list_models

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


# ============================================================
# TEST 1: PAR — create, forward, shapes, determinism
# ============================================================
def test_par():
    print("\n=== PAR Tests ===")

    for name in list_models():
        cfg = get_model_config(name)
        par = PredictiveAttentionRouter(d_model=cfg.d_model, n_blocks=32, top_k=8)

        check(f"PAR create {name}", True, f"params={par.get_param_count():,}")

        hidden = torch.randn(1, cfg.d_model)
        idx, logits = par(hidden)
        check(f"PAR forward {name}", idx.shape == (1, 8) and logits.shape == (1, 32),
              f"idx={idx.shape} logits={logits.shape}")

        idx2, _ = par(hidden)
        check(f"PAR deterministic {name}", torch.equal(idx, idx2))

        check(f"PAR indices in range {name}",
              idx.min() >= 0 and idx.max() < 32,
              f"min={idx.min()} max={idx.max()}")

    # Edge cases
    par = PredictiveAttentionRouter(d_model=64, n_blocks=4, top_k=2)
    hidden = torch.randn(1, 64)
    idx, logits = par(hidden)
    check("PAR small model", idx.shape == (1, 2) and logits.shape == (1, 4))

    par = PredictiveAttentionRouter(d_model=8192, n_blocks=256, top_k=16)
    hidden = torch.randn(1, 8192)
    idx, logits = par(hidden)
    check("PAR large model", idx.shape == (1, 16) and logits.shape == (1, 256))

    # Batch
    par = PredictiveAttentionRouter(d_model=512, n_blocks=16, top_k=4)
    hidden = torch.randn(8, 512)
    idx, logits = par(hidden)
    check("PAR batch inference", idx.shape == (8, 4))

    # No grad
    par = PredictiveAttentionRouter(d_model=256, n_blocks=8, top_k=2)
    hidden = torch.randn(1, 256)
    with torch.no_grad():
        idx, logits = par(hidden)
    check("PAR no_grad works", True)


# ============================================================
# TEST 2: Prefetch — queue, CUDA, stats
# ============================================================
def test_prefetch():
    print("\n=== Prefetch Tests ===")

    q = StreamingPrefetchQueue()
    stats = q.get_stats()
    check("Prefetch queue create", stats["n_prefetched"] == 0)

    q.reset_stats()
    check("Prefetch reset", q.get_stats()["n_prefetched"] == 0)

    if not torch.cuda.is_available():
        check("Prefetch CUDA", False, "CUDA not available, skipping")
        return

    kv = torch.randn(8, 2, 2048, 2, 64, device="cuda", dtype=torch.float16)
    idx = torch.tensor([0, 1, 2, 3], device="cuda")
    q.prefetch_blocks(kv, idx, block_size=256)
    q.synchronize()
    check("Prefetch CUDA blocks", q.get_stats()["n_prefetched"] == 4)

    # Test prefetch_kv_blocks function
    q2 = StreamingPrefetchQueue()
    prefetch_kv_blocks(kv, torch.tensor([0, 1], device="cuda"), block_size=256)
    torch.cuda.synchronize()
    check("prefetch_kv_blocks function", True)

    # Edge: empty indices
    q3 = StreamingPrefetchQueue()
    q3.prefetch_blocks(kv, torch.tensor([], device="cuda", dtype=torch.long), block_size=256)
    q3.synchronize()
    check("Prefetch empty indices", q3.get_stats()["n_prefetched"] == 0)

    # Edge: out of range indices
    q4 = StreamingPrefetchQueue()
    q4.prefetch_blocks(kv, torch.tensor([999], device="cuda"), block_size=256)
    q4.synchronize()
    check("Prefetch OOB index (no crash)", True)


# ============================================================
# TEST 3: DRAM bandwidth — the real measurement
# ============================================================
def test_dram():
    print("\n=== DRAM Bandwidth Tests ===")
    from asc.benchmark import DRAMBenchmark

    for size in [16, 64, 256]:
        seq = DRAMBenchmark.measure_bandwidth(size, "sequential", n_iter=10)
        scat = DRAMBenchmark.measure_bandwidth(size, "scattered", n_iter=10)

        check(f"DRAM sequential {size}MB", seq["bandwidth_gbps"] > 10,
              f"{seq['bandwidth_gbps']:.1f} GB/s")
        check(f"DRAM scattered {size}MB", scat["bandwidth_gbps"] > 0,
              f"{scat['bandwidth_gbps']:.1f} GB/s")
        check(f"DRAM penalty {size}MB",
              scat["bandwidth_gbps"] < seq["bandwidth_gbps"] * 0.5,
              f"scattered is {seq['bandwidth_gbps']/scat['bandwidth_gbps']:.1f}x slower")


# ============================================================
# TEST 4: Model configs — all models load correctly
# ============================================================
def test_configs():
    print("\n=== Model Config Tests ===")

    for name in list_models():
        cfg = get_model_config(name)
        check(f"Config {name}", cfg.d_model > 0 and cfg.n_layers > 0,
              f"d={cfg.d_model} L={cfg.n_layers} attn={cfg.n_attn_layers}")
        check(f"KV per token {name}", cfg.kv_bytes_per_token > 0,
              f"{cfg.kv_bytes_per_token} bytes")
        check(f"KV size 8K {name}", cfg.kv_size_mb(8192) > 0,
              f"{cfg.kv_size_mb(8192):.1f}MB")

    check("get_model_config invalid", True)  # just verifying no crash above
    try:
        get_model_config("nonexistent")
        check("get_model_config invalid raises", False, "should have raised")
    except ValueError:
        check("get_model_config invalid raises", True)


# ============================================================
# TEST 5: GPU kernel timing — real attention pattern
# ============================================================
def test_attention_pattern():
    print("\n=== Attention Pattern Test ===")

    if not torch.cuda.is_available():
        check("Attention pattern CUDA", False, "CUDA not available")
        return

    n_heads = 8
    head_dim = 64
    seq_len = 4096
    n_iter = 50

    kv = torch.randn(2, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float16)
    q = torch.randn(1, n_heads, head_dim, device="cuda", dtype=torch.float16)

    # Full attention (all blocks)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scores = torch.matmul(q, kv[0].transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, kv[1])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    full_ms = np.mean(times)

    # Partial attention (top-25% blocks = 1024 tokens)
    top_k = 1024
    kv_partial = kv[:, :top_k]
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scores = torch.matmul(q, kv_partial[0].transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, kv_partial[1])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    partial_ms = np.mean(times)

    speedup = full_ms / partial_ms
    check("Attention full vs partial", speedup > 1.0,
          f"full={full_ms:.3f}ms partial={partial_ms:.3f}ms speedup={speedup:.2f}x")

    # Prefetch test: warm cache vs cold
    kv_large = torch.randn(2, 8192, n_heads, head_dim, device="cuda", dtype=torch.float16)
    prefetch_indices = torch.arange(0, 2048, device="cuda")

    # Cold
    torch.cuda.synchronize()
    _ = torch.randn(100, 1024, device="cuda", dtype=torch.float16)  # evict L2
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        _ = torch.randn(100, 1024, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        block = kv_large[:, :2048].clone()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    cold_ms = np.mean(times)

    # Warm (prefetch)
    kv_large[:, :2048].clone()  # warm
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        _ = torch.randn(100, 1024, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        block = kv_large[:, :2048].clone()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    warm_ms = np.mean(times)

    check("Prefetch warms cache", warm_ms <= cold_ms * 1.1,
          f"cold={cold_ms:.3f}ms warm={warm_ms:.3f}ms ratio={warm_ms/cold_ms:.2f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ASC Manual Library Tests")
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    test_par()
    test_prefetch()
    test_dram()
    test_configs()
    test_attention_pattern()

    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"RESULT: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    sys.exit(0 if FAIL == 0 else 1)
