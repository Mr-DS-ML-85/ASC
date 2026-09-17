"""Benchmarking utilities for ASC — DRAM bandwidth and model decode measurements."""

import time
import numpy as np
import torch
from typing import Optional
from .models import ModelConfig, get_model_config


class DRAMBenchmark:
    """
    Measure DRAM bandwidth with different access patterns.

    Tests sequential vs scattered reads to validate the paper's core thesis:
    scattered DRAM access suffers 80-84% bandwidth penalty.
    """

    @staticmethod
    def measure_bandwidth(
        size_mb: int,
        mode: str = "sequential",
        n_iter: int = 20,
        device: str = "cpu",
    ) -> dict:
        """
        Measure DRAM read bandwidth.

        Args:
            size_mb: Data size in MB
            mode: 'sequential', 'scattered', or 'block_scattered'
            n_iter: Number of iterations
            device: 'cpu' or 'cuda'

        Returns:
            dict with time_ms, bandwidth_gbps
        """
        n_bytes = size_mb * 1024 * 1024
        n_floats = n_bytes // 4
        data = torch.randn(n_floats, device=device, dtype=torch.float32)

        # Warmup
        _ = data.sum()
        if device == "cuda":
            torch.cuda.synchronize()

        if mode == "sequential":
            times = []
            for _ in range(n_iter):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = data.sum()
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        elif mode == "scattered":
            indices = torch.randperm(n_floats, device=device)[: n_floats // 4]
            times = []
            for _ in range(n_iter):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = data[indices].sum()
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        elif mode == "block_scattered":
            block_size = 256
            n_blocks = n_floats // block_size
            top_k = n_blocks // 4
            block_indices = torch.randperm(n_blocks, device=device)[:top_k]
            offsets = block_indices * block_size
            indices = torch.cat([offsets + i for i in range(block_size)]).sort().values
            times = []
            for _ in range(n_iter):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = data[indices].sum()
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        avg_ms = np.mean(times)
        bandwidth = n_bytes / (avg_ms / 1000) / 1e9

        return {
            "size_mb": size_mb,
            "mode": mode,
            "time_ms": avg_ms,
            "bandwidth_gbps": bandwidth,
        }

    @classmethod
    def run_full_benchmark(cls, sizes_mb=None, device="cpu") -> list:
        """
        Run complete DRAM bandwidth benchmark.

        Returns list of results for sequential, scattered, block_scattered at each size.
        """
        if sizes_mb is None:
            sizes_mb = [16, 32, 64, 128, 256, 512]

        results = []
        for size_mb in sizes_mb:
            for mode in ["sequential", "scattered", "block_scattered"]:
                r = cls.measure_bandwidth(size_mb, mode=mode, device=device)
                penalty = 0.0
                if mode != "sequential":
                    seq_r = cls.measure_bandwidth(size_mb, mode="sequential", device=device)
                    penalty = (r["bandwidth_gbps"] - seq_r["bandwidth_gbps"]) / seq_r["bandwidth_gbps"]
                r["penalty_vs_sequential"] = penalty
                results.append(r)
        return results


class ModelBenchmark:
    """
    Benchmark LLM decode speed with llama.cpp.

    Measures single-token decode time at different context lengths
    to identify when KV cache becomes a bottleneck.
    """

    def __init__(self, model_config: Optional[ModelConfig] = None, model_name: Optional[str] = None):
        if model_config is not None:
            self.config = model_config
        elif model_name is not None:
            self.config = get_model_config(model_name)
        else:
            raise ValueError("Must provide model_config or model_name")

    def measure_decode(
        self,
        n_ctx: int,
        n_threads: int = 8,
        n_gpu_layers: int = 99,
        n_warmup: int = 5,
        n_measure: int = 20,
    ) -> dict:
        """
        Measure single-token decode time.

        Returns dict with avg_ms, median_ms, p95_ms, tokens_per_sec, kv_size_mb.
        """
        from llama_cpp import Llama

        llm = Llama(
            model_path=self.config.path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        fill_len = int(n_ctx * 0.75)
        prompt = "The quick brown fox jumps over the lazy dog. " * (fill_len // 48 + 1)
        prompt = prompt[:fill_len]

        # Prefill
        llm(prompt, max_tokens=1, temperature=0.0)

        # Warmup
        for _ in range(n_warmup):
            llm(prompt, max_tokens=1, temperature=0.0)

        # Measure
        times = []
        for _ in range(n_measure):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm(prompt, max_tokens=1, temperature=0.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times_arr = np.array(times)
        kv_mb = self.config.kv_size_mb(fill_len)

        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "model": self.config.name,
            "n_ctx": n_ctx,
            "fill_len": fill_len,
            "avg_ms": float(np.mean(times_arr)),
            "median_ms": float(np.median(times_arr)),
            "p95_ms": float(np.percentile(times_arr, 95)),
            "tokens_per_sec": 1000.0 / np.mean(times_arr),
            "kv_size_mb": kv_mb,
        }

    def run_sweep(
        self,
        ctx_sizes=None,
        n_threads: int = 8,
        n_gpu_layers: int = 99,
    ) -> list:
        """Run decode benchmark across multiple context sizes."""
        if ctx_sizes is None:
            ctx_sizes = [512, 1024, 2048, 4096, 8192]

        results = []
        for n_ctx in ctx_sizes:
            try:
                r = self.measure_decode(n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
                results.append(r)
            except Exception as e:
                results.append({"model": self.config.name, "n_ctx": n_ctx, "error": str(e)})
        return results
