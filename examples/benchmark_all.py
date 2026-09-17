#!/usr/bin/env python3
"""Example: Benchmark all models with ASC."""

from asc import ModelBenchmark
from asc.models import list_models


def main():
    print("=== ASC: Benchmark All Models ===\n")

    for model_name in list_models():
        print(f"\n--- {model_name} ---")
        try:
            b = ModelBenchmark(model_name=model_name)
            results = b.run_sweep(ctx_sizes=[1024, 2048, 4096, 8192])
            for r in results:
                if "error" in r:
                    print(f"  ctx={r['n_ctx']}: ERROR - {r['error']}")
                else:
                    print(f"  ctx={r['n_ctx']:>5}: {r['avg_ms']:.2f}ms ({r['tokens_per_sec']:.0f} tok/s) | KV={r['kv_size_mb']:.1f}MB")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
