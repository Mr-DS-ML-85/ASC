#!/usr/bin/env python3
"""Example: PAR prediction + DRAM bandwidth benchmark."""

import torch
from asc import PredictiveAttentionRouter, DRAMBenchmark


def main():
    print("=== ASC Example: PAR + DRAM Benchmark ===\n")

    # 1. Create PAR
    par = PredictiveAttentionRouter(d_model=1024, n_blocks=32, top_k=8)
    print(f"PAR params: {par.get_param_count():,}")

    # 2. Predict blocks
    hidden = torch.randn(1, 1024)
    top_indices, logits = par.predict_blocks(hidden)
    print(f"Predicted blocks: {top_indices[0].tolist()}")

    # 3. DRAM bandwidth benchmark
    print("\nDRAM Bandwidth:")
    for size_mb in [64, 128, 256]:
        seq = DRAMBenchmark.measure_bandwidth(size_mb, "sequential")
        scat = DRAMBenchmark.measure_bandwidth(size_mb, "scattered")
        penalty = (scat["bandwidth_gbps"] - seq["bandwidth_gbps"]) / seq["bandwidth_gbps"] * 100
        print(f"  {size_mb}MB: seq={seq['bandwidth_gbps']:.1f} GB/s | "
              f"scat={scat['bandwidth_gbps']:.1f} GB/s ({penalty:.0f}%)")


if __name__ == "__main__":
    main()
