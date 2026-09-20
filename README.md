# Attention Stream Cache (ASC)

**Reducing DRAM Bandwidth Saturation in LLM Attention Layers**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22854905.svg)](https://doi.org/10.5281/zenodo.22854905)
[![License: MIT](https://img.shields.io/badge/License-MIT-00ff88.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![CUDA 11+](https://img.shields.io/badge/CUDA-11+-76b900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![GitHub stars](https://img.shields.io/github/stars/Mr-DS-ML-85/ASC?style=social&color=00ff88)](https://github.com/Mr-DS-ML-85/ASC/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/Mr-DS-ML-85/ASC?color=0088ff)](https://github.com/Mr-DS-ML-85/ASC/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/Mr-DS-ML-85/ASC?color=ff4444)](https://github.com/Mr-DS-ML-85/ASC/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Mr-DS-ML-85/ASC?color=ff8800)](https://github.com/Mr-DS-ML-85/ASC/pulls)

[Paper](https://doi.org/10.5281/zenodo.22810393) | [Docs](https://mr-ds-ml-85.github.io/ASC/) | [Installation](#installation) | [Quick Start](#quick-start) | [Models](#supported-models)

## Overview

Attention Stream Cache (ASC) is a software-hardware co-design that addresses DRAM bandwidth saturation in LLM inference. The core insight: attention's scattered KV cache reads suffer an **80-84% bandwidth penalty** compared to sequential reads.

ASC introduces:
- **Predictive Attention Router (PAR)**: Lightweight MLP that predicts which KV blocks will receive high attention
- **Streaming Prefetch Queue**: CUDA async prefetch to bring predicted blocks into L2 cache

### Key Findings

| Metric | Value |
|--------|-------|
| DRAM scattered-read penalty | 80-84% (5x slower than sequential) |
| ASC bandwidth reduction (KV in DRAM) | 35-40% |
| ASC throughput improvement | ~29% |
| PAR inference overhead | 0.09ms (13% of decode at 8K context) |

## Installation

```bash
# Clone
git clone https://github.com/Mr-DS-ML-85/ASC.git
cd asc

# Install Python library
pip install -e .

# Install with C++ bindings (optional)
pip install -e ".[cpp]"
cd cpp && mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- llama-cpp-python >= 0.2.0
- CUDA-capable GPU (for prefetch testing)

## Quick Start

```python
from asc import PredictiveAttentionRouter, DRAMBenchmark, ModelBenchmark

# 1. Create PAR and predict blocks
par = PredictiveAttentionRouter(d_model=1024, n_blocks=32, top_k=8)
import torch
hidden = torch.randn(1, 1024)
top_indices, logits = par.predict_blocks(hidden)
print(f"Predicted blocks: {top_indices[0].tolist()}")

# 2. Measure DRAM bandwidth
seq = DRAMBenchmark.measure_bandwidth(128, "sequential")
scat = DRAMBenchmark.measure_bandwidth(128, "scattered")
print(f"Sequential: {seq['bandwidth_gbps']:.1f} GB/s")
print(f"Scattered:  {scat['bandwidth_gbps']:.1f} GB/s")

# 3. Benchmark model decode
b = ModelBenchmark(model_name="falcon-h1-0.5b")
results = b.run_sweep(ctx_sizes=[1024, 2048, 4096])
for r in results:
    print(f"ctx={r['n_ctx']}: {r['tokens_per_sec']:.0f} tok/s")
```

## Supported Models

| Model | d_model | Layers | Attn Layers | KV Heads | Type |
|-------|---------|--------|-------------|----------|------|
| Falcon-H1-0.5B | 1024 | 36 | 34 | 2 | Hybrid SSM+Attn |
| Gemma-3-1B | 2048 | 26 | 26 | 8 | Pure Attention |
| Qwen2.5-0.5B | 896 | 24 | 24 | 2 | Pure Attention |
| Ornith-1.0-9B | 4096 | 32 | 8 | 4 | Hybrid SSM+Attn |

## C++ Usage

```cpp
#include "par.hpp"

int main() {
    asc::PredictiveAttentionRouter par(1024, 32, 8);
    std::vector<float> hidden(1024, 0.5f);
    auto result = par.predict(hidden.data());

    for (int idx : result.top_indices) {
        std::cout << "Block " << idx << std::endl;
    }
}
```

## Project Structure

```
asc/
  __init__.py      # Package exports
  par.py           # Predictive Attention Router
  prefetch.py      # Streaming Prefetch Queue
  benchmark.py     # DRAM and model benchmarks
  models.py        # Model configurations
cpp/
  include/         # C++ headers
  src/             # C++ implementations
  CMakeLists.txt   # Build system
tests/             # Test suite
examples/          # Usage examples
```

## Running Tests

```bash
# Manual import tests (51 tests, no llama.cpp)
python manual_tests.py

# Full benchmark suite
python run_tests.py

# With JSON output
python run_tests.py --json results.json

# DRAM tests only
python run_tests.py --dram-only
```

## Citation

```bibtex
@article{mahir2026asc,
  title={Attention Stream Cache: Reducing DRAM Bandwidth Saturation in LLM Attention Layers},
  author={Mahir, Irfan},
  year={2026},
  doi={10.5281/zenodo.22810393},
  url={https://doi.org/10.5281/zenodo.22810393}
}
```

## License

MIT
