"""Attention Stream Cache (ASC) — Reducing DRAM Bandwidth Saturation in LLM Attention Layers."""

__version__ = "0.1.0"
__author__ = "Irfan Mahir"

from .par import PredictiveAttentionRouter
from .prefetch import StreamingPrefetchQueue, prefetch_kv_blocks
from .benchmark import DRAMBenchmark, ModelBenchmark
from .models import ModelConfig, get_model_config

__all__ = [
    "PredictiveAttentionRouter",
    "StreamingPrefetchQueue",
    "prefetch_kv_blocks",
    "DRAMBenchmark",
    "ModelBenchmark",
    "ModelConfig",
    "get_model_config",
]
