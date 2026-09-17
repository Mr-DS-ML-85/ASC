"""Model configurations for ASC benchmarking."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a model to benchmark with ASC."""
    name: str
    path: str
    d_model: int
    n_layers: int
    n_attn_layers: int
    n_kv_heads: int
    head_dim: int
    n_ssm_layers: int = 0
    block_size: int = 256
    max_ctx: int = 8192

    @property
    def kv_bytes_per_token(self) -> int:
        """KV cache bytes per token (K + V, FP16)."""
        return 2 * self.n_attn_layers * 2 * self.n_kv_heads * self.head_dim * 2

    @property
    def kv_mb_per_1k_tokens(self) -> float:
        """KV cache MB per 1K tokens."""
        return self.kv_bytes_per_token * 1000 / 1e6

    def kv_size_mb(self, seq_len: int) -> float:
        """KV cache size in MB for given sequence length."""
        return self.kv_bytes_per_token * seq_len / 1e6

    def n_blocks(self, seq_len: int) -> int:
        """Number of KV blocks for given sequence length."""
        return seq_len // self.block_size


# Pre-configured models
MODELS = {
    "falcon-h1-0.5b": ModelConfig(
        name="Falcon-H1-0.5B",
        path="/home/irfan/.infernix/models/falcon-h1-0.5b/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf",
        d_model=1024,
        n_layers=36,
        n_attn_layers=34,
        n_kv_heads=2,
        head_dim=64,
        n_ssm_layers=2,
    ),
    "gemma-3-1b": ModelConfig(
        name="Gemma-3-1B",
        path="/home/irfan/.infernix/models/gemma-3-1b/gemma-3-1b-it-Q4_K_M.gguf",
        d_model=2048,
        n_layers=26,
        n_attn_layers=26,
        n_kv_heads=8,
        head_dim=256,
        n_ssm_layers=0,
    ),
    "qwen2.5-0.5b": ModelConfig(
        name="Qwen2.5-0.5B",
        path="/home/irfan/.infernix/models/Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        d_model=896,
        n_layers=24,
        n_attn_layers=24,
        n_kv_heads=2,
        head_dim=64,
        n_ssm_layers=0,
    ),
    "ornith-1.0-9b": ModelConfig(
        name="Ornith-1.0-9B",
        path="/run/media/irfan/models/ornith-1.0-9b-Q4_K_M.gguf",
        d_model=4096,
        n_layers=32,
        n_attn_layers=8,
        n_kv_heads=4,
        head_dim=256,
        n_ssm_layers=24,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name."""
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name]


def list_models():
    """List all available model configurations."""
    return list(MODELS.keys())
