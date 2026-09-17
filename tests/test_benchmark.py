"""Tests for benchmark module."""

import pytest
from asc.benchmark import DRAMBenchmark, ModelBenchmark
from asc.models import ModelConfig, get_model_config, list_models


def test_dram_benchmark_sequential():
    r = DRAMBenchmark.measure_bandwidth(16, mode="sequential", n_iter=5)
    assert r["size_mb"] == 16
    assert r["mode"] == "sequential"
    assert r["bandwidth_gbps"] > 0


def test_dram_benchmark_scattered():
    r = DRAMBenchmark.measure_bandwidth(16, mode="scattered", n_iter=5)
    assert r["mode"] == "scattered"
    assert r["bandwidth_gbps"] > 0


def test_dram_scattered_penalty():
    seq = DRAMBenchmark.measure_bandwidth(64, mode="sequential", n_iter=10)
    scat = DRAMBenchmark.measure_bandwidth(64, mode="scattered", n_iter=10)
    # Scattered should be significantly slower
    assert scat["bandwidth_gbps"] < seq["bandwidth_gbps"]


def test_model_config():
    cfg = ModelConfig(
        name="test", path="/tmp/test.gguf",
        d_model=512, n_layers=12, n_attn_layers=12,
        n_kv_heads=2, head_dim=64,
    )
    assert cfg.kv_bytes_per_token == 2 * 12 * 2 * 2 * 64 * 2
    assert cfg.n_blocks(2048) == 8


def test_get_model_config():
    cfg = get_model_config("falcon-h1-0.5b")
    assert cfg.name == "Falcon-H1-0.5B"
    assert cfg.d_model == 1024


def test_get_model_config_invalid():
    with pytest.raises(ValueError):
        get_model_config("nonexistent-model")


def test_list_models():
    models = list_models()
    assert "falcon-h1-0.5b" in models
    assert "qwen2.5-0.5b" in models
    assert "ornith-1.0-9b" in models


def test_model_benchmark_creation():
    b = ModelBenchmark(model_name="falcon-h1-0.5b")
    assert b.config.name == "Falcon-H1-0.5B"
