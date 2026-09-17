"""Tests for PAR module."""

import torch
import pytest
from asc.par import PredictiveAttentionRouter


def test_par_creation():
    par = PredictiveAttentionRouter(d_model=1024, n_blocks=32, top_k=8)
    assert par.d_model == 1024
    assert par.n_blocks == 32
    assert par.top_k == 8


def test_par_forward():
    par = PredictiveAttentionRouter(d_model=1024, n_blocks=32, top_k=8)
    hidden = torch.randn(1, 1024)
    top_indices, logits = par(hidden)

    assert top_indices.shape == (1, 8)
    assert logits.shape == (1, 32)
    assert top_indices.min() >= 0
    assert top_indices.max() < 32


def test_par_predict_blocks():
    par = PredictiveAttentionRouter(d_model=512, n_blocks=16, top_k=4)
    hidden = torch.randn(1, 512)
    top_indices, logits = par.predict_blocks(hidden)

    assert top_indices.shape == (1, 4)
    assert logits.shape == (1, 16)


def test_par_param_count():
    par = PredictiveAttentionRouter(d_model=1024, n_blocks=32, top_k=8)
    count = par.get_param_count()
    # 1024*2048 + 2048 + 2048*2048 + 2048 + 32*2048 + 32
    expected = 1024 * 2048 + 2048 + 2048 * 2048 + 2048 + 32 * 2048 + 32
    assert count == expected


def test_par_from_model_config():
    par = PredictiveAttentionRouter.from_model_config(
        d_model=1024, seq_len=8192, block_size=256, top_k=8
    )
    assert par.n_blocks == 32
    assert par.top_k == 8


def test_par_deterministic():
    par = PredictiveAttentionRouter(d_model=256, n_blocks=8, top_k=2)
    hidden = torch.randn(1, 256)
    idx1, _ = par(hidden)
    idx2, _ = par(hidden)
    assert torch.equal(idx1, idx2)
