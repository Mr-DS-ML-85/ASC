"""Tests for prefetch module."""

import torch
import pytest
from asc.prefetch import StreamingPrefetchQueue, prefetch_kv_blocks


def test_prefetch_queue_creation():
    q = StreamingPrefetchQueue()
    stats = q.get_stats()
    assert stats["n_prefetched"] == 0
    assert stats["total_bytes_prefetched"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_blocks():
    q = StreamingPrefetchQueue()
    kv_cache = torch.randn(8, 2, 2048, 2, 64, device="cuda", dtype=torch.float16)
    block_indices = torch.tensor([0, 1, 2, 3], device="cuda")

    q.prefetch_blocks(kv_cache, block_indices, block_size=256)
    q.synchronize()

    stats = q.get_stats()
    assert stats["n_prefetched"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_reset_stats():
    q = StreamingPrefetchQueue()
    kv_cache = torch.randn(4, 2, 1024, 2, 64, device="cuda", dtype=torch.float16)
    block_indices = torch.tensor([0, 1], device="cuda")

    q.prefetch_blocks(kv_cache, block_indices, block_size=256)
    q.synchronize()
    assert q.get_stats()["n_prefetched"] == 2

    q.reset_stats()
    assert q.get_stats()["n_prefetched"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_kv_blocks_function():
    kv_cache = torch.randn(4, 2, 1024, 2, 64, device="cuda", dtype=torch.float16)
    block_indices = torch.tensor([0, 1, 2], device="cuda")

    prefetch_kv_blocks(kv_cache, block_indices, block_size=256)
    torch.cuda.synchronize()
