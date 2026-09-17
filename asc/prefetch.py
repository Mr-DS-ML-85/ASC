"""Streaming Prefetch Queue — CUDA async prefetch for KV cache blocks."""

import torch
from typing import Optional


class StreamingPrefetchQueue:
    """
    CUDA prefetch queue using high-priority stream.

    Copies predicted KV blocks into L2 cache via async memory operations,
    overlapping with attention computation.

    Args:
        priority: Stream priority (lower = higher priority, default -1)
    """

    def __init__(self, priority: int = -1):
        self.priority = priority
        self.stream = torch.cuda.Stream(priority=priority)
        self.n_prefetched = 0
        self.total_bytes_prefetched = 0

    def prefetch_blocks(
        self,
        kv_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_size: int,
    ):
        """
        Prefetch KV blocks into L2 cache on high-priority stream.

        Args:
            kv_cache: [n_layers, 2, seq_len, n_heads, head_dim] — K and V cache
            block_indices: [top_k] block indices to prefetch
            block_size: Number of tokens per block
        """
        with torch.cuda.stream(self.stream):
            for idx in block_indices:
                start = idx.item() * block_size
                end = min(start + block_size, kv_cache.shape[2])
                if start < kv_cache.shape[2]:
                    # Touch the memory to bring into L2 cache
                    _ = kv_cache[:, :, start:end, :, :].clone()
                    self.n_prefetched += 1
                    self.total_bytes_prefetched += (
                        kv_cache[:, :, start:end, :, :].numel() * kv_cache.element_size()
                    )

    def synchronize(self):
        """Wait for all pending prefetch operations to complete."""
        torch.cuda.current_stream().wait_stream(self.stream)

    def reset_stats(self):
        """Reset prefetch statistics."""
        self.n_prefetched = 0
        self.total_bytes_prefetched = 0

    def get_stats(self) -> dict:
        """Return prefetch statistics."""
        return {
            "n_prefetched": self.n_prefetched,
            "total_bytes_prefetched": self.total_bytes_prefetched,
            "total_mb_prefetched": self.total_bytes_prefetched / 1e6,
        }


def prefetch_kv_blocks(
    kv_cache: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    stream: Optional[torch.cuda.Stream] = None,
):
    """
    Convenience function to prefetch KV blocks.

    Args:
        kv_cache: [n_layers, 2, seq_len, n_heads, head_dim]
        block_indices: [top_k] block indices
        block_size: Tokens per block
        stream: Optional CUDA stream (creates high-priority if None)
    """
    if stream is None:
        stream = torch.cuda.Stream(priority=-1)

    with torch.cuda.stream(stream):
        for idx in block_indices:
            start = idx.item() * block_size
            end = min(start + block_size, kv_cache.shape[2])
            if start < kv_cache.shape[2]:
                _ = kv_cache[:, :, start:end, :, :].clone()
