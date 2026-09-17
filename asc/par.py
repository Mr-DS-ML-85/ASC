"""Predictive Attention Router (PAR) — Predicts top-k KV blocks for prefetching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveAttentionRouter(nn.Module):
    """
    PAR: 3-layer MLP that predicts which KV blocks will receive high attention.

    Architecture (from paper):
        Input:  hidden_state [batch, d_model]
        Layer1: Linear(d_model, 2048) + ReLU
        Layer2: Linear(2048, 2048) + ReLU
        Output: Linear(2048, n_blocks) -> top-k indices

    Args:
        d_model: Model hidden dimension (e.g. 1024 for Falcon-H1-0.5B)
        n_blocks: Number of KV blocks to predict over (seq_len // block_size)
        top_k: Number of blocks to select for prefetch (default: 8)
    """

    def __init__(self, d_model: int, n_blocks: int, top_k: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.top_k = top_k

        self.fc1 = nn.Linear(d_model, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, n_blocks)

    def forward(self, hidden_state: torch.Tensor):
        """
        Forward pass.

        Args:
            hidden_state: [batch, d_model] current hidden state

        Returns:
            top_indices: [batch, top_k] block indices to prefetch
            logits: [batch, n_blocks] raw block scores
        """
        x = F.relu(self.fc1(hidden_state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        _, top_indices = torch.topk(logits, self.top_k, dim=-1)
        return top_indices, logits

    @torch.no_grad()
    def predict_blocks(self, hidden_state: torch.Tensor):
        """Inference-only prediction (no gradients)."""
        return self(hidden_state)

    def get_param_count(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_model_config(cls, d_model: int, seq_len: int, block_size: int = 256, top_k: int = 8):
        """Create PAR from model configuration."""
        n_blocks = seq_len // block_size
        return cls(d_model=d_model, n_blocks=n_blocks, top_k=top_k)
