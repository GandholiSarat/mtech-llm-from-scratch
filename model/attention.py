import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.size()

        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)  # (B, T, C)
        V = self.v_proj(x)  # (B, T, C)

        # Compute attention scores
        attn_scores = Q @ K.transpose(-2, -1)  # (B, T, T)
        attn_scores = attn_scores / math.sqrt(C)

        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T, T)

        # Attention output
        out = attn_weights @ V  # (B, T, C)

        return self.out_proj(out)  # (B, T, C)

