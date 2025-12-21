import math
import torch
import torch.nn as nn


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_context=1024, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # GPT-2 style: single QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout (applied AFTER attention projection)
        self.dropout = nn.Dropout(dropout)

        # Precomputed causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_context, max_context))
            .view(1, 1, max_context, max_context)
        )

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.size()

        # Project once to get Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Attention output
        attn_out = attn_weights @ v  # (B, H, T, D)

        # Recombine heads
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # Output projection + dropout
        out = self.out_proj(attn_out)
        out = self.dropout(out)

        return out

