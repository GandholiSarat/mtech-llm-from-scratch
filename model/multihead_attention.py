import torch
import torch.nn as nn
import math

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_context=1024):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # GPT-2 style: single QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Pre-computed causal mask (registered buffer)
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

        qkv = self.qkv_proj(x)                  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = attn_weights @ v                  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

