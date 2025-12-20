import torch.nn as nn
from model.multihead_attention import MultiHeadCausalSelfAttention
from model.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadCausalSelfAttention(embed_dim, num_heads)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        # Pre-Norm + Residual (Attention)
        x = x + self.attn(self.ln1(x))

        # Pre-Norm + Residual (FFN)
        x = x + self.ffn(self.ln2(x))

        return x

