import torch.nn as nn
from model.multihead_attention import MultiHeadCausalSelfAttention
from model.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_context, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-5)  
        self.attn = MultiHeadCausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_context=max_context,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-5) 
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
