import torch
import torch.nn as nn


class GPT2GELU(nn.Module):
    """
    GPT-2 uses the tanh-based approximate GELU.
    """
    def forward(self, x):
        return 0.5 * x * (
            1.0 + torch.tanh(
                0.7978845608 * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            GPT2GELU(),                    # 🔥 critical
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

