import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        x: (B, T, C)
        """
        return self.net(x)

