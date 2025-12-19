import torch
import torch.nn as nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

    def forward(self, x):
        """
        x: (B, T) where
        B = batch size
        T = sequence length
        """
        B, T = x.size()

        # Token embeddings
        token_emb = self.token_embedding(x)  # (B, T, C)

        # Position indices
        positions = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(positions)  # (T, C)

        # Broadcast position embeddings across batch
        return token_emb + pos_emb        # (B, T, C)

