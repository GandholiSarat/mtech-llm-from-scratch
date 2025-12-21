import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, context_length, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        input_ids: (B, T) LongTensor
        """
        B, T = input_ids.size()

        token_emb = self.token_embedding(input_ids)     # (B, T, C)
        pos_emb = self.position_embedding[:, :T, :]     # (1, T, C)

        x = token_emb + pos_emb
        x = self.dropout(x)

        return x

