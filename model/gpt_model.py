import torch
import torch.nn as nn
from model.embedding import TokenAndPositionEmbedding
from model.transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        embed_dim,
        num_heads,
        num_layers
    ):
        super().__init__()

        self.context_length = context_length

        # Embeddings
        self.embeddings = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            context_length=context_length
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Final LayerNorm (GPT-2 style)
        self.ln_f = nn.LayerNorm(embed_dim)

        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Optional: weight tying (GPT-2 does this)
        self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(self, input_ids):
        """
        input_ids: (B, T)
        returns logits: (B, T, vocab_size)
        """
        B, T = input_ids.size()

        if T > self.context_length:
            raise ValueError("Sequence length exceeds context length")

        x = self.embeddings(input_ids)  # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)                # (B, T, C)
        logits = self.lm_head(x)        # (B, T, vocab)

        return logits

