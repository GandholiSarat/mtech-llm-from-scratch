import torch
from tokenizer.tokenizer import TikTokenWrapper
from data.dataset import TextDataset
from model.embedding import TokenAndPositionEmbedding
from model.attention import CausalSelfAttention

def main():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = TikTokenWrapper("gpt2")
    vocab_size = tokenizer.vocab_size

    dataset = TextDataset(text, tokenizer, context_length=32)
    x, _ = dataset[0]
    x = x.unsqueeze(0)  # (B=1, T)

    embedding = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        embed_dim=128,
        context_length=32
    )

    attn = CausalSelfAttention(embed_dim=128)

    x_emb = embedding(x)
    out = attn(x_emb)

    print("Embedding shape:", x_emb.shape)
    print("Attention output shape:", out.shape)

if __name__ == "__main__":
    main()

