import torch
from tokenizer.tokenizer import TikTokenWrapper
from data.dataset import TextDataset
from model.embedding import TokenAndPositionEmbedding

def main():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = TikTokenWrapper("gpt2")
    vocab_size = tokenizer.vocab_size

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=32
    )

    x, _ = dataset[0]
    x = x.unsqueeze(0)  # add batch dimension

    embedding = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        embed_dim=128,
        context_length=32
    )

    out = embedding(x)

    print("Input shape:", x.shape)
    print("Embedding output shape:", out.shape)

if __name__ == "__main__":
    main()

