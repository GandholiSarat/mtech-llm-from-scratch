import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer.tokenizer import TikTokenWrapper
from data.dataset import TextDataset
from model.gpt_model import GPTModel

# --------------------
# Training config
# --------------------
DEVICE = "cpu"
BATCH_SIZE = 2
CONTEXT_LEN = 128
EPOCHS = 3
LR = 3e-4

EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4

# --------------------
# Training loop
# --------------------
def main():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = TikTokenWrapper("gpt2")

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=CONTEXT_LEN
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)  # (B, T, V)

            # Flatten for loss
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                y.view(B * T)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            '''print("x min/max:", x.min().item(), x.max().item())
            print("y min/max:", y.min().item(), y.max().item())
            print("vocab size:", tokenizer.vocab_size)
            print("logits mean/std:",
                    logits.mean().item(),
                    logits.std().item())

            break
            '''

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed | Avg loss {avg_loss:.4f}")
    torch.save(model.state_dict(), "trained_small_gpt.pt")
    print("Model saved.")


if __name__ == "__main__":
    main()

