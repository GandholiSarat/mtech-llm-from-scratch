import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer.tokenizer import TikTokenWrapper
from data.dataset import TextDataset
from tools.load_gpt2_weights import load_gpt2_weights
from tools.freeze import freeze_gpt2_except_ln_and_head

# --------------------
# Config
# --------------------
DEVICE = "cpu"
BATCH_SIZE = 2
CONTEXT_LEN = 128
EPOCHS = 2
LR = 5e-5   #  small LR for finetuning

# --------------------
# Training
# --------------------
def main():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = TikTokenWrapper("gpt2")

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=CONTEXT_LEN,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("Loading GPT-2 pretrained weights...")
    model = load_gpt2_weights().to(DEVICE)

    print("Freezing GPT-2 weights...")
    freeze_gpt2_except_ln_and_head(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            B, T, V = logits.shape

            loss = criterion(
                logits.view(B * T, V),
                y.view(B * T)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed | Avg loss {avg_loss:.4f}")

    torch.save(model.state_dict(), "gpt2_finetuned.pt")
    print("Fine-tuned model saved.")


if __name__ == "__main__":
    main()

