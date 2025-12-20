import torch

from tokenizer.tokenizer import TikTokenWrapper
from model.gpt_model import GPTModel


def main():
    print("=== LLM From Scratch | Installation Test ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Using device: cpu")
    print("-" * 50)

    # --------------------
    # Minimal config
    # --------------------
    CONTEXT_LEN = 32
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2

    tokenizer = TikTokenWrapper("gpt2")

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )

    # --------------------
    # Dummy input
    # --------------------
    prompt = "Hello world"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    # Truncate if needed
    input_ids = input_ids[:, :CONTEXT_LEN]

    # --------------------
    # Forward pass
    # --------------------
    with torch.no_grad():
        logits = model(input_ids)

    print("Input shape:", input_ids.shape)
    print("Logits shape:", logits.shape)

    # --------------------
    # Verification checks
    # --------------------
    assert logits.shape[0] == 1
    assert logits.shape[1] == input_ids.shape[1]
    assert logits.shape[2] == tokenizer.vocab_size

    print("-" * 50)
    print("✔ Installation verified successfully")
    print("✔ Forward pass works")
    print("✔ No runtime errors")


if __name__ == "__main__":
    main()

