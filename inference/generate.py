import argparse
import torch
import torch.nn.functional as F

from tokenizer.tokenizer import TikTokenWrapper
from model.gpt_model import GPTModel

# --------------------
# Default config (must match trained model)
# --------------------
DEVICE = "cpu"

EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
CONTEXT_LEN = 128

MODEL_PATH = "trained_small_gpt.pt"


def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    greedy=False,
):
    """
    input_ids: (1, T)
    """
    model.eval()

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -CONTEXT_LEN:]

        with torch.no_grad():
            logits = model(input_cond)

        logits = logits[:, -1, :]  # last token

        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

        if greedy:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


def main():
    parser = argparse.ArgumentParser(
        description="GPT-style text generation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt text"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (ignored if greedy)"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )

    args = parser.parse_args()

    tokenizer = TikTokenWrapper("gpt2")

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)],
        dtype=torch.long
    ).to(DEVICE)

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        greedy=args.greedy
    )

    output_text = tokenizer.decode(output_ids[0].tolist())

    print("\n--- Generated Text ---\n")
    print(output_text)


if __name__ == "__main__":
    main()

