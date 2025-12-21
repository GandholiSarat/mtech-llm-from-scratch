import argparse
import torch
import torch.nn.functional as F

from tokenizer.tokenizer import TikTokenWrapper
from model.gpt_model import GPTModel
from tools.load_gpt2_weights import load_gpt2_weights

# --------------------
# Inference config
# --------------------
DEVICE = "cpu"

# Small trained model config
SMALL_EMBED_DIM = 256
SMALL_NUM_HEADS = 4
SMALL_NUM_LAYERS = 4
SMALL_CONTEXT_LEN = 128

SMALL_MODEL_PATH = "trained_small_gpt.pt"
GPT2_FINETUNED_PATH = "gpt2_finetuned.pt"


# --------------------
# Sampling helpers
# --------------------
def top_k_filtering(logits, k):
    if k is None or k <= 0:
        return logits

    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)

    return torch.where(
        logits < min_values,
        torch.full_like(logits, float("-inf")),
        logits
    )


def top_p_filtering(logits, p):
    if p is None or p <= 0.0 or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True
    )

    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)

    cutoff = cumulative_probs > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_logits[cutoff] = float("-inf")
    logits.scatter_(1, sorted_indices, sorted_logits)

    return logits


# --------------------
# Generation function
# --------------------
def generate(
    model,
    input_ids,
    context_len,
    max_new_tokens,
    temperature=1.0,
    greedy=False,
    top_k=None,
    top_p=None,
):
    model.eval()

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -context_len:]

        with torch.no_grad():
            logits = model(input_cond)

        logits = logits[:, -1, :]

        if greedy:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_filtering(logits, top_k)
            logits = top_p_filtering(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


# --------------------
# CLI
# --------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPT-style text generation"
    )

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)

    parser.add_argument(
        "--use_gpt2",
        action="store_true",
        help="Use GPT-2 pretrained weights"
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Use fine-tuned GPT-2 weights (requires --use_gpt2)"
    )

    args = parser.parse_args()

    tokenizer = TikTokenWrapper("gpt2")

    # --------------------
    # Model loading logic
    # --------------------
    if args.use_gpt2:
        print("Loading GPT-2 pretrained weights...")
        model = load_gpt2_weights().to(DEVICE)

        if args.finetuned:
            print("Loading fine-tuned GPT-2 weights...")
            model.load_state_dict(
                torch.load(GPT2_FINETUNED_PATH, map_location=DEVICE),
                strict=False,
            )

        context_len = 1024

    else:
        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            context_length=SMALL_CONTEXT_LEN,
            embed_dim=SMALL_EMBED_DIM,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
        ).to(DEVICE)

        model.load_state_dict(
            torch.load(SMALL_MODEL_PATH, map_location=DEVICE)
        )

        context_len = SMALL_CONTEXT_LEN

    # --------------------
    # Encode prompt
    # --------------------
    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)],
        dtype=torch.long
    ).to(DEVICE)

    # --------------------
    # Generate
    # --------------------
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        context_len=context_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        greedy=args.greedy,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    output_text = tokenizer.decode(output_ids[0].tolist())

    print("\n--- Generated Text ---\n")
    print(output_text)


if __name__ == "__main__":
    main()

