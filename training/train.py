import torch
from model.gpt_model import GPTModel

model = GPTModel(
    vocab_size=50257,
    context_length=1024,
    embed_dim=768,
    num_heads=12,
    num_layers=16
)

input_ids = torch.randint(0, 50257, (1, 8))

logits = model(input_ids)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.1f}M")

print("Logits shape:", logits.shape)

