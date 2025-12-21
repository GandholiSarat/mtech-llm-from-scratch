import torch
from tokenizer.tokenizer import TikTokenWrapper
from tools.load_gpt2_weights import load_gpt2_weights

tokenizer = TikTokenWrapper("gpt2")
model = load_gpt2_weights()

prompt = "Learning is"
input_ids = torch.tensor([tokenizer.encode(prompt)])

with torch.no_grad():
    logits = model(input_ids)

print("Logits shape:", logits.shape)

