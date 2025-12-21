from model.gpt_model import GPTModel

def build_gpt2_compatible_model():
    return GPTModel(
        vocab_size=50257,
        context_length=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
    )

