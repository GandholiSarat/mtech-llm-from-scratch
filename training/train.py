import torch
from model.embedding import TokenAndPositionEmbedding
from model.multihead_attention import MultiHeadCausalSelfAttention
from model.feedforward import FeedForward

B, T = 1, 8

x = torch.randint(0, 50257, (B, T))

embed = TokenAndPositionEmbedding(
    vocab_size=50257,
    embed_dim=768,
    context_length=1024
)

attn = MultiHeadCausalSelfAttention(768, 12)
ffn = FeedForward(768)

x = embed(x)
x = attn(x)
x = ffn(x)

print(x.shape)

