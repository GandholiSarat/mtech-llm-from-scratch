# Day 4 — Token & Positional Embeddings

## Objective
Introduce the first learnable parameters of the language model
by implementing token and positional embeddings.

## Why Embeddings Are Required
- Token IDs are discrete and non-numeric
- Neural networks operate on continuous vectors
- Embeddings map tokens to dense representations

## Positional Information
- Self-attention is permutation-invariant
- Positional embeddings inject order information
- Used learned positional embeddings for simplicity

## Design Choices
- Embedding dimension = 128
- Context length = 32
- Learned position embeddings instead of sinusoidal

## Verification
- Input shape: (B, T)
- Output shape: (B, T, C)
- Shapes verified using a sample batch

## Tensor Shape Notation (B, T, C)

Throughout the project, tensors follow the standard
Transformer notation:

- **B (Batch size)**  
  Number of independent sequences processed in parallel.

- **T (Time steps / Sequence length)**  
  Number of tokens in each input sequence  
  (also called context length).

- **C (Channels / Embedding dimension)**  
  Dimensionality of the token representation.

### Examples
- Input token IDs: `(B, T)`
- Token embeddings: `(B, T, C)`
- Attention output: `(B, T, C)`

This convention is used consistently across embedding,
attention, and Transformer block implementations.


## Next Steps
- Implement causal self-attention
- Build Transformer blocks

