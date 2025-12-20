# LLM From Scratch — Notes Index

This directory contains day-wise technical notes documenting
the design, implementation, and reasoning behind each stage
of building a GPT-style language model from scratch.

The notes are written for:
- future self-review
- interview preparation
- architectural clarity

---

## Day-wise Notes Overview

### Day 1 — Project Setup
**File:** day01_setup.md  
- CPU-first development rationale
- Environment isolation
- Git hygiene
- Motivation for building from scratch

---

### Day 2 — Dataset Pipeline
**File:** day02_dataset.md  
- Sliding window dataset
- Context length handling
- Autoregressive input–target pairs
- Shape discipline (B, T)

---

### Day 3 — Tokenization
**File:** day03_tokenizer.md  
- Character vs subword tokenization
- TikToken (GPT-2 BPE)
- Vocabulary size = 50257
- Encode–decode sanity checks

---

### Day 4 — Embeddings
**File:** day04_embeddings.md  
- Token embeddings
- Positional embeddings
- Why embeddings are required
- Tensor notation: B (batch), T (time), C (channels)

---

### Day 5 — Causal Self-Attention
**File:** day05_attention.md  
- Scaled dot-product attention
- Q, K, V projections
- Causal masking
- Autoregressive constraint
- Interview-focused explanations

---

### Day 6 — Multi-Head Attention & FFN (GPT-2 Style)
**File:** day06_multihead_attention_ffn.md  
- Multi-head attention motivation
- Head dimension splitting
- GPT-2 style QKV projection
- Feedforward (MLP) network
- Parameter scaling analysis (~150M+)

---

### Day 7 — Transformer Block
**File:** day07_transformer_block.md  
- Pre-LayerNorm vs Post-LayerNorm
- Residual connections
- GPT-2 block structure
- Stability and gradient flow

---

### Day 8 — Full GPT Model Assembly
**File:** day08_full_gpt_model.md  
- Stacking Transformer blocks
- Final LayerNorm
- Language modeling head
- Weight tying
- Decoder-only architecture
- ~152.8M parameter model

---

### Day 9 — Training Loop
**File:** day09_training_loop.md  
- Autoregressive training objective
- Cross-entropy loss
- AdamW optimizer
- CPU-safe training configuration
- Loss sanity checks

---

### Day 10 — Inference (Prompt → Text)
**File:** day10_inference.md  
- Autoregressive generation loop
- Greedy vs sampling decoding
- Temperature scaling
- Context cropping
- Command-line inference interface

---

## How to Read These Notes

Suggested order:
1. Day 1 → Day 5 for fundamentals
2. Day 6 → Day 8 for architecture
3. Day 9 for training mechanics
4. Day 10 for end-to-end inference

Reading these sequentially should enable you to
reconstruct the entire model from scratch.

---

## Future Extensions (Planned)

- GPT-2 reference weight integration
- Output comparison with pretrained models
- Top-k / top-p sampling
- Performance benchmarking

