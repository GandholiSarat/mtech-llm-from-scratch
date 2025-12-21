# LLM From Scratch — Notes Index

This directory contains **day-wise and milestone-wise technical notes**
documenting the design, implementation, debugging, and reasoning behind
building a **GPT-style language model from scratch** in PyTorch.

The notes are written for:
- future self-review
- interview preparation
- architectural clarity
- demonstrating end-to-end LLM engineering depth

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
- GPT-2 style single QKV projection
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
- Loss sanity checks and debugging

---

### Day 10 — Inference (Prompt → Text)
**File:** day10_inference.md  
- Autoregressive generation loop
- Greedy vs sampling decoding
- Temperature scaling
- Context cropping
- Command-line inference interface

---

### Day 11 — Evaluation & Debugging
**File:** day11_evaluation_debugging.md  
- Logits inspection
- Loss behavior analysis
- Dataset sanity checks
- Diagnosing repetition and collapse
- Practical debugging workflow

---

### Day 14 — Regularization
**File:** day14_regularization.md  
- Dropout placement (attention, FFN, embeddings)
- Why regularization matters in Transformers
- Training vs inference behavior
- Stability improvements

---

## Post–Day 14: Advanced Extensions (Major Milestone)

### GPT-2 Integration, Weight Loading & Fine-Tuning
**File:** post_day14_notes.md  

This document covers **everything done after Day 14**, including:

- Making the model **GPT-2 compatible**
- Loading GPT-2 pretrained weights into a custom implementation
- Handling transposed linear weights
- Positional embedding shape alignment
- Weight tying (token embedding ↔ LM head)
- Numerical fidelity fixes:
  - GPT-2 GELU approximation
  - LayerNorm epsilon matching
  - Dropout disabling during inference
- Unified inference CLI:
  - scratch-trained model
  - GPT-2 pretrained
  - GPT-2 fine-tuned
- Freezing GPT-2 and parameter-efficient fine-tuning
- Behavioral comparison across models
- Interview-ready technical takeaways

This marks the transition from **educational implementation** to
**industry-grade LLM engineering**.

---

## How to Read These Notes

Suggested reading order:

1. **Day 1 → Day 5**  
   Fundamentals: data, tokenization, attention

2. **Day 6 → Day 8**  
   Core architecture and scaling

3. **Day 9 → Day 10**  
   Training and inference end-to-end

4. **Day 11 & Day 14**  
   Debugging and regularization

5. **Post–Day 14 Notes**  
   Pretrained models, weight loading, fine-tuning, and comparison

Reading sequentially should enable you to:
- rebuild the model from scratch
- reason about architectural choices
- explain design trade-offs in interviews

---

## Project Maturity Summary

By the end of these notes, the project demonstrates:

- GPT model built from first principles
- Training from scratch on custom data
- GPT-2 pretrained weight integration
- Numerical equivalence debugging
- Parameter-efficient fine-tuning
- Unified inference and comparison tooling

This goes **well beyond a typical academic assignment** and reflects
real-world LLM engineering practices.
