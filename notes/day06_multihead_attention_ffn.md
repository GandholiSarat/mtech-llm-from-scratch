# Day 6 — Multi-Head Attention & FeedForward Network (GPT-2 Style)

## Objective
Implement GPT-2–style multi-head causal self-attention and
the feedforward (MLP) network, aligning architecture for
future GPT-2 weight integration and comparison.

---

## Why Single-Head Attention Is Insufficient
A single attention head:
- Operates in one representation subspace
- Must capture syntax, semantics, locality, and long-range relations
- Is capacity-limited

---

## Why Multi-Head Attention Works Better
Multi-head attention:
- Splits embeddings into H subspaces
- Each head attends independently
- Captures diverse relationships in parallel

> Key idea: different heads learn different inductive biases.

---

## Multi-Head Attention Mathematics

Let:
- Input x ∈ ℝ(B × T × C)
- Heads = H
- Head dim = C / H

Steps:
1. Project x → QKV using one linear layer
2. Reshape → (B, H, T, C/H)
3. Compute scaled dot-product attention per head
4. Apply causal mask
5. Concatenate heads
6. Final output projection

---

## Causal Masking (Autoregression)
Language models must not see future tokens.
Causal masking enforces:

Token t can attend only to tokens ≤ t.

This guarantees autoregressive training and generation.

---

## FeedForward Network (FFN)

Purpose:
- Attention mixes across tokens
- FFN mixes across feature dimensions

Architecture:
Linear(C → 4C) → GELU → Linear(4C → C)

Applied independently at each token position.

---

## Why FFN Uses 4× Expansion
- Increases model capacity
- Adds non-linearity
- Dominates parameter count in Transformer layers

---

## Shape Discipline (Critical)

Every sublayer preserves shape:

(B, T, C) → (B, T, C)

This enables:
- Residual connections
- Layer stacking
- Stable deep architectures

---

## Parameter Scaling
With:
- C = 768
- Layers = 16

Approximate parameters:
- Attention + FFN ≈ 12 × C² per layer
- Total ≈ 160–165M parameters

This aligns closely with GPT-2-scale models.

---

##  Q&A

**Q: Why multiple heads instead of one big head?**  
A: To attend to multiple representation subspaces simultaneously.

**Q: Why is attention O(T²)?**  
A: Every token attends to every other token.

**Q: Which has more parameters — attention or FFN?**  
A: FFN.

**Q: Why preserve (B, T, C) everywhere?**  
A: Enables residual connections and deep stacking.

---

## Why This Design Enables GPT-2 Integration
- Same embedding size
- Same head count
- Same projection layout
- Same causal masking
- Same FFN structure

Only weights differ.

---

## Next Step
Add LayerNorm + residual connections and assemble
a full GPT-2–style Transformer block.
---

## Model Configuration (Locked for Future GPT-2 Integration)

```text 

VOCAB_SIZE = 50257        # GPT-2 vocab
CONTEXT_LEN = 1024        # GPT-2 context
EMBED_DIM = 768           # GPT-2 embedding size
NUM_HEADS = 12            # GPT-2 heads
HEAD_DIM = 64             # 768 / 12
FFN_MULT = 4              # GPT-2 MLP expansion
NUM_LAYERS = 16           # > GPT-2 small (12) → ~163M params

```
