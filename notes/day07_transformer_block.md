# Day 7 — Transformer Block (GPT-2 Style)

## Objective
Assemble attention and feedforward layers into a
GPT-2–style Transformer block using pre-LayerNorm
and residual connections.

---

## Why Transformer Blocks Matter
Individual components (attention, FFN) are not useful
on their own. Transformer blocks provide:

- Stability
- Depth
- Reusability
- Trainability

All modern LLMs are stacks of Transformer blocks.

---

## GPT-2 Block Structure

Each block consists of:
1. LayerNorm
2. Multi-Head Causal Self-Attention
3. Residual connection
4. LayerNorm
5. FeedForward Network
6. Residual connection

This is known as **Pre-Norm Transformer**.

---

## Pre-Norm vs Post-Norm

### Pre-Norm (GPT-2)
LN → Sublayer → Residual

Benefits:
- Better gradient flow
- More stable deep training
- Preferred in large LLMs

### Post-Norm (Original Transformer)
Sublayer → LN

Less stable for deep models.

---

## Residual Connections

Residuals allow:
- Identity mapping
- Gradient shortcuts
- Deep stacking of layers

Mathematically:
x ← x + f(x)

Without residuals, Transformers do not scale.

---

## Shape Discipline

Every sublayer preserves shape:

(B, T, C) → (B, T, C)

This enables:
- Residual addition
- Layer stacking
- Modular design

---

## Why Attention Comes Before FFN

- Attention mixes information **across tokens**
- FFN mixes information **across channels**

This ordering is consistent across GPT-style models.

---

## Q&A

**Q: Why pre-LayerNorm instead of post-LayerNorm?**  
A: Pre-Norm improves gradient flow and stabilizes training in deep Transformers.

**Q: Why do residual connections matter?**  
A: They prevent vanishing gradients and enable deep architectures.

**Q: Why preserve embedding dimension throughout?**  
A: To allow stacking, residuals, and consistent interfaces.

---

## Parameter Impact

LayerNorm adds negligible parameters compared to
attention and FFN but dramatically improves stability.

---

## Why This Design Enables GPT-2 Integration

- Same block structure
- Same normalization placement
- Same residual flow
- Same tensor shapes

This allows meaningful comparison and partial
weight alignment with GPT-2.

---

## Next Step
Stack multiple Transformer blocks to build
the full GPT-style decoder model.

