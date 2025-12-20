# Day 8 — Full GPT-Style Model Assembly

## Objective
Assemble embeddings, Transformer blocks, normalization,
and output head into a complete decoder-only Transformer
language model.

---

## High-Level Architecture

Input IDs
→ Token + Positional Embeddings
→ Stack of Transformer Blocks
→ Final LayerNorm
→ Linear LM Head
→ Logits over vocabulary

This mirrors GPT-2 architecture.

---

## Decoder-Only Design

- No encoder
- No cross-attention
- Autoregressive generation
- Causal masking enforced inside attention

This design is ideal for language modeling and text generation.

---

## Transformer Block Stack

- Identical blocks stacked N times
- Depth controls expressiveness
- Residual connections preserve gradient flow

GPT-2 small uses 12 layers; this model uses 16 layers.

---

## Final LayerNorm (ln_f)

Purpose:
- Stabilize activations before logits
- Improve training dynamics
- Standard in GPT-style models

---

## Language Modeling Head

- Linear projection from embedding space to vocabulary
- Produces logits for each token position
- No softmax applied here (handled by loss or sampling)

---

## Weight Tying

- Token embedding weights reused in LM head
- Reduces parameters
- Encourages consistency between input and output representations

---

## Shape Discipline

Input IDs:        (B, T)
Embeddings:       (B, T, C)
Transformer out:  (B, T, C)
Logits:           (B, T, V)

Shape consistency is critical for training and inference.

---

## Q&A

**Q: Why decoder-only instead of encoder-decoder?**  
A: Decoder-only models are simpler and better suited for autoregressive generation.

**Q: Why stack identical blocks?**  
A: Repetition increases depth while keeping design simple and scalable.

**Q: Why no softmax in the model?**  
A: Softmax is applied during loss computation or sampling, not in the forward pass.

**Q: Why tie embedding and output weights?**  
A: Reduces parameters and improves generalization.

---

## Parameter Scaling

- Attention + FFN dominate parameter count
- LayerNorm adds negligible parameters
- Embeddings + LM head add vocabulary-scale parameters

Total parameters ≈ 163M (152.8M).

---

## Why This Enables GPT-2 Comparison

- Same tensor shapes
- Same block structure
- Same normalization placement
- Same tokenization (tiktoken)

Only training data and weights differ.

---

## Next Step
Implement training loop and loss function
to train the model autoregressively.

