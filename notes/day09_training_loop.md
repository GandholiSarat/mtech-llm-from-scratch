# Day 9 — Autoregressive Training Loop

## Objective
Train a GPT-style language model using
autoregressive next-token prediction.

---

## Training Objective

Given a token sequence:
[t0, t1, t2, ..., tT]

The model learns:
Predict t(i+1) given t(0..i)

This is implemented via shifted targets.

---

## Loss Function

Cross-entropy loss over vocabulary:

- Input: logits of shape (B, T, V)
- Targets: token IDs of shape (B, T)

Flattened to:
(B·T, V) vs (B·T)

Softmax is applied internally by CrossEntropyLoss.

---

## Why Use AdamW

- Adaptive learning rates
- Weight decay improves generalization
- Standard optimizer for Transformers

---

## CPU-Safe Training Strategy

- Smaller embedding dimension
- Fewer layers
- Small batch size
- Same architecture code

This validates correctness without expensive compute.

---

## Sanity Check: Loss Behavior

- Initial loss ≈ log(vocab_size)
- Loss must decrease
- If loss does not decrease, architecture or data is incorrect

---

## Q&A

**Q: Why not train the full 150M model?**  
A: Training cost is high; correctness is verified on a smaller configuration.

**Q: Why flatten logits for loss?**  
A: Cross-entropy expects (N, C) inputs.

**Q: Why no softmax in model?**  
A: Softmax is handled by loss or sampling.

---

## Next Step
Add inference code for prompt → text generation.

