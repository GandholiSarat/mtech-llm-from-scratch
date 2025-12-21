# Day 12 — Sampling Strategies (Top-k & Top-p)

## Objective
Improve text generation quality by implementing
modern decoding strategies used in GPT-style models.

---

## Why Decoding Matters

Training produces probabilities.
Decoding decides how text is generated.

Bad decoding → bad outputs
Even if the model is correct.

---

## Greedy Decoding

Always selects the most probable token.

Pros:
- Deterministic
- Simple

Cons:
- Repetition
- Low diversity

---

## Top-k Sampling

Restricts sampling to top-k most probable tokens.

Benefits:
- Removes low-probability noise
- Improves fluency

Common values:
- k = 20–50

---

## Top-p (Nucleus) Sampling

Samples from smallest token set
with cumulative probability ≥ p.

Benefits:
- Adaptive cutoff
- Better diversity control

Common values:
- p = 0.8–0.95

---

## Temperature Interaction

Temperature scales logits before sampling.

- Low temperature → safer text
- High temperature → more randomness

Used together with top-k / top-p.

---

## Q&A

Q: Why not sample from full softmax?
A: Low-probability tokens introduce noise.

Q: Why is top-p better than top-k?
A: It adapts cutoff based on distribution shape.

---

## Outcome of Day 12

- Generation quality significantly improved
- Decoding strategies understood
- Inference pipeline closer to production LLMs

---

## Next Step
Compare outputs against a reference model
and finalize project demo.

