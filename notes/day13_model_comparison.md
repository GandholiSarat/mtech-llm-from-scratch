# Day 13 — Model Comparison & Analysis (Ours vs GPT-2)

## Objective
Analyze differences between a from-scratch GPT-style model
and a pretrained GPT-2 model to understand the impact of
scale, data, and training on generation quality.

---

## Comparison Setup

Prompt:
"Learning is"

Models compared:
- Custom GPT (from scratch)
- GPT-2 (small)

---

## Observed Differences

Custom model:
- Partial grammatical correctness
- Topic awareness
- Occasional repetition
- Limited vocabulary diversity

GPT-2:
- Fluent sentence structure
- Long-range coherence
- Rich vocabulary
- Minimal repetition

---

## Why GPT-2 Performs Better

### Data Scale
GPT-2 trained on billions of tokens.
Custom model trained on a single document.

### Parameter Scale
GPT-2 has significantly more parameters
and better parameter utilization.

### Training Strategy
GPT-2 uses dropout, LR schedules,
and extensive compute.

---

## Key Insight

Quality differences arise from scale,
not architectural correctness.

---

## What the Custom Model Demonstrates

- Correct autoregressive learning
- Valid attention mechanism
- Context-aware generation
- Proper probability modeling

---

## Q&A

Q: Why does your model underperform GPT-2?
A: Due to limited data and compute, not architectural flaws.

Q: What is the value of a from-scratch model?
A: Understanding learning dynamics, debugging, and architecture.

---

## Outcome of Day 13

- Model limitations understood
- Scaling effects clearly identified
- Project positioned for GPT-2 integration

---

## Next Step
Integrate GPT-2 pretrained weights
and compare outputs directly.

