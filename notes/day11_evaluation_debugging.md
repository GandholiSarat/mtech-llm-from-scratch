# Day 11 — Evaluation, Debugging & Sanity Checks

## Objective
Validate the correctness of the GPT-style language model by
analyzing training loss behavior, debugging dataset issues,
and understanding common failure modes in language model training.

This day focuses on **diagnosing problems**, not adding new features.

---

## Why Evaluation Matters

A language model can:
- compile
- train
- produce output

and still be **incorrect**.

Day 11 ensures:
- the learning signal is valid
- the dataset is wired correctly
- low or high loss values are interpreted correctly

---

## Loss Sanity Check (First Principle)

For autoregressive language modeling with cross-entropy loss:

Initial loss ≈ log(vocab_size)

For GPT-2 tokenizer:
```
vocab_size = 50257
log(50257) ≈ 10.82
```

### Expected Behavior
- Initial loss around **10–11**
- Gradual decrease during training

---

## Debugging Extremely High Loss

Observed earlier:
```
Loss ≈ 168
```

This is **impossible** for a correctly wired model.

### Root Causes
- Targets outside vocabulary range
- Incorrect target shifting
- Dataset corruption

### Fix Applied
- Ensured tokens are generated using tokenizer.encode
- Ensured targets are within `[0, vocab_size - 1]`
- Verified using min/max checks on input and target tensors

---

## Token Range Verification

During training, the following checks were added:

```
x min/max: 11 46184
y min/max: 11 46184
vocab size: 50257
```

This confirms:
- No negative token IDs
- No out-of-range class indices
- CrossEntropyLoss is receiving valid targets

---

## Logits Sanity Check

Logged statistics:
```
logits mean ≈ 0
logits std ≈ 16
```

Interpretation:
- Mean near zero → healthy initialization
- Standard deviation moderately high but stable
- No exploding or vanishing activations

---

## Low Loss Observation (~0.35)

After fixing dataset bugs, training loss dropped to:
```
~0.35
```

### Important Insight
This **does not indicate a bug**.

### Why This Happens
- Sliding window stride = 1
- Large context length
- Small dataset
- High overlap between consecutive samples

This creates highly correlated training examples, allowing the model
to predict next tokens almost perfectly.

This is known as the **sliding-window overlap illusion**.

---

## Overfitting Test (Gold Standard)

A deliberate overfitting test was performed using a tiny dataset:

```
hello world hello world hello world
```

Expected behavior:
- Loss approaches zero
- Model memorizes sequence
- Generated text repeats training pattern

Passing this test confirms:
- Correct dataset construction
- Correct masking
- Correct loss wiring
- Correct optimization flow

---

## Dataset Implementation Bug (Fixed)

Original issue:
```python
return torch.tensor(x), torch.tensor(y)
```

Problem:
- `x` and `y` were already tensors
- Redundant tensor construction
- Triggered PyTorch warnings
- Poor practice

### Correct Implementation
```python
return x, y
```

---

## Common Generation Failure Modes

### Repetition
Cause:
- Greedy decoding
- Low entropy

### Gibberish
Cause:
- Undertraining
- High temperature

### Early Termination
Cause:
- EOS token
- Short generation length

These are expected behaviors.

---

## Q&A

**Q: How do you verify a language model is correct?**  
A: Loss sanity checks, token range validation, and overfitting tests.

**Q: Why can loss become artificially low?**  
A: Extreme overlap in sliding-window datasets.

**Q: Is low training loss always good?**  
A: No. It may indicate memorization rather than generalization.

---

## Outcome of Day 11

- Dataset correctness verified
- Loss behavior understood
- Real bugs fixed
- Model correctness established

---

## Next Step
Improve inference quality using top-k and top-p sampling.
