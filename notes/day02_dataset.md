# Day 2 — Dataset & Token Stream

## Objective
Implement a minimal, CPU-friendly dataset pipeline for
autoregressive language model training.

## Key Concepts Learned

### 1. Token Stream
- Language models do not see text, only token IDs
- The entire dataset is treated as one long sequence of tokens

### 2. Sliding Window
Given a token sequence:

[t0, t1, t2, t3, t4, ...]

For context length = N:
- Input  (x): tokens[i : i+N]
- Target (y): tokens[i+1 : i+N+1]

This allows the model to learn:
"Predict the next token given previous tokens"

### 3. Autoregressive Training
- Targets are just inputs shifted by one position
- Same sequence length for x and y
- This formulation is used in GPT-style models

### 4. Character-Level Tokenization (Temporary)
- Implemented a simple char-level tokenizer inline
- Purpose: correctness and clarity, not efficiency
- Will refactor into a separate tokenizer module later

## Design Decisions
- Context length = 32 (CPU-friendly)
- Batch size kept small (future work)
- Dataset size intentionally small for fast iteration

## Verification
- Printed decoded input and target
- Verified target is input shifted by one character
- Dataset pipeline works end-to-end on CPU

## Next Steps
- Extract tokenizer into its own module
- Add vocabulary size computation
- Prepare for embedding layer

