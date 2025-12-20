# Day 10 — Inference: Prompt → Text Generation (CLI-Based)

## Objective
Implement a complete autoregressive inference pipeline that allows
the GPT-style model to generate text from a user-provided prompt.

This includes:
- Token-level generation
- Greedy and sampling-based decoding
- Temperature control
- Command-line interface (CLI)

After this day, the project supports a full **prompt → answer** workflow.

---

## What Inference Means in GPT Models

GPT-style models are **autoregressive**:
- They generate text one token at a time
- Each new token is conditioned on all previous tokens

Inference repeats the same operation used during training:
Predict the next token given the previous tokens.

---

## Autoregressive Generation Loop

1. Encode the input prompt into token IDs  
2. Feed tokens into the model  
3. Obtain logits for all positions  
4. Select logits for the **last token**  
5. Convert logits to probabilities  
6. Choose the next token  
7. Append token to input  
8. Repeat until desired length  

This loop is the core of GPT inference.

---

## Why Only the Last Token Is Used

Although the model outputs logits for all positions `(B, T, V)`:
- Only the **last position** predicts the *next* token
- Earlier logits correspond to already-known tokens

This makes inference efficient and correct.

---

## Sampling Strategies

### Greedy Decoding
- Selects the token with the highest probability
- Deterministic output
- Useful for debugging and sanity checks
- Often repetitive

```python
next_token = torch.argmax(probs, dim=-1, keepdim=True)
```

---

### Temperature Sampling
- Introduces randomness
- Logits are scaled before softmax

```python
logits = logits / temperature
```

Effects:
- Lower temperature (< 1.0): safer, more deterministic output
- Higher temperature (> 1.0): more diverse output

Sampling is performed using:
```python
torch.multinomial(probs, num_samples=1)
```

---

## Context Cropping

Models have a fixed context window.

If the input exceeds `context_length`:
- Only the most recent tokens are used
- Older tokens are discarded

This preserves correctness and prevents shape errors.

---

## Why Softmax Is Not Inside the Model

- During training, `CrossEntropyLoss` applies softmax internally
- During inference, softmax is applied **only for sampling**
- Keeping softmax out of the model improves numerical stability

---

## Command-Line Interface (CLI)

Inference is exposed via a CLI for usability and reproducibility.

### Sampling mode
```bash
python -m inference.generate   --prompt "Machine learning is"   --max_new_tokens 50   --temperature 0.8
```

### Greedy mode
```bash
python -m inference.generate   --prompt "Machine learning is"   --max_new_tokens 50   --greedy
```

CLI arguments:
- `--prompt`: input text
- `--max_new_tokens`: number of tokens to generate
- `--temperature`: sampling randomness
- `--greedy`: deterministic decoding flag

---

## Expected Output Characteristics

- Output is syntactically valid text
- Content quality depends on:
  - dataset size
  - training time
  - model capacity
- Repetition or incoherence is normal at this stage

Correctness is defined by:
Prompt is extended coherently without crashes.

---

## Q&A

**Q: Why generate one token at a time?**  
A: Because GPT models are autoregressive; each token depends on the previous output.

**Q: Why not apply softmax in the model forward pass?**  
A: Softmax is applied by the loss during training and during sampling in inference.

**Q: Why does temperature affect output quality?**  
A: It controls the entropy of the probability distribution over tokens.

**Q: Why does greedy decoding often repeat tokens?**  
A: It always selects the most likely token, reducing diversity.

---

## Limitations

- Small training dataset
- Limited training epochs
- No instruction tuning

These limitations are expected for a from-scratch implementation.

---

## Outcome of Day 10

- End-to-end prompt → text generation works
- Model can be demoed interactively
- Inference pipeline matches GPT-style generation
- Project is now a complete mini-LLM system

---

## Next Step
Evaluate generation behavior and compare
outputs with a reference model such as GPT-2.
