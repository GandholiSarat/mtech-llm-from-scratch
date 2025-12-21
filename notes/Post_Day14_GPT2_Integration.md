
# Post–Day 14 Notes: GPT-2 Integration, Weight Loading, and Fine-Tuning

These notes cover **everything done after Day 14**, starting from regularization
and extending the project to a **fully functional GPT-2-compatible system**
with pretrained loading, inference comparison, and parameter-efficient fine-tuning.

This document is written as **revision + interview preparation notes**.

---

## 1. Day 14 Recap (Baseline)

By the end of Day 14, the project had:

- Decoder-only GPT architecture implemented from scratch
- Causal multi-head self-attention
- Feedforward blocks
- Pre-LayerNorm architecture
- Dropout added for regularization
- Stable training and inference
- Custom CLI for generation

At this stage:
- Model was trained **from scratch**
- Generation quality was reasonable but limited by data and scale

---

## 2. Motivation After Day 14

Goals beyond Day 14:

1. Compare our implementation against a **real pretrained LLM**
2. Prove architectural correctness (not just “working code”)
3. Demonstrate industry-level skills:
   - weight loading
   - numerical compatibility
   - fine-tuning strategies
4. Make the project **interview-ready**

Chosen reference model:
- **GPT-2 (small)**

---

## 3. GPT-2 Compatibility Requirements

To load GPT-2 weights, the following had to match **exactly**:

- Vocabulary size = 50257
- Tokenizer = GPT-2 BPE (tiktoken)
- Decoder-only transformer
- Pre-norm architecture
- Single QKV projection
- Same hidden size, heads, layers
- Same activation, LayerNorm behavior

Any mismatch causes **severe generation degradation**.

---

## 4. GPT-2 Compatible Model Configuration

A new GPT-2-compatible configuration was introduced:

- embed_dim = 768
- num_heads = 12
- num_layers = 12
- context_length = 1024
- dropout = 0.1

This was kept **separate** from the small scratch-trained model.

Key design principle:
> Never overwrite or break the scratch model.

---

## 5. Loading GPT-2 Pretrained Weights

### 5.1 HuggingFace Usage Policy

HuggingFace was used **only** to:
- download pretrained weights

NOT used for:
- model forward pass
- attention
- training
- inference logic

All computation flows through **our code**.

---

### 5.2 Weight Mapping Challenges (Important)

#### (a) Transposed Linear Weights

HuggingFace stores linear weights as:

- shape: (in_features, out_features)

PyTorch nn.Linear expects:

- shape: (out_features, in_features)

Therefore, weights for:
- QKV projection
- FFN c_fc
- FFN c_proj

**must be transposed** when loading.

---

#### (b) Positional Embeddings Shape

GPT-2 stores:
- (context_length, embed_dim)

Our model uses:
- (1, context_length, embed_dim)

Solution:
- Unsqueeze positional embeddings during load.

---

#### (c) LayerNorm Name Mismatch

GPT-2 naming:
- ln_1, ln_2, ln_f

Our naming:
- ln1, ln2, ln_f

Explicit mapping required.

---

## 6. Weight Tying (Critical Insight)

GPT-2 ties:
- token embedding weights
- output LM head weights

Initially, they were loaded separately, which caused:
- broken token probabilities
- strange Unicode
- incoherent text

Fix:
- Explicitly tie:
  lm_head.weight = token_embedding.weight

This single step dramatically improved generation quality.

---

## 7. Numerical Fidelity Fixes (Most Important Section)

Even after correct weight loading, generation was still degraded.

Root cause:
> GPT-2 is extremely sensitive to numerical details.

### 7.1 GELU Variant

GPT-2 uses **approximate GELU**, not PyTorch’s default.

Formula:

0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

Using nn.GELU() breaks GPT-2 equivalence.

A custom GPT2GELU module was implemented.

---

### 7.2 LayerNorm Epsilon

GPT-2 uses:
- eps = 1e-5

PyTorch default:
- eps = 1e-12

All LayerNorm layers were updated.

---

### 7.3 Dropout During Inference

Dropout was explicitly disabled after loading GPT-2 weights.

---

## 8. Successful GPT-2 Generation

After all fixes:
- GPT-2 pretrained weights generated fluent, coherent English
- Output matched expected GPT-2 quality

This confirmed:
> GPT-2 is running **inside our implementation**, not HuggingFace’s.

---

## 9. Unified Inference CLI

The inference CLI was extended with flags:

- Default: scratch-trained model
- --use_gpt2: GPT-2 pretrained
- --use_gpt2 --finetuned: GPT-2 fine-tuned

Same sampling code used for all modes:
- greedy
- top-k
- top-p
- temperature

This enables **controlled comparison**.

---

## 10. Freezing GPT-2 for Fine-Tuning

### 10.1 Why Freeze?

- Prevent catastrophic forgetting
- Enable training on small datasets
- Reduce compute
- Standard industry practice

---

### 10.2 Fine-Tuning Strategy

Frozen:
- Attention layers
- MLP layers
- Token embeddings

Trainable:
- LayerNorms
- LM head

Result:
- <1% parameters updated
- Strong style adaptation
- GPT-2 fluency preserved

---

## 11. Fine-Tuning Results

Observed behavior:

- GPT-2 pretrained:
  - Fluent but generic
- GPT-2 fine-tuned:
  - Fluent + dataset-specific tone
- Scratch model:
  - Strong alignment, lower fluency

This demonstrates:
> Pretraining + small fine-tuning beats training from scratch on small data.

---

## 12. Final Project Capabilities

At this point, the project supports:

- GPT from scratch (training + inference)
- GPT-2 pretrained weight loading
- Numerical equivalence debugging
- Parameter-efficient fine-tuning
- Unified CLI
- Model comparison

This is **well beyond a typical academic project**.

---

## 13. Interview Key Takeaways (Memorize)

- “I implemented GPT from scratch and loaded GPT-2 weights into my model.”
- “I handled transposed linear weights and embedding tying.”
- “I matched GELU variants and LayerNorm eps for numerical fidelity.”
- “I froze GPT-2 and fine-tuned only normalization and output layers.”
- “I compared scratch, pretrained, and fine-tuned models using the same inference pipeline.”

---

## 14. Final Reflection

This project demonstrates:

- Deep understanding of Transformer internals
- Practical ML engineering skills
- Debugging numerical issues
- Industry-relevant workflows

This is a **complete, defensible, interview-ready LLM system**.
