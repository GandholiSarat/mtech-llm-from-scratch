# LLM From Scratch (CPU-First)

This project implements a **GPT-style decoder-only Transformer language model**
from first principles using **PyTorch**.

The focus is on:
- Explicit tensor operations
- Architectural clarity
- CPU-first development for correctness and debuggability
- Understanding how modern LLMs work internally

This project is inspired by Sebastian Raschka’s *LLMs-from-scratch* repository,
but all code is written independently as part of an M.Tech project.

---

## Goals

- Build a GPT-style language model completely from scratch
- Implement tokenization, attention, training, and inference manually
- Avoid high-level LLM frameworks (e.g. HuggingFace Trainer)
- Maintain a clean, well-documented, interview-ready codebase
- Enable future comparison with pretrained GPT-2 models

---

## Current Status

- ✅ Project structure and Git hygiene set up
- ✅ CPU-only PyTorch environment verified
- ✅ Sliding-window text dataset pipeline implemented
- ✅ Industry-grade BPE tokenization using TikToken
- ✅ Token and positional embeddings implemented
- ✅ GPT-2–style multi-head causal self-attention
- ✅ Feedforward (MLP) network
- ✅ Pre-LayerNorm Transformer blocks with residual connections
- ✅ Full GPT-style decoder model assembled (~152.8M parameters)
- ✅ Autoregressive training loop (CPU-safe configuration)
- ✅ CLI-based inference (prompt → text generation)
- ⏳ Evaluation, sampling improvements, GPT-2 reference comparison (next)

---

## Project Structure

```text
mtech-llm-from-scratch/
├── data/
│   ├── input.txt
│   └── dataset.py
├── tokenizer/
│   └── tokenizer.py
├── model/
│   ├── embedding.py
│   ├── multihead_attention.py
│   ├── feedforward.py
│   ├── transformer_block.py
│   └── gpt_model.py
├── training/
│   └── train.py
├── inference/
│   └── generate.py
├── notes/
│   ├── notes_readme.md
│   ├── day01_setup.md
│   ├── day02_dataset.md
│   ├── day03_tokenizer.md
│   ├── day04_embeddings.md
│   ├── day05_attention.md
│   ├── day06_multihead_attention_ffn.md
│   ├── day07_transformer_block.md
│   ├── day08_full_gpt_model.md
│   ├── day09_training_loop.md
│   └── day10_inference.md
├── requirements.txt
└── README.md
```
---

## Setup

```bash
python -m venv llm-scratch
source llm-scratch/bin/activate

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python training/train.py
```

Expected:
- CUDA available: False
- Successful forward pass / loss output
- No runtime errors

---

## Training (CPU-Safe)

A smaller configuration is used for CPU training
to verify correctness.

```bash
python -m training.train
```

---

## Inference (Prompt → Text)

### Sampling mode
```bash
python -m inference.generate \
  --prompt "Machine learning is" \
  --max_new_tokens 50 \
  --temperature 0.8
```

### Greedy mode
```bash
python -m inference.generate \
  --prompt "Machine learning is" \
  --max_new_tokens 50 \
  --greedy
```

---

## Notes & Documentation

Day-wise technical notes are available in the `notes/` directory.
Start with:

notes/README.md

---

## Limitations

- Small training dataset
- Limited training epochs
- No instruction tuning or RLHF

---

## Planned Extensions

- GPT-2 reference weight integration
- Output comparison with pretrained models
- Top-k / top-p sampling
- Performance benchmarking
- Optional GPU support

---

## Key Takeaway

This project demonstrates a **full GPT-style language model pipeline**
—from raw text to tokenization, attention, training, and inference—
implemented entirely from scratch with a focus on understanding and correctness.
