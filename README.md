# LLM From Scratch (CPU-First)

This project implements a GPT-style Transformer language model
from first principles using PyTorch.

The focus is on:
- Explicit tensor operations
- Educational clarity
- CPU-first development

This project is inspired by
Sebastian Raschka’s *LLMs-from-scratch* repository.
All code is independently written as part of my M.Tech work.

---

## Goals
- Build a decoder-only Transformer (GPT-style) from scratch
- Implement data pipeline, tokenization, attention, training, and inference manually
- Avoid high-level LLM frameworks
- Maintain a clean, well-documented, reproducible codebase

---
## Current Status

- Project structure and Git hygiene set up
- CPU-only PyTorch environment verified
- Text dataset pipeline with sliding window implemented
- Industry-grade BPE tokenization using TikToken
- Token and positional embedding layers implemented
- GPT-2–style multi-head causal self-attention implemented
- Feedforward (MLP) network implemented
- Pre-LayerNorm Transformer blocks with residual connections
- Full GPT-style decoder model assembled (~153M parameters)
- Autoregressive training loop implemented (CPU-safe configuration)
- Inference (prompt → text generation) [next]


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
├── training/
│   └── train.py
├── inference/
├── experiments/
├── notes/
│   ├── day01_setup.md
│   ├── day02_dataset.md
│   └── day03_tokenizer.md
├── requirements.txt
└── README.md
```
## Setup 

```bash
python -m venv llm-scratch
source llm-scratch/bin/activate

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt


## Verify installation 
python training/train.py
