
# LLM From Scratch (CPU-First)

This project implements a **GPT-style decoder-only Transformer language model**
from first principles using **PyTorch**, with a strong emphasis on
**architectural correctness, numerical fidelity, and debuggability**.

The project progresses from:
- building a GPT model entirely from scratch  
- loading **GPT-2 pretrained weights** into the same implementation  
- **parameter-efficient fine-tuning** and controlled inference comparison  

All core model logic (attention, MLP, training, inference) is written manually.

This project is inspired by Sebastian Raschka’s *LLMs-from-scratch*,
but all code is implemented independently as part of an **M.Tech project**.

---

## Core Focus

- Explicit tensor operations and shape discipline
- Decoder-only Transformer internals
- CPU-first development for correctness and debugging
- Avoiding high-level LLM abstractions
- Understanding *why* LLMs work, not just *how to use them*

---

## Goals

- Build a GPT-style language model completely from scratch
- Implement tokenization, attention, training, and inference manually
- Avoid high-level training frameworks (e.g. HuggingFace Trainer)
- Maintain a clean, well-documented, interview-ready codebase
- Load and run **GPT-2 pretrained weights inside a custom implementation**
- Compare scratch-trained, pretrained, and fine-tuned models

---

## Current Status

- Project structure and Git hygiene
- CPU-only PyTorch environment verified
- Sliding-window autoregressive dataset pipeline
- Industry-grade BPE tokenization using TikToken (GPT-2 compatible)
- Token and positional embeddings
- GPT-2–style multi-head causal self-attention
- Feedforward (MLP) network
- Pre-LayerNorm Transformer blocks with residual connections
- Full GPT-style decoder model (~152.8M parameters)
- Autoregressive training loop (CPU-safe)
- CLI-based inference (prompt → text)
- Evaluation and debugging utilities
- Regularization (dropout)
- **GPT-2 pretrained weight loading**
- **Numerical fidelity fixes (GELU, LayerNorm eps, weight tying)**
- **Unified inference for scratch / GPT-2 / fine-tuned GPT-2**
- **Parameter-efficient GPT-2 fine-tuning**

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
│   ├── attention.py
│   ├── multihead_attention.py
│   ├── feedforward.py
│   ├── transformer_block.py
│   ├── gpt_model.py
│   └── gpt2_compatible.py
├── training/
│   ├── train.py
│   ├── train_gpt2_finetune.py
│   └── test.py
├── inference/
│   └── generate.py
├── tools/
│   ├── load_gpt2_weights.py
│   ├── load_gpt2_load.py
│   └── freeze.py
├── notes/
│   ├── README.md
│   ├── day01_setup.md
│   ├── day02_dataset.md
│   ├── day03_tokenizer.md
│   ├── day04_embeddings.md
│   ├── day05_attention.md
│   ├── day06_multihead_attention_ffn.md
│   ├── day07_transformer_block.md
│   ├── day08_full_gpt_model.md
│   ├── day09_training_loop.md
│   ├── day10_inference.md
│   ├── day11_evaluation_debugging.md
│   ├── day14_regularization.md
│   └── Post_Day14_GPT2_Integration.md
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
python -m training.test
```

Expected:
- CUDA available: False
- Successful forward pass / loss output
- No runtime errors

---

## Training (From Scratch, CPU-Safe)

```bash
python -m training.train
python -m training.train_gpt2_finetune
```

---

## Inference (Prompt → Text)

### Scratch-trained model
```bash
python -m inference.generate   --prompt "Learning is"   --top_p 0.9   --temperature 0.8
```

### GPT-2 pretrained
```bash
python -m inference.generate   --prompt "Learning is"   --use_gpt2   --top_p 0.9   --temperature 0.8
```

### GPT-2 fine-tuned
```bash
python -m inference.generate   --prompt "Learning is"   --use_gpt2   --finetuned   --top_p 0.9   --temperature 0.7   --max_new_tokens 100
```

---

## Notes & Documentation

Detailed technical notes are available in:

```
notes/README.md
```

---

## Limitations

- Small fine-tuning dataset
- CPU-only training
- No instruction tuning or RLHF
- No distributed training

---

## Key Takeaway

This project demonstrates a **complete, end-to-end GPT-style LLM pipeline**
implemented from scratch and extended to support **pretrained GPT-2 loading
and fine-tuning**, reflecting real-world LLM engineering practices.
