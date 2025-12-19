# Day 1 — Project Setup & Environment

## Objective
Establish a clean, reproducible, CPU-first development environment
for building a Transformer-based language model from scratch.

## Key Decisions

### 1. CPU-First Development
- Chose CPU-only training for initial development
- Prioritized correctness, debuggability, and portability
- Ensures the project runs on any machine without GPU dependencies

### 2. Project Inspiration
- Project inspired by Sebastian Raschka’s *LLMs-from-scratch*
- Code is independently written with a focus on clarity and learning
- Structure and style emphasize explicit tensor operations

### 3. Minimal Abstractions
- Avoided high-level LLM frameworks (e.g., HuggingFace Trainer)
- Used PyTorch only for tensor operations and autograd
- Goal is to understand and implement core building blocks manually

## Outcomes
- Python virtual environment created and isolated
- CPU-only PyTorch installation verified
- Project structure initialized with Git version control
- Basic sanity check confirmed tensor operations work correctly

## Notes
This foundation ensures that subsequent model components
are built on a stable, transparent, and reproducible setup.

