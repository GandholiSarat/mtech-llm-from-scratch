# Day 3 — Tokenization with TikToken

## Objective
Replace character-level tokenization with a realistic
subword tokenizer used in production LLMs.

## Why Tokenizers Matter
- Raw text is inefficient for neural models
- Subword tokenization reduces sequence length
- Enables better generalization and faster training

## Design Choice: TikToken
- Used OpenAI's GPT-2 BPE tokenizer via `tiktoken`
- Provides industry-grade tokenization
- Avoids reinventing token compression

## Architectural Decision
- Wrapped `tiktoken` inside a custom tokenizer class
- Ensures tokenizer can be swapped later
- Keeps dataset and model code tokenizer-agnostic

## Comparison with Char-Level Tokenization
- Fewer tokens per sentence
- Larger vocabulary
- Better alignment with real LLMs


