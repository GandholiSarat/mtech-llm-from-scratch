# Day 5 — Causal Self-Attention (Core of GPT)

## Objective
Implement and deeply understand **causal self-attention**, the core
mechanism behind GPT-style autoregressive language models.

This component is the *heart of the Transformer* and one of the most
frequently discussed topics in machine learning interviews.

---

## Why Self-Attention Exists

### Problems with Earlier Sequence Models
- RNNs and LSTMs process tokens sequentially
- Difficult to model long-range dependencies
- Poor parallelization → slow training

### What Self-Attention Solves
- Allows each token to directly attend to all other tokens
- Captures long-range dependencies efficiently
- Enables parallel computation across the sequence

Self-attention completely replaces recurrence in Transformers.

---

## Input to Self-Attention

At this stage, the model input consists of:
- Token embeddings
- Positional embeddings

### Tensor Shape Convention
The input tensor has shape:

(B, T, C)

Where:
- **B** — Batch size  
- **T** — Sequence length (context window)  
- **C** — Embedding dimension (channels)

Each token is represented as a vector in R^C.

---

## Queries, Keys, and Values (Q, K, V)

### What Are Q, K, and V?
They are **learned linear projections** of the input embeddings.

Q = x · Wq  
K = x · Wk  
V = x · Wv  

Where:
- Wq, Wk, Wv ∈ R^(C × C)
- Q, K, V ∈ R^(B × T × C)

### Why Separate Q, K, V?
- Queries: what the token is looking for
- Keys: what the token contains
- Values: information to be passed forward

This separation increases model expressiveness.

---

## Scaled Dot-Product Attention

### Attention Score Computation

AttentionScores = Q · Kᵀ / √C

Shapes:
- Q: (B, T, C)
- Kᵀ: (B, C, T)
- Scores: (B, T, T)

Each row corresponds to how much one token attends to all others.

---

## Why Divide by √C?

### Problem Without Scaling
- Dot products grow large when C is large
- Softmax becomes extremely peaked
- Gradients vanish → unstable training

Scaling prevents softmax saturation and stabilizes gradients.

---

## Causal Masking (Critical for GPT)

### Why Masking Is Needed
Language modeling is **autoregressive**:
- Token at position t must not see tokens at t+1, t+2, …

Without masking:
- The model cheats using future tokens
- Training objective becomes meaningless

### Causal Mask Structure

1 0 0 0  
1 1 0 0  
1 1 1 0  
1 1 1 1  

Future positions are set to -∞ before softmax.

---

## Softmax and Attention Weights

AttentionWeights = softmax(AttentionScores)

- Shape: (B, T, T)
- Each row sums to 1
- Represents attention distribution

---

## Computing Attention Output

Output = AttentionWeights · V

Shapes:
- AttentionWeights: (B, T, T)
- V: (B, T, C)
- Output: (B, T, C)

Each token output is context-aware.

---

## Output Projection

Final linear projection:
Output = Output · Wo

Purpose:
- Mix information across channels
- Preserve embedding dimension

---

## Full Shape Flow Summary

Input x              : (B, T, C)  
Q, K, V              : (B, T, C)  
Attention scores     : (B, T, T)  
Masked scores        : (B, T, T)  
Attention weights    : (B, T, T)  
Attention output     : (B, T, C)  
Final projection     : (B, T, C)  

---

## Why This Is the Heart of GPT
- Enables global context modeling
- Removes recurrence entirely
- Allows full parallelism
- Scales with data and model size

---

## Common Questions & Answers

**Why is attention O(T²)?**  
Because it computes interactions between every pair of tokens.

**Why keep output shape (B, T, C)?**  
To allow residual connections and stacking.

**What happens without masking?**  
The model sees future tokens and training breaks.

---

## Key Points to Remember
- Self-attention replaces recurrence
- Causal masking enforces autoregression
- Q, K, V are learned projections
- √C scaling stabilizes training
- Output shape always remains (B, T, C)

---

## Next Steps
- Multi-head attention
- Feedforward network
- Transformer block
