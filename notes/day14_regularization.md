# Day 14 — Regularization: Dropout & Training Stability

## Objective
Reduce overfitting and improve generalization by adding
dropout and proper regularization to the GPT-style model.

This aligns the implementation closer to GPT-2–style training
and explains why regularization matters in language models.

---

## Why Regularization Is Needed

Observed during training:
- Very low training loss
- Strong memorization of the corpus
- Repetition collapse during early decoding

Root causes:
- Small dataset
- High model capacity
- No regularization initially applied

Regularization helps the model learn more robust representations.

---

## Dropout in Transformer Models

Dropout is a training-time regularization technique that
randomly disables a fraction of activations to prevent
co-adaptation of neurons.

### Correct Dropout Locations

Dropout is applied:

1. **After attention output projection**
   - Prevents overfitting in attention outputs

2. **After feedforward (MLP) layers**
   - Feedforward layers contain most parameters

3. **On token + positional embeddings**
   - Regularizes early representations

### Incorrect Dropout Locations

Dropout should NOT be applied:
- Inside softmax
- On attention weights directly
- On logits
- During inference

---

## Dropout Configuration

Dropout probability:
```
p = 0.1
```

This matches the default used in GPT-2.

---

## Weight Decay with AdamW

AdamW is used with weight decay:
```
weight_decay = 0.01
```

Benefits:
- Decouples weight decay from gradient updates
- Prevents parameter explosion
- Improves training stability

This matches modern Transformer training practices.

---

## Expected Effects

After adding dropout and regularization:

- Training loss decreases more slowly
- Final loss is higher than before
- Overfitting is reduced
- Generated text becomes more diverse
- Sampling strategies become more effective

These are **desired outcomes**, not regressions.

---

## Q&A

**Q: Why use dropout in Transformers?**  
A: To prevent co-adaptation and improve generalization, especially in high-capacity layers.

**Q: Why is dropout disabled during inference?**  
A: Inference must be deterministic; dropout is a training-time regularizer.

**Q: Why apply dropout after attention and MLP layers?**  
A: These layers contribute most to model capacity and overfitting.

---

## Outcome of Day 14

- Dropout added in correct architectural locations
- Weight decay understood and justified
- Overfitting reduced
- Model aligned closer to GPT-style training
- Project made extension-ready for pretrained weights

---

## Next Step

Optional extensions:
- Load GPT-2 pretrained weights
- Compare layer-wise activations
- Evaluate perplexity
- Scale training with more data or GPU
