---
layout: post
title: "Transformer: Self-Attention From Scratch"
date: 2026-04-20
description: Building a transformer from scratch — Q/K/V attention, multi-head, positional encoding, and why it replaced RNNs.
tags: from-scratch transformer
categories: from-scratch
---

## Why Not Just Use LSTMs?

LSTMs process sequences one token at a time. Token 100 has to wait for tokens 1-99 to finish. This is inherently sequential — you can't parallelize it.

The transformer (Vaswani et al., 2017) processes all tokens simultaneously. The key mechanism: **self-attention**.

## Self-Attention

Every token produces three vectors from the same input:
- **Query (Q)**: what am I looking for?
- **Key (K)**: what do I contain?
- **Value (V)**: what information do I carry?

```python
Q = x @ W_q  # (T, d)
K = x @ W_k  # (T, d)
V = x @ W_v  # (T, d)

scores = Q @ K.T / sqrt(d)  # (T, T) — every token scores against every other
weights = softmax(scores)
output = weights @ V         # weighted combination of values
```

The T x T scores matrix is the core of the transformer. Each entry says "how much should token i attend to token j?"

Division by `sqrt(d)` prevents the dot products from getting too large, which would push softmax into regions where the gradient is near zero.

## Multi-Head Attention

Instead of one big attention, split into multiple heads that each look at a different subspace:

```python
# d_model=768, n_heads=12 → head_dim=64
# Each head learns different relationships:
# - one head might learn positional patterns
# - another might learn syntactic dependencies
# - another might track coreference
```

The outputs of all heads get concatenated and projected back to d_model.

## Positional Encoding

Self-attention treats the input as a set — there's no notion of position. "dog bites man" and "man bites dog" would produce the same output. Positional encoding (sin/cos at different frequencies) adds position information:

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

## The Full Block

```
Input
  ↓
LayerNorm → Multi-Head Attention → + (residual)
  ↓
LayerNorm → Feed-Forward Network → + (residual)
  ↓
Output
```

Residual connections let the gradient flow directly through addition. Layer normalization keeps activations stable. Stack 12 of these blocks and you get GPT-2.

## O(T^2) — The Cost

The scores matrix is T x T. For sequence length 1024, that's ~1M entries. For 128K (modern LLMs), that's 16 billion. This is why long context is expensive, and why Flash Attention exists — but that comes later.

## What I Took Away

Building this made the Q/K/V mechanism intuitive instead of abstract. The transformer isn't magic — it's matrix multiplications with a clever structure that lets every token talk to every other token in one step, instead of relying on a hidden state to carry information across the whole sequence.

**Code:** [github.com/aserputov/attention-from-scratch](https://github.com/aserputov/attention-from-scratch)
