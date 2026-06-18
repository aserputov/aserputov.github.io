---
layout: page
title: Transformer From Scratch
description: Full transformer encoder-decoder — self-attention, multi-head attention, positional encoding, layer normalization.
img:
importance: 4
category: from-scratch
github: https://github.com/aserputov/attention-from-scratch
---

Built a complete transformer from scratch — self-attention, multi-head attention, positional encoding, residual connections, and layer normalization.

### What I Learned

- Self-attention: Q, K, V matrices — Query (what am I looking for), Key (what do I contain), Value (what do I give)
- Why we divide by sqrt(d_k) — prevents softmax saturation with large dot products
- Multi-head attention — multiple "perspectives" running in parallel, each learning different relationships
- Positional encoding — sin/cos functions to encode word position since attention has no inherent order
- Residual connections — skip connections that prevent vector degradation across layers

### From-Scratch Series

Project 3 of 5: [Word2Vec](https://github.com/aserputov/word2vec-from-scratch) → [RNN/LSTM](https://github.com/aserputov/rnn-from-scratch) → **Transformer** → [GPT-2](https://github.com/aserputov/gpt2-from-scratch) → [Inference Engine](https://github.com/aserputov/inference-engine)

### Tech Stack

Python, PyTorch
