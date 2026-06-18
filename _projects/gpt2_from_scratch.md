---
layout: page
title: GPT-2 From Scratch
description: Complete GPT-2 (124M) forward pass from scratch — loading real OpenAI weights, generating coherent text.
img:
importance: 3
category: from-scratch
github: https://github.com/aserputov/gpt2-from-scratch
---

Built the entire GPT-2 forward pass from scratch and loaded real OpenAI weights (124M parameters). No HuggingFace inference — raw matrix operations producing coherent English text.

### What I Learned

- GPT-2 is decoder-only: 12 layers, 12 heads, 768 d_model, 50,257 vocab
- Pre-norm architecture (LayerNorm before attention, not after)
- Weight tying — embedding matrix reused as output projection (logits = x @ wte.T)
- GELU activation vs ReLU — smoother gradient flow
- Autoregressive generation — predict one token, append, repeat
- Top-k sampling and temperature — controlling randomness vs determinism

### Architecture

```
Token IDs → Embedding + Position → 12× [LayerNorm → Attention → LayerNorm → FFN] → LayerNorm → Logits
```

### From-Scratch Series

Project 4 of 5: [Word2Vec](https://github.com/aserputov/word2vec-from-scratch) → [RNN/LSTM](https://github.com/aserputov/rnn-from-scratch) → [Transformer](https://github.com/aserputov/attention-from-scratch) → **GPT-2** → [Inference Engine](https://github.com/aserputov/inference-engine)

### Tech Stack

Python, PyTorch, tiktoken
