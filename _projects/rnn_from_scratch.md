---
layout: page
title: RNN & LSTM From Scratch
description: RNN and LSTM with Bahdanau Attention — character-level text generation, vanishing gradient analysis.
img:
importance: 5
category: from-scratch
github: https://github.com/aserputov/rnn-from-scratch
---

Built RNN, LSTM, and sequence-to-sequence with Bahdanau Attention from scratch. Trained character-level text generators and demonstrated the vanishing gradient problem empirically.

### What I Learned
- Hidden state mechanics — how a single vector carries sequence memory
- Why RNN gradients vanish through long sequences (chain rule compounding)
- LSTM gates (forget, input, output) as the solution — cell state acts as a gradient highway
- Bahdanau Attention — letting the decoder look back at all encoder states instead of compressing into one vector

### From-Scratch Series
This is project 2 of 5: [Word2Vec](https://github.com/aserputov/word2vec-from-scratch) → **RNN/LSTM** → [Transformer](https://github.com/aserputov/attention-from-scratch) → [GPT-2](https://github.com/aserputov/gpt2-from-scratch) → [Inference Engine](https://github.com/aserputov/inference-engine)

### Tech Stack
Python, PyTorch
