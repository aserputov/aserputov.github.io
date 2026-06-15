---
layout: post
title: "RNN and LSTM: Sequence Memory From Scratch"
date: 2026-03-15
description: Building RNN and LSTM from scratch — hidden states, vanishing gradients, and why LSTM gates fix them.
tags: from-scratch
categories: from-scratch
---

## The Problem with Feedforward Networks

Word2Vec gives us good word vectors, but it treats each word independently. For language, order matters — "dog bites man" is not "man bites dog".

A recurrent neural network processes words one at a time, carrying a **hidden state** that acts as memory:

```python
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b)
```

That's it. One line. The hidden state `h` gets updated at each time step, mixing the previous state with the new input.

## Training: Backpropagation Through Time

To train an RNN, you unroll it across time steps and backpropagate through all of them. The gradient flows backward through every step:

```
h_4 ← h_3 ← h_2 ← h_1 ← h_0

Gradient at h_0 = ∂L/∂h_4 × ∂h_4/∂h_3 × ∂h_3/∂h_2 × ∂h_2/∂h_1 × ∂h_1/∂h_0
```

Each `∂h_t/∂h_{t-1}` involves multiplying by `W_hh` and the derivative of `tanh`. When these values are less than 1, the chain of multiplications drives the gradient toward zero.

## The Vanishing Gradient Problem

I trained a character-level RNN on Shakespeare. It learns short patterns fine — common words, basic grammar. But it can't maintain context over more than ~20 characters. The gradients literally vanish — information from 50 steps ago has zero influence on the current output.

This isn't a hyperparameter tuning problem. It's a fundamental issue with how RNNs propagate information.

## LSTM: The Fix

LSTM (Hochreiter & Schmidhuber, 1997) adds a **cell state** — a separate path that information can flow through without being multiplied by weights at every step:

```python
f = sigmoid(W_f @ [h_{t-1}, x_t])    # forget gate: what to erase
i = sigmoid(W_i @ [h_{t-1}, x_t])    # input gate: what to write
o = sigmoid(W_o @ [h_{t-1}, x_t])    # output gate: what to expose

c_t = f * c_{t-1} + i * tanh(W_c @ [h_{t-1}, x_t])  # cell update
h_t = o * tanh(c_t)                                    # hidden state
```

The cell state `c` is the key. The forget gate can be close to 1, letting the cell state flow through almost unchanged. This means gradients can travel back many steps without vanishing.

## Bahdanau Attention

Even with LSTM, there's a bottleneck in sequence-to-sequence models: the entire input gets compressed into a single fixed-size vector. Long inputs lose information.

Bahdanau attention (2014) lets the decoder look back at all encoder states and decide which ones are relevant for the current output step. This was the precursor to the transformer's self-attention.

## Results

The LSTM text generator produces noticeably better output than the RNN — it maintains character-level consistency over longer spans and handles patterns like matching brackets and quotes.

The real lesson: gradient flow determines what a network can learn. If gradients can't reach early layers, those layers can't learn long-range patterns. Everything after this — transformers, residual connections, layer normalization — is in some way about keeping gradients alive.

**Code:** [github.com/aserputov/rnn-from-scratch](https://github.com/aserputov/rnn-from-scratch)
