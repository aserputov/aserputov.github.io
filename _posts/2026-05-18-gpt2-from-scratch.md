---
layout: post
title: "GPT-2: Loading Real Weights and Generating Text"
date: 2026-05-18
description: Writing GPT-2's forward pass from scratch, loading OpenAI's 124M weights, and generating coherent English text.
tags: from-scratch transformer
categories: from-scratch
---

## The Goal

Take OpenAI's published GPT-2 weights (124M parameters), write the entire forward pass by hand, and generate real text. No HuggingFace, no abstractions — just matrix operations.

## Architecture

GPT-2 is a decoder-only transformer. The full structure:

```
Token IDs
  → wte[token_id]  (token embedding, 50257 × 768)
  → + wpe[position] (position embedding, 1024 × 768)
  → 12 × Transformer Block
  → LayerNorm
  → logits = x @ wte.T  (weight tying)
  → sample next token
```

Weight tying: the same embedding matrix used to convert token IDs to vectors is reused (transposed) to convert the final hidden state back to vocabulary logits. This means the model has fewer parameters and the embedding space stays consistent.

## Loading the Weights

OpenAI published the weights as TensorFlow checkpoints. Loading them into a from-scratch PyTorch model means mapping every parameter name:

```python
# OpenAI's naming → our model's structure
"model/h0/attn/c_attn/w"  → blocks[0].attn.c_attn.weight
"model/h0/mlp/c_fc/w"     → blocks[0].mlp.c_fc.weight
# ... 148 parameters total
```

If any single weight is loaded wrong — transposed when it shouldn't be, assigned to the wrong layer — the output is garbage. Getting coherent English text out confirms the implementation is correct.

## GELU Activation

GPT-2 uses GELU instead of ReLU in the feed-forward layers:

```python
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
```

Unlike ReLU which hard-clips at 0, GELU has a smooth curve that allows small negative values through. This means neurons don't "die" as easily during training.

## Text Generation

The model predicts a probability distribution over the entire vocabulary for the next token. How you sample from that distribution controls the output:

```python
logits = model(tokens)[:, -1, :]     # last token's predictions
logits = logits / temperature         # sharpen or flatten distribution

# Top-k: only consider the k most likely tokens
top_k_logits, top_k_indices = torch.topk(logits, k=40)
probs = softmax(top_k_logits)
next_token = top_k_indices[torch.multinomial(probs, 1)]
```

- **Temperature < 1**: more deterministic, sticks to obvious completions
- **Temperature > 1**: more random, more creative/weird
- **Top-k**: prevents sampling extremely unlikely tokens

## What It Produces

```
Prompt: "The meaning of life is"
Output: "The meaning of life is not to be found in the
         pursuit of happiness, but in the quiet moments
         of reflection that come when we least expect them."
```

Coherent English from raw matrix multiplications. Every operation — every attention head, every layer norm, every GELU — done by hand.

## What I Took Away

GPT-2 is not complicated. It's the same transformer block from the previous project, repeated 12 times, with specific weight shapes and an embedding/unembedding layer. The complexity is in the training (which OpenAI did) and in the inference optimizations needed to make it fast (which is the next project).

**Code:** [github.com/aserputov/gpt2-from-scratch](https://github.com/aserputov/gpt2-from-scratch)
