---
layout: post
title: "Speculative Decoding: Trading Draft Guesses for Parallel Verification"
date: 2026-06-17
description: How speculative decoding uses a small draft model to propose tokens and a large model to verify them in one pass, achieving 2-3x speedup on GPU.
tags: from-scratch inference
categories: from-scratch
---

## Why Decode is Slow

Autoregressive generation produces one token per forward pass. Each token waits for the previous one. With KV-cache, we avoid recomputing past tokens, but the new token still passes through every layer of the model.

The problem: during decode of a single token, the GPU is loaded at maybe 5-10%. Most of the time is spent loading model weights from memory, not computing. The GPU has thousands of cores with almost nothing to do.

## The Idea

Use two models:

- **Draft model** (small, fast): generates K candidate tokens sequentially
- **Target model** (large, accurate): verifies all K candidates in one forward pass

Verification is the same operation as prefill — all tokens are known, so the GPU processes them in parallel. One forward pass to check 5 tokens costs about the same as one forward pass to generate 1 token, because the bottleneck is loading weights, not computation.

```
Standard (target only, 5 tokens):
  tok1 → tok2 → tok3 → tok4 → tok5
  5 forward passes = 500ms

Speculative:
  Draft generates:  tok1, tok2, tok3, tok4, tok5  = 150ms (small model)
  Target verifies:  [all 5 at once]                = 100ms (one pass)
  Total: 250ms for 5 tokens (if all accepted)
```

## Acceptance and Rejection

The target model checks each draft token against what it would have generated:

```python
for j in range(len(draft_tokens)):
    target_tok = argmax(target_logits[j])
    if target_tok == draft_tokens[j]:
        accept(draft_tokens[j])      # draft guessed right
    else:
        accept(target_tok)           # use target's choice
        break                        # stop here, rest is invalid
```

We accept tokens from the beginning until the first mismatch. After a wrong token, everything after it is contaminated — the K,V values depend on all previous tokens, so a wrong token means wrong context for everything downstream.

If the draft model has 70% per-token accuracy and we draft 5 tokens, we typically accept 3-4 per iteration — that's 3-4 tokens for the cost of one target forward pass.

## Why This Doesn't Lose Quality

The target model makes every final decision. Draft tokens are only kept if the target agrees. The output is mathematically identical to what the target would produce on its own. The draft model is just guessing to save time — wrong guesses cost a little draft compute but nothing else.

## Implementation

The draft model is a truncated version of the target — same architecture, fewer layers:

```python
class DraftModel(nn.Module):
    def __init__(self, full_model, n_draft_layers=10):
        super().__init__()
        self.wte = full_model.wte        # shared embeddings
        self.wpe = full_model.wpe        # shared positions
        self.blocks = full_model.blocks[:n_draft_layers]  # fewer layers
        self.ln_f = full_model.ln_f      # shared layer norm
```

In production, the draft model is typically a separately trained small model (like a distilled version), which gives much better acceptance rates.

## Results

On CPU (Apple M1), speculative decoding doesn't show speedup because CPU processes 5 tokens ~5x slower than 1 — there's no parallelism advantage. The speedup is GPU-specific, where verification of K tokens costs roughly the same as generating 1 token due to the memory-bound nature of decode.

Published benchmarks on GPU:

```
Google (T5-XXL):     2-3x speedup
NVIDIA (H200):       3.6x speedup
AMD (MI300X):        1.5-2.9x speedup
```

## Why It's Memory-Bound

```
Decode 1 token:   load 500MB of weights → multiply (1, 768) × (768, 768)
Decode 5 tokens:  load 500MB of weights → multiply (5, 768) × (768, 768)

Loading weights: ~95% of the time (same for both)
Compute:         ~5% of the time (5x more, but 5% × 5 = 25%, still small)
```

The weights have to be loaded regardless. Speculative decoding makes the GPU do useful work during the time it would otherwise spend waiting.

**Code:** [github.com/aserputov/inference-engine](https://github.com/aserputov/inference-engine)
