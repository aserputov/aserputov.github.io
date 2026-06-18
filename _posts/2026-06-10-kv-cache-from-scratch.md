---
layout: post
title: "KV-Cache: Why Naive Inference is O(n²) and How Caching Fixes It"
date: 2026-06-10
description: How KV-cache reduces autoregressive generation from O(n²) to O(n), with from-scratch implementation and 2.6x speedup benchmarks.
tags: from-scratch inference
categories: from-scratch
---

## The Problem

When generating text with a transformer, we produce one token at a time. Without optimization, generating token 100 means running the full model on all 100 tokens — even though tokens 1-99 haven't changed.

```
Step 1:  process [The]                    → predict "meaning"
Step 2:  process [The, meaning]           → predict "of"       ← recomputes "The"
Step 3:  process [The, meaning, of]       → predict "life"     ← recomputes both
...
Step 99: process [all 99 tokens]          → predict token 100  ← recomputes ALL
```

Total attention computations: 1 + 2 + 3 + ... + n = **O(n²)**

## The Insight

In attention, we compute Q, K, V for each token. But K and V for past tokens **never change** — the causal mask means past tokens can't see the future. So we cache them.

## Two Phases

| Phase       | Input                    | Work                                               |
| ----------- | ------------------------ | -------------------------------------------------- |
| **Prefill** | Entire prompt (N tokens) | Process all at once, build KV-cache                |
| **Decode**  | 1 new token              | Compute Q,K,V for new token only, reuse cached K,V |

## Implementation

```python
def forward(self, x, kv_cache=None):
    q, k, v = self.c_attn(x).split(C, dim=2)

    if kv_cache is not None:
        prev_k, prev_v = kv_cache
        k = torch.cat([prev_k, k], dim=2)  # append new K to cached
        v = torch.cat([prev_v, v], dim=2)  # append new V to cached

    new_cache = (k, v)
    scores = (q @ k.T) / sqrt(d)
    return output, new_cache
```

## Results

```
No cache:   5.32s  |  18.8 tokens/sec
KV-cache:   2.02s  |  49.4 tokens/sec
Speedup:    2.63x
```

The speedup grows with sequence length — longer generations benefit more.

## The Trade-off

KV-cache trades **memory for speed**. Each request stores O(n × layers × d_model) of cached tensors. This is why context length is expensive — longer context means bigger cache means more memory per request. Understanding this trade-off is fundamental to LLM serving systems like vLLM and TGI.

## What's Next

The naive `torch.cat` approach has its own O(n²) problem — it copies the entire cache at every step. [PagedAttention](/blog/2026/pagedattention-from-scratch/) solves this with pre-allocated memory pages.

**Code:** [github.com/aserputov/inference-engine](https://github.com/aserputov/inference-engine)
