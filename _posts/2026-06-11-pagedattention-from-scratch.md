---
layout: post
title: "PagedAttention: Virtual Memory for KV-Cache"
date: 2026-06-11
description: Why torch.cat creates O(n²) memory copies, and how PagedAttention eliminates them with pre-allocated page pools.
tags: inference pagedattention vllm optimization
categories: deep-dives
---

## The Hidden Cost of torch.cat

KV-cache gives us a 2.6x speedup by caching Key and Value matrices. But the naive implementation using `torch.cat` has a problem:

```python
k = torch.cat([prev_k, k], dim=2)  # creates NEW tensor, copies ALL previous data
```

Every step:
1. Allocate a new tensor (old size + 1)
2. Copy the entire old cache into it
3. Add the one new token
4. Deallocate the old tensor

At step 100, we copy 99 tokens to add 1. At step 1000, we copy 999. Total copies: 1 + 2 + ... + n = **O(n²)**.

## PagedAttention: The OS Analogy

Operating systems solved this decades ago with virtual memory. Instead of contiguous allocation, they use **pages** — fixed-size blocks mapped through a page table.

PagedAttention applies the same idea to KV-cache:

```
┌─────────────────────────────────────┐
│  Page Pool (pre-allocated)          │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │Page 0│ │Page 1│ │Page 2│  ...   │
│  │16 slots│16 slots│16 slots│       │
│  └──────┘ └──────┘ └──────┘        │
└─────────────────────────────────────┘

Sequence A: [Page 0, Page 2]  ← non-contiguous, that's fine
Sequence B: [Page 1, Page 4]
```

Each new token writes directly to the next slot — **zero copies, zero allocations**.

## Implementation

```python
class PagedKVCache:
    def __init__(self, n_layers, n_heads, head_dim, page_size=16, max_pages=256):
        # Pre-allocate everything upfront
        self.k_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)
        self.v_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)
        self.free_pages = list(range(max_pages))

    def append(self, seq_id, layer_idx, new_k, new_v):
        slot = position % self.page_size
        if slot == 0:  # page full, grab a new one
            page_id = self.free_pages.pop(0)
        # Write directly — no copy
        self.k_pool[page_id, :, slot, :] = new_k
        self.v_pool[page_id, :, slot, :] = new_v
```

## Results

```
torch.cat cache:  2.02s  |  49.4 tokens/sec
PagedAttention:   1.74s  |  57.4 tokens/sec
Speedup:          1.16x
```

The advantage grows with sequence length since we avoid the O(n²) copy overhead.

## Why This Matters

This is exactly how [vLLM](https://github.com/vllm-project/vllm) manages KV-cache memory in production. Understanding PagedAttention from scratch reveals why memory management is the key bottleneck in LLM serving — not compute.

**Code:** [github.com/aserputov/inference-engine](https://github.com/aserputov/inference-engine)
