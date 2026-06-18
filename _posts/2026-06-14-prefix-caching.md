---
layout: post
title: "Prefix Caching: Reusing KV Across Requests"
date: 2026-06-14
description: Why every request recomputes the same system prompt, and how prefix caching eliminates redundant prefill by sharing KV pages.
tags: from-scratch inference
categories: from-scratch
---

## The Problem

Every ChatGPT-style request starts with the same system prompt:

```
"You are a helpful assistant that answers concisely..."  ← same for everyone
"What is Flash Attention?"                                ← unique per user
```

Without prefix caching, every request runs prefill on the full prompt — including the system prompt that hasn't changed. 100 users = 100 identical prefills of the same 500 tokens.

## The Insight

KV values for a prefix are deterministic. If the same tokens appear at the same positions, they produce the same K,V at every layer. This is because:

- **Layer 1**: K,V depend on token embedding + position embedding. Same tokens, same positions = same K,V.
- **Layer 2+**: K,V depend on the output of the previous layer, which mixed information from all previous tokens via attention. Same previous tokens = same mixed output = same K,V.

This only works for **prefixes** — tokens from the beginning. You can't cache "thanks for your help" from the middle of one conversation and reuse it in another, because by that point K,V depend on everything that came before.

## Implementation

Built on top of PagedAttention. The prefix cache is a dict mapping token tuples to page table entries:

```python
class PrefixCache:
    def __init__(self, ...):
        # Same page pool as PagedAttention
        self.k_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)
        self.v_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)

        # NEW: prefix → page table mapping
        self.cache = {}   # hash(prefix_tokens) → page references

    def lookup(self, tokens):
        # Find longest matching prefix in cache
        key = tuple(tokens)
        for cached_key, entry in self.cache.items():
            match = common_prefix_length(cached_key, key)
            if match > best:
                best = match
        return best, entry
```

On cache miss — prefill the prefix, save KV pages to cache, then prefill remaining tokens:

```python
# Cache MISS: first request with this system prompt
logits = prefill(prefix_tokens, start_pos=0)
prefix_cache.save_prefix(prefix_tokens, seq_id)  # save for future
logits = prefill(question_tokens, start_pos=len(prefix_tokens))
```

On cache hit — clone page table pointers, prefill only the unique part:

```python
# Cache HIT: same system prompt seen before
prefix_cache.clone_pages_from_entry(seq_id, entry, hit_len)
logits = prefill(question_tokens, start_pos=hit_len)  # skip system prompt
```

## Why Hash Tokens, Not Text?

Different text can produce different tokens:

```
"You are"     → [1639, 389]       # normal
"You  are"    → [1639, 220, 389]  # extra space = different tokens
```

The model operates on tokens. KV is computed from tokens. So the cache key must be tokens.

## Benchmark Results

```
System prompt: 13 tokens | 5 questions | 30 tokens each

Without prefix caching:  4.72s  (31.8 tokens/sec)
With prefix caching:     4.41s  (34.0 tokens/sec)
Speedup:                 1.07x
Prefill tokens saved:    52 (system prompt computed once)
```

The speedup is small because our system prompt is only 13 tokens. In production, system prompts are 500-2000 tokens — that's where prefix caching saves significant compute. Anthropic's prompt caching, for example, reduces costs by up to 90% on cached prefixes.

## Why This Matters

This is how every major API (Claude, GPT-4, Gemini) handles system prompts efficiently. Without it, serving costs would scale linearly with the number of requests sharing the same prefix. Understanding this connects PagedAttention (memory management) to real serving economics.

**Code:** [github.com/aserputov/inference-engine](https://github.com/aserputov/inference-engine)
