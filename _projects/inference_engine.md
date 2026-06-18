---
layout: page
title: Inference Engine
description: From-scratch inference engine — KV-cache (2.6x), PagedAttention, Flash Attention (Triton GPU kernel), prefix caching, continuous batching.
img:
importance: 1
category: from-scratch
github: https://github.com/aserputov/inference-engine
---

A complete inference engine for GPT-2 implementing every major optimization used in production LLM serving (vLLM, TGI, TensorRT-LLM) — written from scratch.

### Optimizations Implemented

**KV-Cache (2.6x speedup)**
Cache Key and Value matrices from past tokens. Reduces decode from O(n) recomputation to O(1) per step. Trade-off: O(n x layers x d_model) memory per request.

**PagedAttention (1.16x over torch.cat)**
Pre-allocated page pool eliminates torch.cat's O(n^2) copy overhead. Each new token writes directly to the next slot — zero copies, zero allocations. Same concept as OS virtual memory.

**Flash Attention (custom Triton GPU kernel)**
Tiles Q x K^T computation in GPU SRAM, avoiding the O(T^2) memory attention scores matrix. Implemented online softmax for incremental softmax computation across tiles. Correctness verified against standard attention at all sequence lengths.

**Prefix Caching (reuse KV across requests)**
Cache KV pages for shared prefixes (system prompts). On cache hit, skip prefill entirely for the cached portion — only compute the unique user query. Built on top of PagedAttention page pool.

**Continuous Batching**
Background scheduler with per-request KV-cache and output queues. New requests fill finished slots immediately — no static batch waiting.

**Streaming (SSE)**
Token-by-token output via Server-Sent Events with Flask.

### Benchmark Results

```
CPU (Apple M1):
  No cache:    18.8 tokens/sec
  KV-cache:    49.4 tokens/sec (2.6x)
  PagedAttn:   57.4 tokens/sec (1.16x over torch.cat)

GPU (Tesla T4, Google Colab):
  GPU Flash:   51.5 tokens/sec
  CPU cache:   18.2 tokens/sec
  Speedup:     2.8x (GPU over CPU)

Memory (Flash Attention):
  T=1024: 256x less memory (48MB → 0.188MB)
```

### From-Scratch Series

Project 5 of 5: [Word2Vec](https://github.com/aserputov/word2vec-from-scratch) → [RNN/LSTM](https://github.com/aserputov/rnn-from-scratch) → [Transformer](https://github.com/aserputov/attention-from-scratch) → [GPT-2](https://github.com/aserputov/gpt2-from-scratch) → **Inference Engine**

### Tech Stack

Python, PyTorch, Triton (GPU kernels), CUDA, Flask
