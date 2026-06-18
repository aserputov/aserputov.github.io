---
layout: post
title: "Flash Attention: Writing a GPU Kernel in Triton"
date: 2026-06-13
description: How Flash Attention avoids materializing the O(T²) attention matrix by tiling Q×K^T in GPU SRAM, with a from-scratch Triton kernel.
tags: from-scratch inference gpu
categories: from-scratch
---

## The O(T²) Problem

Standard attention computes the full scores matrix:

```python
scores = Q @ K.T       # T×T matrix — O(T²) memory
weights = softmax(scores)
output = weights @ V
```

For T=1024, that's 1 million entries. For T=128K (modern LLMs), it's 16 billion — doesn't fit in any GPU.

## The Key Insight

We don't need the full T×T matrix at once. We can compute attention in **tiles** — small blocks that fit in GPU SRAM (fast on-chip memory, ~20MB) instead of HBM (main GPU memory, ~8-80GB).

```
Instead of:  compute entire 1024×1024 matrix
Do this:     compute 64×64 tiles, one at a time, accumulate result
```

## Online Softmax

The trick is computing softmax incrementally. Normally softmax needs ALL scores. But with the online algorithm, we maintain a running max and running sum:

```python
# Process K tiles one at a time
for each K_tile:
    scores = Q_tile @ K_tile.T           # 64×64 — fits in SRAM
    m_new = max(m_old, max(scores))       # update running max
    alpha = exp(m_old - m_new)            # rescale factor
    p = exp(scores - m_new)               # softmax numerator
    acc = acc * alpha + p @ V_tile        # rescale old + add new
    l = l * alpha + sum(p)                # update denominator

output = acc / l  # final normalization — identical to standard softmax
```

## The Triton Kernel

[Triton](https://github.com/triton-lang/triton) (by OpenAI) lets you write GPU kernels in Python that compile to PTX (NVIDIA assembly). Here's the core of the kernel:

```python
@triton.jit
def _flash_attn_fwd(Q, K, V, Out, ...):
    # Each GPU "thread group" processes one tile of queries
    pid_m = tl.program_id(0)   # which Q tile (0..T/64)
    pid_z = tl.program_id(1)   # which batch × head

    q = tl.load(q_ptrs)        # load 64 rows of Q into SRAM
    q = q * (1.0 / sqrt(D))

    # Iterate over K,V tiles
    for start_n in range(0, causal_bound, BLOCK_N):
        kt = tl.load(k_ptrs)   # load K tile (transposed)
        s = tl.dot(q, kt)      # 64×64 matmul in SRAM
        s = tl.where(causal_mask, s, -inf)

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new)
        v = tl.load(v_ptrs)
        acc = acc * tl.exp(m_i - m_new) + tl.dot(p, v)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(p, 1)
        m_i = m_new

    tl.store(o_ptrs, acc / l_i)
```

## GPU Benchmark Results (Tesla T4)

**Attention kernel (standard vs flash):**

| Seq Length | Standard | Flash  | Match |
| ---------- | -------- | ------ | ----- |
| 128        | 0.16ms   | 0.30ms | yes   |
| 256        | 0.32ms   | 0.85ms | yes   |
| 512        | 0.88ms   | 2.16ms | yes   |
| 1024       | 2.66ms   | 7.78ms | yes   |

Flash is **slower** at GPT-2's short context (max 1024) because the T×T matrix fits in GPU memory. The real win is **memory**:

| Seq Length | Standard | Flash    | Savings |
| ---------- | -------- | -------- | ------- |
| 256        | 3.0 MB   | 0.188 MB | 16x     |
| 512        | 12.0 MB  | 0.188 MB | 64x     |
| 1024       | 48.0 MB  | 0.188 MB | 256x    |

Flash tile is always 64×64 = constant memory, regardless of sequence length.

**Full generation:**

```
GPU (Flash + KV-cache):  51.5 tokens/sec
CPU (KV-cache only):     18.2 tokens/sec
GPU speedup:             2.8x
```

## Model Design

Flash Attention optimizes **prefill** (processing the full prompt). During decode, Q is just 1 token — no T×T matrix to tile. So the model uses:

- **Prefill**: Flash Attention (Triton kernel)
- **Decode**: Standard attention with KV-cache

This matches what production serving systems do.

## Why This Matters

Every major LLM (GPT-4, Claude, Gemini) uses Flash Attention. Without it, long-context models (128K+ tokens) would be impossible — the attention matrix simply wouldn't fit in memory. Understanding it from the kernel level up is fundamental to AI infrastructure work.

**Code:** [github.com/aserputov/inference-engine](https://github.com/aserputov/inference-engine)
