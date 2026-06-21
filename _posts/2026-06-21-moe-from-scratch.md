---
layout: post
title: "Mixture of Experts: 8x Parameters, Same Compute"
date: 2026-06-21
description: How MoE replaces the dense FFN with multiple experts and a learned router, achieving large model quality at small model speed.
tags: from-scratch
categories: from-scratch
---

## The Problem with Dense Models

In a standard transformer, every token passes through every parameter. A 47B model does 47B operations per token. Want better quality? Make the model bigger. But bigger = slower.

Most of those operations are wasted — a token about math doesn't need the weights that handle poetry. But a dense model has no choice: everything goes through everything.

## The Idea

Replace the single FFN with multiple smaller FFNs (experts) and a router that picks which ones to use:

```
Dense:   token → Attention → FFN → output
                              ↑
                         all parameters

MoE:     token → Attention → Router → Expert 2,5 → output
                              ↑              ↑
                         picks 2 of 8    only 2 compute
```

8 experts = 8x the parameters (knowledge), but each token only uses 2 = same compute per token.

## Router

The router is one linear layer:

```python
self.router = nn.Linear(768, 8, bias=False)  # hidden → 8 scores
```

For each token, it produces 8 scores, takes top-2, normalizes their weights, and routes:

```python
router_probs = softmax(router(hidden_state))     # [0.05, 0.35, 0.1, 0.3, ...]
top_k_indices = topk(router_probs, k=2)          # experts 1 and 3
top_k_weights = normalize([0.35, 0.3])           # [0.54, 0.46]

output = 0.54 * expert_1(token) + 0.46 * expert_3(token)
```

The router learns through backprop — no manual assignment of "this expert handles math."

## Experts

Each expert is an identical FeedForward network:

```python
self.experts = nn.ModuleList([
    FeedForward(d_model) for _ in range(8)
])
```

They start identical and specialize during training. One might learn to handle numbers, another syntax, another rare words — but these aren't human-interpretable categories.

## Load Balancing

Without a penalty, the router sends everything to 1-2 experts and ignores the rest. The load balancing loss keeps usage even:

```python
# fraction: what % of tokens each expert got
# avg_prob: average router probability for each expert
balance_loss = num_experts * (fraction * avg_prob).sum()
```

Minimum when all experts get equal traffic (~12.5% each for 8 experts). Added to the main loss with a small coefficient.

## Per-Token, Not Per-Request

Routing happens at the token level, not the sequence level. Within one sentence, "cat" might go to expert 3 while "jumped" goes to expert 5. Different tokens need different processing — even within the same topic.

## Results

Trained a small MoE (d=256, 4 layers, 8 experts) vs equivalent dense model:

```
Dense:  16.3M params, all active        loss: 0.33
MoE:    31.0M params, ~18.4M active     loss: 0.45

MoE has 1.9x params but ~1.1x active compute
Router usage: ~12-13% per expert (balanced)
```

At this small scale, MoE overhead from routing slightly hurts. The benefit shows at scale:

```
Mixtral 8x7B:      47B total, 13B active  ≈  LLaMA 2 70B quality
DeepSeek-V3:       256 experts, top-8     →  3% of params active per token
```

## The Trend

More experts, each smaller:

```
Mixtral (2023):      8 experts,  top-2  → 25% active
DeepSeek-V2 (2024):  160 experts, top-6 → 4% active
DeepSeek-V3 (2024):  256 experts, top-8 → 3% active
```

Finer-grained experts = less wasted compute. The limit: too many tiny experts and the router can't pick accurately, plus routing overhead grows.

## Why This Matters

GPT-4 is widely believed to be a MoE model. Most frontier models use some form of MoE — it's the only way to scale knowledge (parameters) without proportionally scaling inference cost (compute). Understanding MoE is understanding how modern LLMs achieve their capability while remaining servable.

**Code:** [github.com/aserputov/moe-from-scratch](https://github.com/aserputov/moe-from-scratch)
