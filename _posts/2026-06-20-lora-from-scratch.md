---
layout: post
title: "LoRA: Fine-Tuning 0.12% of the Parameters"
date: 2026-06-20
description: How LoRA freezes 124M base parameters and trains only 147K by adding low-rank A,B matrices alongside Q and V projections.
tags: from-scratch inference
categories: from-scratch
---

## The Problem with Full Fine-Tuning

GPT-2 has 124M parameters. Fine-tuning all of them means:

- **Backward pass** computes gradients for 124M params
- **Adam optimizer** stores 2 extra copies per param (momentum + variance) = 3x model size in memory
- **Storage**: every fine-tuned version is a full 500MB checkpoint
- **Serving**: swapping between tasks means loading entirely different models

For one task that's manageable. For 100 tasks (customer support, translation, summarization, code...) it's 50GB of checkpoints and no way to hot-swap.

## The Idea

Most of the knowledge is already in the pretrained weights. Fine-tuning only adjusts behavior slightly. The weight update matrix `delta_W` during fine-tuning is empirically **low-rank** — it can be decomposed into two small matrices.

Instead of updating the full weight matrix W (768x768), add a parallel branch:

```
                    ┌─────────┐
              ┌────►│  W      │────┐
              │     │ (frozen)│    │
    x ────────┤     └─────────┘    ├────► output
              │     ┌────┐ ┌────┐ │
              └────►│ A  │►│ B  │─┘
                    │4×768│ │768×4│
                    └────┘ └────┘
```

W stays frozen. A and B are the only trainable parameters. The forward pass computes `W(x) + B(A(x)) * alpha`.

## Why Q and V Only

In each transformer block, the combined QKV projection `c_attn` maps 768 → 2304 (three copies of 768 for Q, K, V). We add LoRA to:

- **Q slice** (0:768) — what the token is looking for
- **V slice** (1536:2304) — what the token provides as content

K is skipped. The original LoRA paper found that Q+V gives the best quality per parameter. Adding K doesn't help much because Q and K are mathematically symmetric in the attention score — adjusting Q already changes which keys get attended to.

## The Math

For rank r=4, d_model=768, applied to Q and V in 12 blocks:

```
Per block:  2 × (A: 768×4 + B: 4×768) = 2 × 6,144 = 12,288 params
Total:      12 blocks × 12,288 = 147,456 trainable params

Base model: 124,587,264 params (frozen)
LoRA:       147,456 params (trainable)
Ratio:      0.12%
```

## Where the Savings Come From

The forward pass is the same cost — all 124M params are used. LoRA saves on everything else:

**Backward pass**: gradients computed only for 147K params instead of 124M. The chain rule still flows through frozen layers (to compute gradients for LoRA params deeper in the network), but no weight updates are stored for frozen params.

**Optimizer memory**: Adam stores momentum and variance per trainable param. Full fine-tuning: 124M × 3 × 4 bytes ≈ 1.5GB. LoRA: 147K × 3 × 4 bytes ≈ 1.7MB.

**Storage**: full checkpoint is 500MB. LoRA adapter is 576KB. You can store 850 different adapters in the space of one full model.

**Serving**: base model stays in memory. Swap adapters by loading 576KB — instant task switching.

## Implementation

The `LoRALinear` wrapper keeps the original layer frozen and adds A,B pairs for each target slice:

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0, target_slices=None):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha

        # For GPT-2's combined QKV: target_slices = [(0, 768), (1536, 2304)]
        for start, end in target_slices:
            out_dim = end - start
            A = nn.Linear(in_features, rank, bias=False)     # 768 → 4
            B = nn.Linear(rank, out_dim, bias=False)          # 4 → 768
            nn.init.zeros_(B.weight)  # start with zero delta

    def forward(self, x):
        base_out = self.original(x)           # frozen W(x)
        for (start, end), (A, B) in zip(...):
            lora_out = B(A(x)) * self.alpha   # low-rank delta
            base_out[:, :, start:end] += lora_out
        return base_out
```

B is initialized to zeros so at the start, LoRA has zero effect — the model behaves exactly like the original.

## Results

Fine-tuned GPT-2 on Shakespeare quotes (1,111 tokens, 5 epochs, 12 seconds on M1):

```
Before: "Shakespeare said: I think it's a good idea to have a lot
         of people who are not in the business of writing..."

After:  "Shakespeare said, 'A man must undergo the mind. He must
         suffer the slings and arrows, or he must suffer the bad fortune."
```

Training loss: 3.58 → 0.24

```
Base params:    124,587,264 (frozen)
LoRA params:    147,456 (trainable, 0.12%)
Adapter size:   576 KB (48 tensors)
Training time:  11.9s
```

48 tensors = 2 pairs (A, B) x 2 slices (Q, V) x 12 transformer blocks.

## Why This Matters for Serving

LoRA is how production LLMs handle multi-tenant fine-tuning. One base model in GPU memory, hundreds of adapters on disk. When a request comes in for "customer-support-v3", load 576KB, add the delta to the forward pass, serve. Next request is "code-review-v2" — swap adapter, same base model.

Combined with PagedAttention and continuous batching, you can serve different LoRA adapters in the same batch — each request gets its own adapter applied during the attention computation.

**Code:** [github.com/aserputov/inference-engine](https://github.com/aserputov/inference-engine)
