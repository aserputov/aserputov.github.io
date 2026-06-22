---
layout: post
title: "Reasoning Models: Teaching GPT-2 to Think Step by Step"
date: 2026-06-22
description: How reasoning models work — Chain-of-Thought SFT, Process Reward Models, Best-of-N selection, and tree search, implemented from scratch.
tags: from-scratch
categories: from-scratch
---

## The Problem

Standard LLMs generate answers in one shot. Ask "What is 23 + 45?" and the model immediately predicts the next token — no intermediate computation. For simple questions this works, but for multi-step reasoning it fails because the model has to compress the entire solution into a single forward pass.

## Chain-of-Thought (CoT)

Force the model to show its work:

```
Before: "Q: 23 + 45? A: 68"

After:  "Q: 23 + 45?
         <thinking>
         Step 1: I need to add 23 and 45.
         Step 2: 20 + 40 = 60. 3 + 5 = 8.
         Step 3: 60 + 8 = 68.
         </thinking>
         Answer: 68"
```

Each step is a separate generation — the model can use the output of Step 1 as input to Step 2. This is strictly more powerful than one-shot because the model gets intermediate "scratch space."

Training is simple SFT (LoRA fine-tuning) on examples with step-by-step format. The model learns to generate `<thinking>` blocks before answering.

## Process Reward Model (PRM)

A reward model that scores **each step**, not just the final answer:

```python
class ProcessRewardModel(nn.Module):
    def __init__(self, base_model):
        self.base = base_model                    # GPT-2
        self.step_head = nn.Linear(768, 1)        # hidden → score

    def score_full(self, input_ids):
        hidden = self.base.get_hidden(input_ids)
        return sigmoid(self.step_head(hidden[:, -1, :]))
```

Trained on pairs: correct reasoning chains get score 1.0, wrong chains get 0.0. Same Bradley-Terry approach as RLHF reward models.

After training:
```
Correct: "Step 1: add 12 and 15. Step 2: 12+15=27. Answer: 27"  → 0.984
Wrong:   "Step 1: multiply 12 and 15. Step 2: 12*15=180"        → 0.004
```

## Best-of-N Selection

Generate N complete reasoning chains, score each with PRM, return the best:

```
Chain 1: Step1→Step2→Step3  PRM: 0.92  ← winner
Chain 2: Step1→Step2→Step3  PRM: 0.71
Chain 3: Step1→Step2→Step3  PRM: 0.45
```

Simple but effective. 3x compute for significantly better accuracy. This is what early versions of reasoning models used.

## Tree Search

Smarter than Best-of-N — prune bad branches early instead of generating full chains:

```
Step 1: [3 variants] → PRM scores → keep top 2
Step 2: [3 variants each] → PRM scores → keep top 2
Step 3: [3 variants each] → PRM scores → best chain
```

Don't waste compute on chains that went wrong at Step 1. This is closer to how o1/o3 work — allocating more compute to harder problems by exploring more branches.

## Results on GPT-2 124M

The pipeline works, but GPT-2 is too small to actually reason:

```
SFT:  learned the Step 1/2/3 format (loss: 1.78 → 0.06)
PRM:  0.984 vs 0.004 — correctly distinguishes good/bad reasoning
CoT:  "8 * 7 = 56" (correct) alongside "8 * 7 = 26" (wrong)
```

GPT-2 doesn't compute — it pattern-matches from training data. On trained examples it sometimes gets the right answer, on new questions it guesses randomly. The format is correct, the math is not.

This is expected. Real reasoning models (o1, Claude) use 100-1000x larger models that actually learned arithmetic from billions of training tokens. The pipeline is identical — CoT + PRM + search — just at massive scale.

## How the "Thinking" UI Works

When you see Claude or ChatGPT "thinking" — those are real tokens being generated. The model literally outputs:

```
<thinking>I need to figure out...the user is asking about...</thinking>
The answer is...
```

The UI just renders `<thinking>` tokens in a gray collapsed block. No special mechanism — same autoregressive generation, different formatting.

**Code:** [github.com/aserputov/reasoning-from-scratch](https://github.com/aserputov/reasoning-from-scratch)
