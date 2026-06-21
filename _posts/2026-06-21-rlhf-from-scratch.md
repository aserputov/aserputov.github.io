---
layout: post
title: "RLHF: From Text Predictor to Assistant"
date: 2026-06-21
description: How RLHF turns a language model into an assistant — reward model training on human preferences, PPO optimization, and the full pipeline from scratch.
tags: from-scratch
categories: from-scratch
---

## The Problem

GPT-2 predicts the next token. Ask it "What is 2+2?" and it continues:

```
Q: What is 2+2?
Q: What is 3+3?
Q: What is 4+4?
```

It doesn't answer — it continues, because that's what internet text looks like. SFT (supervised fine-tuning) teaches the format: question → answer. But the model still generates wordy, hedging, wrong answers. How do you define "good answer" as a loss function?

You can't. But humans can look at two answers and say which is better. RLHF turns that human judgment into a training signal.

## Three Steps

```
GPT-2 (base) → Reward Model → PPO → GPT-2 (assistant)
```

**Step 1: Reward Model** — a model that scores text quality as a single number.

**Step 2: Human Preferences** — train the reward model on pairs: "this answer is better than that one."

**Step 3: PPO** — optimize the language model to generate text that gets high reward scores.

## Reward Model

Same GPT-2 architecture, but instead of predicting the next token, it outputs one number:

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model                           # GPT-2
        self.reward_head = nn.Linear(768, 1)             # hidden → scalar

    def forward(self, input_ids):
        hidden = self.base.get_hidden(input_ids)         # all layers
        last_hidden = hidden[:, -1, :]                   # last token
        return self.reward_head(last_hidden).squeeze(-1)  # one number
```

The last token's hidden state has seen the entire sequence (through causal attention), so it summarizes the whole text.

## Training on Preferences

Humans rank pairs of answers. The loss pushes the reward model to score chosen answers higher than rejected ones:

```python
r_chosen = reward_model(chosen_ids)      # score for good answer
r_rejected = reward_model(rejected_ids)  # score for bad answer

loss = -log_sigmoid(r_chosen - r_rejected)
```

This is the Bradley-Terry model — the same math used in chess Elo ratings. After training:

```
"Q: What is 2+2? A: 4."                    → score: -0.7
"Q: What is 2+2? A: Well, it could be..."  → score: -8.2
```

The reward model learned: concise, correct answers score higher.

## PPO (Proximal Policy Optimization)

The RL loop that optimizes the language model:

```python
for step in range(n_steps):
    # 1. Generate response
    response = policy_model.generate(prompt)

    # 2. Score it
    reward = reward_model(prompt + response)

    # 3. Compute how much policy changed
    current_log_probs = policy_model.log_prob(response)
    old_log_probs = old_policy.log_prob(response)
    ratio = exp(current - old)

    # 4. Clipped objective (don't change too much per step)
    clipped = clamp(ratio, 1-eps, 1+eps)
    loss = -min(ratio * reward, clipped * reward)

    # 5. KL penalty (don't forget how to speak)
    kl = current_log_probs - ref_log_probs
    loss += kl_coeff * kl
```

Two safety mechanisms:

**Clipping** — limits how much the policy changes per step. Without it, the model could jump to degenerate behavior in one update.

**KL penalty** — penalizes divergence from the original model. Without it, the model would "reward hack" — find degenerate text that scores high on the reward model but is nonsensical. The KL penalty keeps it close to real language.

## Results

On GPT-2 124M, the pipeline works but results are modest — the model is too small for real Q&A:

```
Reward model:  100% accuracy on preference pairs (8 pairs, 20 epochs)
PPO:           50 steps, reward improved on trained prompts

Before RLHF:  "The capital of France. Q: What is the capi..."  [reward: 3.7]
After RLHF:   "The capital of France is Paris."                 [reward: 9.0]
```

In production (InstructGPT paper, OpenAI 2022):
- GPT-3 175B as base
- 40 human labelers writing ideal answers (SFT)
- 33K comparison pairs for reward model
- PPO with KL penalty against SFT model

Result: InstructGPT **1.3B** was preferred over base GPT-3 **175B** by human raters. RLHF is so effective that a small aligned model beats a large unaligned one.

## Why This Matters

Every major LLM assistant (ChatGPT, Claude, Gemini) uses some form of RLHF or its successors (DPO, RLAIF). The base model learns language from internet text. RLHF teaches it to be helpful, honest, and harmless — properties that can't be captured by next-token prediction alone.

**Code:** [github.com/aserputov/rlhf-from-scratch](https://github.com/aserputov/rlhf-from-scratch)
