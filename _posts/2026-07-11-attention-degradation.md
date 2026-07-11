---
layout: post
title: "Your LLM Doesn't Forget Instructions — It Just Stops Caring"
date: 2026-07-11
description: Measuring system prompt adherence degradation on Qwen 72B across long conversations. Spending $35 on GPUs to discover that attention is not reasoning.
tags: research
categories: research
---

I was debugging a long conversation with an LLM. The system prompt was clear: follow these formatting rules for every response. For the first few messages, it worked perfectly. By message 30, the model was ignoring every rule I'd set.

My first thought: it forgot.

So I tested it. I quoted the first few words of the system prompt and asked the model to continue. It completed the sentence word-for-word. It remembered everything. The system prompt was right there, sitting in context, fully intact.

The model wasn't forgetting. It was failing to **prioritize**.

That observation sent me down a rabbit hole that ended with renting 4x A100 80GB GPUs, running experiments on a 72-billion parameter model, attempting to build an architectural fix, and learning more about how transformers actually work than any textbook could teach.

Here's what I found.

## The Experiment

I wanted to measure this precisely, not just notice it anecdotally. So I built a simple benchmark.

**Setup:** Give the model a system prompt with strict, objectively measurable rules:
- Start every response with `[SUPPORT]`
- Never use the words "sorry", "apologize", or "apologies"
- End every response with `STATUS: OPEN` or `STATUS: RESOLVED`
- Keep responses under 40 words

Then simulate a multi-turn support conversation. After each batch of turns, ask a new question and check: does the response still follow all four rules?

I ran this on **Qwen 2.5 72B Instruct** — one of the strongest open-source models available — across 4x A100 80GB GPUs.

### The Results

| Conversation length | Rule adherence |
|-------------------|---------------|
| 0 turns (96 tokens) | **98%** |
| 5 turns (226 tokens) | 72% |
| 10 turns (358 tokens) | 60% |
| 20 turns (619 tokens) | 55% |
| 30 turns (878 tokens) | 48% |
| 50 turns (1,352 tokens) | **50%** |

From near-perfect to a coin flip. On a 72B model. In 1,352 tokens — that's roughly two pages of text.

I ran a second test with simpler rules (just a routing prefix and department tag). Same pattern:

| Conversation length | Rule adherence |
|-------------------|---------------|
| 0 turns | **100%** |
| 20 turns | 90% |
| 30 turns | 70% |
| 50 turns | **70%** |

More complex rules degrade faster. But all rules degrade.

## The Paradox That Changed How I Think About Transformers

Here's what bothered me most: **more context should help, not hurt.**

Think about it. After 50 turns of a billing support conversation, the model has overwhelming evidence about what format the response should be in. Every turn reinforces the pattern. If you showed a human 50 examples of the expected format, they'd be *more* confident about following it, not less.

But the model gets worse. Why?

Because **attention is not reasoning.**

The model doesn't think: "I've seen 50 turns confirming I should use the [SUPPORT] prefix, so now I'm very confident." It computes a weighted sum over all tokens in context, where the weights come from dot-product similarity between tokens. That's it. No rule tracking. No checklist. No goal.

The system prompt at position 0 is just another group of tokens competing for attention with everything else in context. And as context grows, it loses — because softmax distributes attention over more and more tokens, and recent tokens tend to win.

The model sees the last question louder than the first instruction.

## The Smoking Gun: RLHF Rules Don't Degrade

One finding made the picture crystal clear.

Among the four rules I tested, one was: "never say sorry." This is a behavior that models learn through RLHF (Reinforcement Learning from Human Feedback) — it gets baked into the model's weights during training.

Result: **100% adherence at every context length.** Zero degradation. 50 turns, 1,352 tokens — perfect compliance.

Same model. Same conversation. Same context. Format rules from the system prompt collapsed. The RLHF rule held perfectly.

This reveals two fundamentally different kinds of memory in an LLM:

**Weight memory** — behaviors encoded in parameters through RLHF. Permanent. Independent of context length. The model "knows" this the way you know not to touch a hot stove — it's built in.

**Context memory** — instructions provided via system prompt. Temporary. Dependent on attention. Degrades as the context grows. The model "follows" this the way you follow a sticky note on your monitor — it works until something covers it up.

System prompt instructions aren't rules. They're suggestions that get quieter over time.

## Why This Matters Beyond Formatting Rules

This isn't just about losing a `[SUPPORT]` prefix. The same mechanism affects everything a system prompt tries to do:

- **Safety instructions** that say "don't discuss X" become less effective in long conversations
- **Persona definitions** ("you are a medical advisor") drift toward generic assistant behavior
- **Output constraints** (JSON format, specific schemas) break down at scale
- **Tool-use instructions** become unreliable after many turns

Every production LLM system that relies on system prompts to control behavior has this problem. Most teams handle it with prompt engineering hacks — repeating instructions, adding reminders, truncating context. These are bandaids on an architectural limitation.

## My Attempt at a Fix (And Why It Failed)

Understanding the problem, I tried to build a solution.

**The idea:** Extract the system prompt into a persistent "goal vector" — a compressed representation of the instructions that lives outside the attention mechanism. Then inject this vector directly into every N-th transformer layer through gated addition, giving the model a constant signal about what the task is.

```
GoalExtractor:  attention_pool(system_prompt) → goal vector
GoalInjection:  hidden_state + gate * projected_goal  [at every 8th layer]
```

The goal modules are lightweight — about 60M parameters on top of a frozen 72B base model (0.08% overhead). The base model doesn't change; the goal vector acts as a persistent reminder.

### On a 3B model: it worked perfectly

| Context | Baseline | With Goal Vector |
|---------|----------|-----------------|
| 0 turns | 100% | 100% |
| 5 turns | 15% | **100%** |
| 10 turns | 5% | **100%** |
| 20 turns | 0% | **100%** |
| 50 turns | 0% | **100%** |

From 0% to 100% at every context length. And it generalized — trained on "start with NO," it worked on unseen instructions like YES, HELLO, WAIT. The model learned a general mechanism for amplifying instructions, not a specific word.

### On a 72B model: it didn't scale

The same architecture at 72B provided marginal improvement at long contexts (+2-5%) and actively hurt at short contexts (-48%). The goal vector was too small relative to the model — 60M parameters trying to steer 72 billion.

This is a negative result, but an informative one. It tells us:

1. The mechanism works in principle (3B proves the concept)
2. Lightweight patches can't override production-scale models
3. The fix needs to scale proportionally (~1% of base params, or ~720M trainable parameters for 72B)

## What I Learned

### 1. The context window is not the understanding window

Everyone is racing to build models with 128K, 1M, 10M token context windows. But my experiments show degradation starting at **300 tokens**. The context window is how much the model can *see*. How much it can effectively *use* is a completely different number — and it's much smaller.

### 2. Recall is not reasoning

The model can quote the system prompt back word-for-word at turn 50. But it can't use that information to make decisions. It's like someone who memorized traffic laws but drives on autopilot — the knowledge exists, the application doesn't. These are different cognitive processes, and transformers only have the first one.

### 3. RLHF is the only reliable enforcement mechanism

If you want a model to **guarantee** a behavior, it needs to be in the weights, not the prompt. System prompts are wishes. RLHF is law. This has practical implications: prompt engineering has a ceiling, and that ceiling is context length.

### 4. We found the edge of the architecture

This isn't a bug or a training issue. It's an architectural limitation. Softmax attention distributes a fixed budget across all tokens. No amount of data, parameters, or clever prompting changes the fact that 1.0 divided by more tokens means less attention per token.

## What's Next

The goal vector approach works at small scale and fails at large scale. The obvious next step: scale it proportionally and fix the training methodology. Specifically:

- **Train on full response quality**, not just the first token
- **Train on long contexts** that match the evaluation distribution
- **Scale the goal modules** to ~1% of base model parameters
- **Compare against trivial baselines** like repeating the system prompt at the end of context

The benchmark itself — measuring system prompt adherence degradation across context lengths — is something I haven't seen done systematically at this scale. If it's useful to you, the code is [open source](https://github.com/aserputov/attention-degradation).

## The Uncomfortable Implication

Here's what this research suggests, stated plainly:

Transformers don't reason. They predict. They're extraordinarily good at predicting the next token based on local patterns, and this is powerful enough to produce behavior that looks like reasoning. But when you test whether the model can maintain a goal across a long context — not recall it, but actively prioritize it — the illusion breaks.

A 72-billion parameter model, one of the most capable open-source models ever built, can't reliably follow four simple formatting rules over two pages of conversation. Not because it forgot them. Because it has no mechanism to care about them.

That's not a model problem. That's an architecture problem.

And I think it's worth working on.
