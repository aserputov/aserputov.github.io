---
layout: post
title: "Word2Vec: How Words Become Vectors"
date: 2026-02-08
description: Building skip-gram Word2Vec from scratch — embedding lookup tables, backpropagation, and cosine similarity.
tags: from-scratch
categories: from-scratch
---

## The Question

How do you turn a word into something a neural network can work with? You can't feed the string "king" into a matrix multiplication. You need numbers.

## One-Hot Encoding (The Naive Way)

The simplest approach: give each word a unique index, make a vector of zeros, put a 1 at that index.

```
"king"  → [0, 0, 0, 1, 0, 0, ...]  (10,000-dimensional)
"queen" → [0, 0, 0, 0, 1, 0, ...]
```

Problems:

- Every word is equally distant from every other word
- "king" is as similar to "queen" as it is to "banana"
- Vectors are huge and sparse

## Word2Vec: Learning Dense Embeddings

The idea (Mikolov et al., 2013): instead of hand-crafting word representations, **learn** them from context. Words that appear in similar contexts should have similar vectors.

**Skip-gram** predicts context words from a center word:

```
Input: "the [cat] sat on"
Training pairs: (cat, the), (cat, sat), (cat, on)
```

The network is simple — just two matrices:

```python
# Embedding: vocab_size × embed_dim
# Output:    embed_dim × vocab_size

hidden = W_embed[word_index]        # lookup, not matmul
scores = hidden @ W_output.T        # predict context
loss = cross_entropy(scores, target)
```

The first matrix **is** the embedding. After training, `W_embed[42]` is a dense vector for word 42 that encodes its meaning.

## What the Embeddings Learn

After training on enough text, the vectors capture relationships:

```python
similarity("king", "queen")  = 0.73
similarity("king", "banana") = 0.12

# The classic analogy
king - man + woman ≈ queen
```

This works because the network learns that "king" and "queen" appear in similar contexts ("the **_ decreed", "the _** ruled"), so their vectors end up nearby.

## Implementation Details

The from-scratch version includes:

- Embedding lookup (not one-hot × matrix, just index the row directly)
- Negative sampling for efficient training
- SGD with manual gradient computation
- Cosine similarity for evaluation

No frameworks — raw numpy, manual backprop through every layer.

## What I Took Away

The embedding lookup table is the same operation that starts every transformer. GPT-2's `wte` matrix is exactly this — 50,257 words, each mapped to a 768-dimensional vector. Understanding how these vectors are learned made everything downstream click.

**Code:** [github.com/aserputov/word2vec-from-scratch](https://github.com/aserputov/word2vec-from-scratch)
