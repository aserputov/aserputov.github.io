---
layout: about
title: about
permalink: /
subtitle: Software Engineer | AI Infrastructure | Inference Optimization

profile:
  align: right
  image: prof_pic.jpg
  image_circular: false
  more_info: >
    <p>Toronto, Canada</p>

selected_papers: false
social: true

announcements:
  enabled: false
  scrollable: true
  limit: 5

latest_posts:
  enabled: true
  scrollable: true
  limit: 3
---

Software Engineer building **AI infrastructure**. I implement inference optimizations from scratch — KV-cache, PagedAttention, Flash Attention (custom Triton GPU kernels), continuous batching — to understand how LLM serving systems work at the lowest level.

I built a [series of 5 from-scratch implementations](https://github.com/aserputov) tracing the evolution from word embeddings to production inference: **Word2Vec → RNN/LSTM → Transformer → GPT-2 → Inference Engine**. Each project loads real weights and produces real results — no toy examples.

At work, I architect **autonomous LLM agent systems** with Model Context Protocol (MCP), design distributed microservice platforms, and build ML-driven infrastructure for high-throughput workloads.

I write about what I learn — deep dives into inference optimization, GPU programming, and the engineering behind LLM serving.
