---
layout: page
title: LLM Inference Benchmark
description: Benchmarking framework for inference throughput, latency, and memory across quantization levels on Apple Metal GPU.
img:
importance: 1
category: ai-systems
github: https://github.com/aserputov/llm-inference-benchmark
---

A benchmarking framework measuring inference throughput (tokens/sec), per-token latency, and memory footprint across model sizes.

### Key Features
- **INT4/INT8 quantization** benchmarks on Apple Metal GPU
- Thread-scaling analysis for decode vs prefill operations
- Sub-linear OPS parallelization analysis
- KV cache utilization profiling across context lengths
- Memory footprint tracking across quantization levels

### Tech Stack
Python, llama.cpp, Apple Metal, GGUF format
