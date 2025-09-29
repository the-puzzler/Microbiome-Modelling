## Microbiome Model: Transformer Denoising with DNA & Text Language Models

This repository contains the model and training code for learning **complex, abstract interactions** among microbes in their communities—aimed at explaining **emergent behavior** from context and composition.

### Goal
Teach the model to internalize microbe–microbe and microbe–environment relationships so it can:
- **Predict which microbes are plausibly present** in a given context
- **Flag microbes that are unlikely** under that same context

### Embedding Backbones
- **DNA embeddings:** produced by a **DNA language model** (transformer-based) over sequences
- **Context embeddings:** produced by a **text language model** (transformer-based) over biome/environmental metadata

These are aligned in a shared space where **(context, microbe)** compatibility can be scored.

### Architecture (High Level)
1. **Encode context** (metadata → transformer text LM → dense vector)  
2. **Encode microbes** (DNA seq → transformer DNA LM → dense vector)  
3. **Compatibility head** (e.g., dot product or MLP) maps (context, microbe) to a probability of plausibility
4. **Main transformer**

### Training Objective: Denoising with Bernoulli Corruption
We train as a **denoising** task:
- Start with a set of microbes known/observed for a context.
- **Corrupt** the set by **adding “imposter” microbes** sampled Bernoulli-style.
- The model receives the **union** of true + corrupted microbes and must **label each microbe** as present vs. imposter.
- Optimize **binary cross-entropy (BCE)** over microbe logits:
  - **Positive label = 1** for true members
  - **Negative label = 0** for injected imposters

This teaches the model to **disentangle true ecological/functional compatibility** from random co-occurrence.

> For an extended walkthrough and ablations, see the blog post: **[https://the-puzzler.github.io/the-puzzler/post.html?p=posts%2Fmicro-modelling%2Fmicro-modelling.html]**.
