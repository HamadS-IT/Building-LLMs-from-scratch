# How GPT-3 Really Works

## Overview

GPT-3 is a **decoder-only Transformer**, scaled up massively compared to its predecessors. Its core mechanics are simple — tokenize, embed, predict the next token, repeat — but the *scale* at which it does this is what makes it capable of surprisingly broad, flexible behavior.

---

## The Basic Mechanics

### 1. Tokenization
- Input text is broken into tokens — subword chunks, not always full words.
- Each token is mapped to a numeric ID.
- GPT-3 uses a vocabulary of roughly **50,000 possible tokens**.

### 2. Embedding
- Each token ID is converted into a vector — a list of numbers in high-dimensional space.
- The largest GPT-3 model uses **12,288 dimensions** per token vector.
- **Positional information** is added alongside the token embedding, since Transformers don't inherently understand word order the way RNNs do — the model needs to be explicitly told "this token is 1st, this one is 5th," etc.

### 3. Stacked Decoder Blocks
- GPT-3's 175 billion parameters are organized into **96 stacked decoder layers**.
- Each layer performs two main operations:

| Component | Function |
|---|---|
| **Self-attention** | Every token looks at every other token that came *before* it and decides how much attention to pay to each when building its own representation. This captures context and relationships across long stretches of text. |
| **Feed-forward network** | A standard neural network layer that further transforms each token's representation. |

- Because GPT-3 is **decoder-only**, attention is **causal** (also called "masked") — a token can only attend to tokens that came before it, never tokens that come after.
- This causal restriction is what makes GPT-3 a left-to-right, next-word predictor — as opposed to something like BERT, which sees an entire sentence (both directions) at once.

### 4. Prediction
- After passing through all 96 layers, the model outputs a **probability distribution over the entire vocabulary** for what the next token should be.
- It samples (or picks the most likely) token, appends it to the sequence, and feeds the updated sequence back in as input.
- This repeats one token at a time until generation is complete.

---

## How GPT-3 Got Its Abilities

### Pretraining
- Trained on a huge, broad text corpus — hundreds of billions of tokens from web pages, books, Wikipedia, and more.
- Training objective: **predict the next token**, over and over, at massive scale.
- No labeled data and no explicit task instructions were used during this phase.

### Emergent Behavior
- GPT-3 was never explicitly taught to translate, summarize, or answer questions.
- Because these patterns naturally occur throughout its training data — and because the model is large enough to absorb them — these capabilities emerged as side effects of becoming very good at next-token prediction.

### In-Context Learning
- One of GPT-3's most notable traits: it can perform new tasks just from examples given directly in the prompt — **no weight updates required**.
- Show it a couple of examples of a pattern, and it continues the pattern ("few-shot" learning).
- This ability was one of the headline findings of the original GPT-3 paper.

---

## Why Scale Mattered

- GPT-3 wasn't a fundamentally new architecture compared to GPT-2 — it was mostly a **scale-up**:

| | GPT-2 | GPT-3 |
|---|---|---|
| Parameters | 1.5 billion | 175 billion |
| Layers | Fewer | 96 |
| Training data | Smaller | Much larger |

- This jump in scale pushed GPT-3 from a "decent text generator" into a model capable of broad, flexible, human-like task performance.
- This is the core reason parameter count became such a defining metric in the field — scale itself, not just architectural novelty, drove major capability jumps.

---

## Cheat Sheet

- **Architecture:** decoder-only Transformer, causal (masked) self-attention.
- **Pipeline:** tokenize → embed (+ positional info) → pass through 96 decoder layers (self-attention + feed-forward) → predict next token → repeat.
- **Vocabulary:** ~50,000 tokens. **Embedding size:** 12,288 dimensions (largest model).
- **Training objective:** next-token prediction on massive, broad text data — no labels, no task-specific instructions.
- **Emergent abilities:** translation, summarization, Q&A, etc. arose naturally from scale + next-token prediction, not explicit training.
- **In-context / few-shot learning:** GPT-3 can learn a new task from examples in the prompt alone, without updating its weights.
- **Key insight:** GPT-3's leap over GPT-2 was mostly about **scale** (117x more parameters), not a new architecture.