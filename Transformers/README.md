# Introduction to Transformers

## Overview

Transformers are the architecture behind most modern large language models. This covers the basic intuition — what a Transformer is, how it works at a high level, how it relates to GPT and BERT, and how "Transformer" and "LLM" differ as terms (they are not interchangeable, despite common usage).

---

## What Is a Transformer?

A Transformer is a **deep neural network architecture** introduced in the 2017 paper *"Attention Is All You Need."*

- The paper has accumulated more than 100,000 citations within 6–7 years of publication.
- The GPT architecture (foundation of ChatGPT) is heavily based on — but not identical to — this original Transformer architecture.
- **Important historical note:** the original Transformer was designed for **machine translation** (e.g., English → German, English → French), not for open-ended text generation. Text completion (GPT's main strength) wasn't the original use case — that capability was discovered and built out later.

---

## Simplified Transformer Architecture (8 Steps)

Using an English-to-German translation example, the process can be broken into eight steps:

| Step | What happens |
|---|---|
| 1 | **Input text** — the sentence to be translated (in English) |
| 2 | **Tokenization** — the sentence is broken into tokens (for simplicity, think one word ≈ one token) and each token gets a unique ID |
| 3 | **Encoder** — token IDs are passed to the encoder |
| 4 | **Vector embedding** — the encoder converts tokens into vector embeddings that capture semantic meaning |
| 5 | **Partial output** — the decoder receives the partial translation generated so far (translation happens one word at a time) |
| 6 | **Decoder input** — the partial output is tokenized and fed into the decoder, along with the vector embeddings from the encoder |
| 7 | **Prediction** — the decoder predicts the next word in the translation |
| 8 | **Final output** — the complete translated sentence |

### Tokenization

- Breaks sentences into smaller units (tokens) and assigns each a unique numerical ID.
- In practice, one token is *not* always one full word (it can be a subword), but treating tokens as whole words is a fine simplification for building intuition.

### Vector Embedding

- Raw token IDs don't capture meaning or relationships between words — two unrelated words could get IDs right next to each other.
- **Vector embedding** solves this by mapping each token into a high-dimensional vector space (often hundreds of thousands of dimensions) such that semantically related words end up positioned closer together.
- Example: in a simplified 2D embedding space, *King*, *Man*, *Woman* cluster together; *Apple*, *Banana*, *Orange* cluster together (fruits); *Football*, *Golf*, *Tennis* cluster together (sports).
- This structure isn't random — it's learned through training.

### Encoder vs. Decoder

- **Encoder:** converts input text into vector embeddings.
- **Decoder:** takes the embeddings *plus* the partial output generated so far, and predicts the next word — one word at a time.

These two components — encoder and decoder — are the two main building blocks of the original Transformer.

> **Note:** The GPT architecture only uses a *decoder* — it has no encoder. This is covered further below.

---

## Self-Attention: The Core Mechanism

The paper is called *"Attention Is All You Need"* because of the **self-attention mechanism** — the key innovation that makes Transformers so effective.

**What it does:** allows the model to weigh the importance of different words/tokens *relative to each other*, regardless of how far apart they are in the text (long-range dependencies).

**Why it matters:** predicting the next word often requires context from much earlier in the text — not just the immediately preceding sentence. Self-attention lets the model look back across an entire passage and assign an "attention score" indicating which prior words matter most for predicting what comes next.

- Example: to correctly predict the next word in a fourth sentence about Harry Potter boarding a train, the model may need context from sentences 1–3 (mentions of "Harry," "platform," "train," etc.).
- This is fundamentally different from simply looking at the immediately previous word — attention can jump to any earlier relevant token.
- In the architecture diagram, this shows up as "multi-head attention" blocks.

---

## BERT vs. GPT

Both models descend from the Transformer architecture, but they work very differently.

| | **BERT** | **GPT** |
|---|---|---|
| **Full name** | Bidirectional Encoder Representations from Transformers | Generative Pre-trained Transformer |
| **Core task** | Predicts masked (hidden) words within a sentence | Predicts the next word, given preceding text |
| **Direction** | Bidirectional — looks at both left and right context | Left-to-right only — predicts based on prior words |
| **Architecture used** | Encoder only | Decoder only |
| **Strength** | Understanding nuance/context (e.g., distinguishing "bank" as a financial institution vs. a riverbank) | Generating fluent, coherent new text one word at a time |
| **Common use case** | Sentiment analysis | Text generation, chat, completion |

**Why BERT excels at sentiment analysis:** because it reads a sentence from both directions simultaneously, it captures relationships between words more fully — including cases where a masked/ambiguous word could appear anywhere in the sentence.

**Why GPT dominates today's conversation:** it generates new text fluently one word at a time and can also perform tasks like sentiment analysis, even though that's not its specialty.

---

## Transformers ≠ LLMs (Important Distinction)

These terms are often used interchangeably, but they are **not the same thing**.

### Not all Transformers are LLMs
Transformers are also used outside of language tasks — most notably in computer vision:
- **Vision Transformers (ViT)** are used for image classification, object/defect detection (e.g., detecting potholes), and medical image analysis (e.g., classifying tumors as malignant vs. benign).
- ViTs can achieve strong results with less compute than CNNs for pretraining, though they show a generally weaker inductive bias than convolutional approaches.

### Not all LLMs are Transformers
Before Transformers existed, other architectures were already doing sequence modeling and text completion — meaning they qualify as (earlier) language models:
- **Recurrent Neural Networks (RNNs)** — introduced 1980s; use feedback loops to retain memory across a sequence.
- **Long Short-Term Memory networks (LSTMs)** — introduced 1997; maintain two separate memory paths — one for long-term memory, one for short-term memory — to make predictions.
- Some convolutional architectures have also been used for language modeling.

**Bottom line:** "Transformer" and "LLM" describe different, overlapping-but-distinct concepts. Use them precisely.

---

## Cheat Sheet

- **Transformer** = a deep neural network architecture from the 2017 paper *"Attention Is All You Need,"* originally built for machine translation.
- **8-step simplified pipeline:** input text → tokenize → encode into vector embeddings → decoder combines embeddings + partial output → predicts next word → repeats until full output is generated.
- **Tokenization** = breaking text into smaller units (tokens) and assigning each an ID.
- **Vector embedding** = mapping tokens into high-dimensional vectors so that semantically related words are positioned closer together.
- **Encoder** = converts input into embeddings. **Decoder** = generates output text one word at a time using embeddings + prior output.
- **Self-attention** = lets the model weigh the importance of other words (near or far) when predicting the next word — enables long-range context.
- **BERT** = encoder-only, bidirectional, predicts masked words, strong at sentiment analysis.
- **GPT** = decoder-only, left-to-right, predicts the next word, strong at text generation.
- **Key distinction:** not all Transformers are LLMs (e.g., Vision Transformers for images); not all LLMs are Transformers (e.g., RNNs, LSTMs predate Transformers but still model language).