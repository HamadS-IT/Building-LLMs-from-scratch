# Stages of Building a Large Language Model

## Overview

Building an LLM from scratch breaks down into **three stages**: building the core architecture, pretraining it into a foundational model, and finetuning it for specific applications. Each stage builds on the one before it, and skipping the early stages to jump straight to deployment tools leaves major gaps in understanding how things actually work.

---

## Stage 1: Building the LLM

The goal of this stage is to construct the core building blocks needed before any training happens. It covers three main areas: data preparation, the attention mechanism, and the overall LLM architecture.

### Data Preparation and Sampling

- **Tokenization** — breaking sentences into individual tokens (units of a sentence), following a specific, defined process.
- **Vector embedding** — after tokenization, each word is transformed into a high-dimensional vector so that semantic meaning is captured. Words with related meanings end up closer together in this vector space (e.g., *apple*, *banana*, *orange* cluster together; *king*, *man*, *woman* cluster together; *football*, *golf*, *tennis* cluster together).
- **Positional encoding** — the order in which words appear in a sentence matters, so this ordering information also needs to be fed into the model.
- **Data batching** — once tokenized and embedded, data needs to be organized into batches for efficient training. This involves:
  - Framing training as a **next-word prediction task** — given a sequence of words, predict what comes next.
  - Defining **context length** — how many prior words/tokens are used to predict the next one.
  - Implementing a batching sequence so data can be fed into the model efficiently.

### Attention Mechanism

- Understanding the components of the Transformer's attention system in detail: multi-head attention, masked multi-head attention, positional encoding, input embedding, output embedding.
- Implementing the attention mechanism from scratch in Python — including key, query, value vectors and how the attention score is computed.

### LLM Architecture

- Assembling the pieces above into a full architecture: stacking layers, placing attention heads correctly, and building the overall model structure.

**Outcome of Stage 1:** a fully assembled (but untrained) LLM architecture, ready for training.

---

## Stage 2: Pretraining

Once the data pipeline and architecture are ready, Stage 2 is about writing the training code and actually training the model on the underlying dataset — producing a **foundational model** trained on unlabeled data.

Key components covered in this stage:

| Component | Purpose |
|---|---|
| **Training loop** | Break the training data into epochs, compute the gradient of the loss in each epoch, and update model parameters accordingly |
| **Sample text generation** | Periodically generate sample text during training for visual inspection of progress |
| **Model evaluation** | Track training and validation losses to assess how well the model is learning |
| **Weight saving/loading** | Implement functions to save and load model weights, so training doesn't need to restart from scratch each time — saving significant compute and cost |
| **Loading pretrained weights** | Load publicly available pretrained weights (e.g., from OpenAI) into the model, rather than always training entirely from scratch |

**Outcome of Stage 2:** a pretrained foundational model, trained on unlabeled data using next-word prediction (auto-regressive training — the sentence structure itself generates the labels).

---

## Stage 3: Finetuning

The goal of this final stage is to adapt the pretrained foundational model for specific, labeled tasks. Two applications are built in this stage:

### 1. Classifier (Spam Detection)
- Example task: classify emails as spam or not spam.
- The foundational model alone isn't sufficient — it needs additional **labeled data** (examples explicitly marked as spam/not spam) to be finetuned into a reliable classifier.

### 2. Personal Assistant / Chatbot
- A chatbot that responds to instructions and inputs with appropriate outputs.
- Built by finetuning the LLM on instruction-input-output style labeled data.

**Why finetuning matters:** production-level LLM applications (airlines, restaurants, banks, educational companies, etc.) virtually always finetune a pretrained model rather than deploying the raw foundational model directly. Finetuned models consistently outperform pretrained-only models on the specific tasks they're tuned for.

---

## Pretraining vs. Finetuning (Quick Comparison)

| Aspect | Pretraining | Finetuning |
|---|---|---|
| **Data** | Unlabeled, massive (billions of words) | Labeled, smaller, task-specific |
| **Labels** | Derived automatically from sentence structure (next-word prediction) | Explicitly provided (e.g., spam/not spam, good/bad answer) |
| **Cost** | Very high — GPT-3 pretraining cost ~$4.6 million | Much lower |
| **Output** | Foundational/general-purpose model | Task-specific, production-ready model |

---

## Recap: Key Concepts Covered So Far

- **LLMs have transformed NLP** — older NLP approaches needed a separate model per task; LLMs are generic enough that training on next-word prediction alone produces **emergent properties** — abilities like multiple-choice question answering, summarization, emotion classification, and translation that were never explicitly trained for.
- **Two-step training process:** pretraining (on unlabeled data, producing a foundational model) followed by finetuning (on labeled, task-specific data) for production use.
- **Transformer architecture** is the secret sauce behind LLMs, and the **attention mechanism** is its core innovation — it gives the model selective access to the entire input sequence (not just the current sentence) when generating each output word, allowing it to weigh which prior words matter most for predicting the next one.
- **Transformer ≠ GPT:**
  - The original Transformer (2017) had both an **encoder and a decoder**.
  - GPT (Generative Pre-trained Transformer), introduced in **2018**, uses **only the decoder** — no encoder. This remains true even for GPT-4.
- **GPT model timeline:**

| Year | Model | Notes |
|---|---|---|
| 2018 | GPT | First generative pretrained Transformer, decoder-only |
| 2019 | GPT-2 | — |
| 2020 | GPT-3 | 175 billion parameters — unprecedented scale at the time |
| — | GPT-4 | Current generation |

---

## Cheat Sheet

- **Stage 1 — Build:** data preparation & sampling (tokenization, vector embedding, positional encoding, batching) + attention mechanism (key/query/value, attention scores) + overall LLM architecture.
- **Stage 2 — Pretrain:** train the assembled architecture on unlabeled data (next-word prediction) → produces a foundational model. Includes training loop, evaluation, and weight saving/loading.
- **Stage 3 — Finetune:** adapt the foundational model using labeled, task-specific data → produces production-ready applications (e.g., a spam classifier, a chatbot/personal assistant).
- **Pretraining** = unlabeled, expensive, general-purpose. **Finetuning** = labeled, cheaper, task-specific — and required for real production deployments.
- **Attention mechanism** = gives the model selective access to the full input context (not just the current sentence) when predicting each next word.
- **Transformer (2017)** = encoder + decoder. **GPT (2018+)** = decoder only — including GPT-4.
- **Emergent properties**: training only on next-word prediction still produces abilities like classification, translation, and summarization — never explicitly trained for.