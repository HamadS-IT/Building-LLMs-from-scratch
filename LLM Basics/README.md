# Large Language Models: The Basics

## What Is a Large Language Model?

At its core, an LLM is a **neural network designed to understand, generate, and respond to human-like text.**

Breaking that down:

- **Neural network** — a system loosely modeled on brain circuitry: input data feeds into layers of interconnected "neurons," stacked together, producing an output. Neural networks (also called deep neural networks) are used across many domains — image recognition, self-driving cars, medical diagnosis, text generation, and more.
- **Understand, generate, and respond to human-like text** — LLMs are neural networks specialized for text-based tasks. When you interact with something like ChatGPT — asking it to plan your day, answer a question, or hold a conversation — you're seeing this in action. The responses don't just process language; they mimic human conversational style.

**Bottom line:** an LLM is a deep neural network trained on massive amounts of data to understand, generate, and respond to human-like text — often convincingly enough to sound human itself.

---

## Why "Large"?

The "large" refers to **parameter count** — the number of adjustable values inside the model. Before LLMs, language models were comparatively small. LLMs changed that dramatically.

### GPT Parameter Growth

| Model | Parameters |
|---|---|
| GPT-3 Small | 125 million |
| GPT-3 Medium | 350 million |
| GPT-3 Large | 760 million |
| GPT-3 13B | 13 billion |
| GPT-3 175B | 175 billion |
| GPT-4 | even more |

Other specs also scaled up alongside parameters:

| Model | Decoders | Token size |
|---|---|---|
| GPT-1 | 12 | 512 |
| GPT-2 | 48 | 1024 |
| GPT-3 | 96 | 2048 |

Growth wasn't linear — GPT-1 to GPT-2 was roughly a 10x jump (100M → ~1.5B parameters), and GPT-2 to GPT-3 was roughly 100x (1.5B → 175B).

A broader historical chart (covering AI models from 1950–2022, log scale) shows parameter counts crawling from thousands in the 1950s–60s to ~100,000 by the 1980s–2000s, then exploding to hundreds of millions, billions, and now approaching a trillion by 2020+. LLMs occupy the extreme high end of this curve.

**"Language"** is the second half of the name because these models are built specifically for language-related tasks — translation, question answering, sentiment analysis, and more — rather than other data types like images or video.

---

## LLMs vs. Earlier NLP Models

Natural language processing (NLP) existed long before LLMs, but the two differ in key ways:

| | Earlier NLP models | Modern LLMs |
|---|---|---|
| **Scope** | Built for one specific task (e.g., translation only, or sentiment analysis only) | One architecture (e.g., GPT) handles many tasks — translation, completion, summarization, etc. |
| **Flexibility** | Struggled with tasks like drafting a custom email | Handles such tasks trivially (e.g., ChatGPT drafting a full email with appropriate tone/emojis in seconds) |
| **Applications** | Narrow | Broad and rapidly expanding |

This generality is a major reason LLMs have become so widely useful compared to their predecessors.

---

## The Secret Sauce: Transformer Architecture

What makes LLMs so much more capable than earlier models is the **Transformer architecture**, introduced in the 2017 paper *"Attention Is All You Need"* by eight authors at Google Brain.

- The paper is only ~15 pages but extremely dense — each page could arguably fill several lecture videos.
- It has accumulated over 100,000 citations within about five years of publication.
- Key components of the architecture (to be covered in depth later) include:
  - Input embedding
  - Positional encoding
  - Multi-head attention
  - Add & Norm layers
  - Feed-forward layers
  - Output embedding
  - Key, query, value vectors and the attention formula

This architecture is the foundational innovation behind modern LLMs' performance.

---

## Untangling the Terminology: AI, ML, DL, LLM, GenAI

These terms nest inside each other like umbrellas, from broadest to narrowest:

```
Artificial Intelligence (AI)
  └─ Machine Learning (ML)
       └─ Deep Learning (DL)
            └─ Large Language Models (LLM)
```

### Artificial Intelligence (AI)
The broadest category — any system that behaves with some form of intelligence, even if it doesn't "learn."
- *Example:* A rule-based airline chatbot (like "Elisa") that offers fixed menu options and gives scripted responses. It doesn't adapt to individual users — same input, same output, regardless of who's asking. This counts as AI but **not** ML.

### Machine Learning (ML)
Systems that **learn and adapt** based on data/interaction, rather than following fixed rules.
- Includes neural networks **and** non-neural-network methods like **decision trees**.
- *Example:* A decision tree predicting heart disease risk from patient data (age, cholesterol, ECG, etc.) — this is ML, but not DL, since no neural network is involved.

### Deep Learning (DL)
A subset of ML that specifically uses **neural networks**.
- *Examples:* A convolutional neural network classifying images (e.g., identifying a coffee cup), or a neural network trained to recognize handwritten digits.
- DL covers multiple data types — images, audio, text, etc.

### Large Language Models (LLM)
A subset of DL restricted specifically to **text** (not images, audio, or video).

### Generative AI
Best understood as **LLM + Deep Learning**, extended to other modalities. Generative AI creates new content — text, images, audio, video — using deep neural networks. LLMs are the text-only slice of this broader generative capability.

---

## Applications of LLMs

Applications are expanding constantly, but they broadly fall into five categories:

| Category | Description | Example |
|---|---|---|
| **1. Content creation** | Generating original text that didn't exist before | Writing a poem, story, or full book on request |
| **2. Chatbots / virtual assistants** | Conversational agents for customer service, bookings, support | Airline, hotel, and bank customer service bots |
| **3. Machine translation** | Translating text between languages | Translating a poem into French instantly |
| **4. Text generation (long-form)** | Producing news articles, media content, books | Automated article/content drafting |
| **5. Sentiment analysis** | Detecting emotional tone or intent in text | Hate speech detection on social media |

### Real-world example: education tools
A portal built for school teachers using LLMs includes tools like:
- Lesson plan generator (e.g., a gravity lesson aligned to a specific curriculum, generated in seconds)
- MCQ generator (e.g., producing leveled World War II questions with answers and explanations in ~5 seconds)
- Text summarizer, text rewriter, worksheet generator, YouTube content generator

These illustrate how quickly LLMs can produce professional-quality output that would have taken significant manual effort before.

**Caution:** knowing how to run existing LLM code/applications isn't the same as understanding LLMs. Real depth comes from understanding the underlying mechanics — Transformer internals, attention, key/query/value, positional encoding — rather than just using pre-built tools.

---

## Cheat Sheet

- **LLM = a neural network trained to understand, generate, and respond to human-like text.**
- **"Large"** = billions (even trillions) of parameters — far beyond pre-LLM models.
- **LLMs vs. old NLP models:** old models were narrow (one task each); LLMs are generalists (many tasks, one architecture) and handle complex applications (like custom email drafting) easily.
- **Secret sauce** = the **Transformer architecture**, from the 2017 "Attention Is All You Need" paper.
- **Nesting of terms:** AI ⊃ ML ⊃ DL ⊃ LLM. Generative AI ≈ LLM + DL, extended beyond text to images/audio/video.
  - AI = any intelligent-seeming behavior (even rule-based, non-learning systems)
  - ML = systems that learn from data (neural nets + methods like decision trees)
  - DL = ML specifically using neural networks
  - LLM = DL specifically for text
- **Five major LLM application areas:** content creation, chatbots, translation, long-form text generation, sentiment analysis.
- **Key takeaway:** real mastery comes from understanding the internals (Transformers, attention, key/query/value, positional encoding) — not just running existing tools.