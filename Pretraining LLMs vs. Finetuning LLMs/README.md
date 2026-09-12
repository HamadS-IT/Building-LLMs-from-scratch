# Pretraining LLMs vs. Finetuning LLMs

Large language models are built in stages, and the two most important stages are **pretraining** and **finetuning**. They serve very different purposes, use different data, and require very different amounts of compute.

---

## Pretraining

Pretraining is the initial, large-scale training phase where a model learns general language patterns from scratch (random weights).

- **Goal:** Learn grammar, facts, reasoning patterns, and general world knowledge by predicting the next word (or a masked word) in massive amounts of text.
- **Data:** Huge, broad, mostly unlabeled text corpora — web pages, books, code, articles, forums, etc. Often hundreds of billions to trillions of tokens.
- **Objective:** Typically self-supervised — e.g., next-token prediction (GPT-style) or masked-token prediction (BERT-style). No human-labeled examples are needed.
- **Compute cost:** Extremely high — can take weeks/months on thousands of GPUs/TPUs and cost millions of dollars.
- **Output:** A "base model" — capable of generating fluent text but not necessarily good at following instructions, being safe, or being helpful in a conversational sense.

---

## Finetuning

Finetuning takes an already-pretrained base model and adapts it for a narrower purpose using a smaller, more targeted dataset.

- **Goal:** Specialize the model's behavior — follow instructions, adopt a particular tone, perform a specific task, or align with human preferences.
- **Data:** Much smaller, curated, often labeled or human-generated datasets (thousands to a few million examples), specific to the target task or behavior.
- **Objective:** Supervised learning on task-specific examples, and often further refined with techniques like:
  - **Supervised Finetuning (SFT)** — training on high-quality input/output pairs (e.g., instruction–response pairs).
  - **RLHF (Reinforcement Learning from Human Feedback)** — using human preference rankings to further align model outputs.
- **Compute cost:** Much lower than pretraining — can often be done on a handful of GPUs in hours or days.
- **Output:** A model tuned for a specific use case — e.g., a chat assistant, a customer support bot, a coding assistant, or a domain-specific expert (legal, medical, etc.).

---

## Side-by-Side Comparison

| Aspect | Pretraining | Finetuning |
|---|---|---|
| **Starting point** | Random weights | Pretrained base model |
| **Data size** | Massive (billions–trillions of tokens) | Small to moderate (thousands–millions of examples) |
| **Data type** | Broad, mostly unlabeled text | Curated, often labeled/task-specific |
| **Objective** | Next-token / masked-token prediction | Task-specific supervised learning (often + human feedback) |
| **Compute cost** | Very high (weeks/months, huge clusters) | Relatively low (hours/days, fewer GPUs) |
| **Purpose** | Learn general language ability and world knowledge | Adapt general ability to a specific task, domain, or behavior |
| **Result** | Base model (general-purpose, less instruction-following) | Specialized model (task-aligned, instruction-following, safer) |
| **Who typically does it** | Large labs with major compute resources (OpenAI, Google, Meta, etc.) | Companies/developers adapting existing base models for their use case |

---

## How They Fit Together

1. **Pretraining** builds the foundation — a model that understands language broadly but isn't tailored to any particular use.
2. **Finetuning** takes that foundation and shapes it into something useful and aligned — like turning a general-purpose base model into ChatGPT-style assistant that follows instructions and stays on-topic.

Think of pretraining as teaching a model to "know language," and finetuning as teaching it "how to behave" in a specific context.

---

## Common Finetuning Variants

- **Full finetuning** — updating all of the model's parameters (expensive, but most flexible).
- **Parameter-efficient finetuning (PEFT)** — updating only a small subset of parameters or adding lightweight adapter layers (e.g., **LoRA**), which is much cheaper and faster.
- **Instruction finetuning** — training on (instruction, response) pairs so the model learns to follow user commands.
- **RLHF** — using human preference data and reinforcement learning to further refine tone, helpfulness, and safety.

---

## Cheat Sheet

- **Pretraining** = learning language from scratch on massive, general data → produces a *base model*. Expensive, slow, done by large labs.
- **Finetuning** = adapting a base model to a specific task/behavior using smaller, curated data → produces a *specialized model*. Cheap, fast, done by many developers.
- **Data:** pretraining = huge & unlabeled; finetuning = small & labeled/curated.
- **Objective:** pretraining = next/masked-token prediction; finetuning = supervised learning (+ often RLHF).
- **Analogy:** pretraining teaches the model "how language works"; finetuning teaches it "how to behave for this job."
- **Efficient finetuning options:** LoRA and other PEFT methods reduce cost by updating only a small part of the model.