# Byte Pair Encoding (BPE)

## Overview

A fixed, word-level vocabulary (as built with `SimpleTokenizerV1`/`V2`) has a hard limitation: it can't represent a word it has never seen — it either throws an error or falls back to a generic `<|unk|>` token, losing all information about what the word actually was. GPT models solve this differently, using **byte pair encoding (BPE)**, which breaks words down into **subword units** instead of relying on a whole-word vocabulary.

Because BPE operates on subword pieces (and ultimately down to individual characters/bytes if needed), it can represent *any* input text — including words invented on the spot — without ever needing an `<|unk|>` token.

---

## Why BPE Instead of a Fixed Vocabulary?

- A word-level tokenizer's vocabulary is only as good as its training data — if a word wasn't in the training text, it's out-of-vocabulary.
- BPE instead builds a vocabulary of common **subword fragments**. Rare or novel words get broken into smaller known pieces rather than being replaced with a placeholder.
- This is why GPT tokenizers don't need `<|unk|>`, `[BOS]`, `[EOS]`, or `[PAD]` tokens — the only special token GPT uses is `<|endoftext|>`.

---

## Using the `tiktoken` Library

OpenAI's `tiktoken` library provides a fast, production-grade BPE tokenizer — the same one used by GPT-2 and related models.

### Setup

```python
! pip3 install tiktoken
```

```python
import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))
```

### Loading the GPT-2 encoding

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

### Encoding text

```python
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
```

- `allowed_special={"<|endoftext|>"}` explicitly tells the tokenizer to treat `<|endoftext|>` as a recognized special token rather than trying to break it apart as regular text.
- Note the made-up word `"someunknownPlace"` — despite never having been "trained" on this exact word, BPE still encodes it successfully by breaking it into subword pieces.

### Decoding back to text

```python
strings = tokenizer.decode(integers)
print(strings)
# Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.
```

The original text is recovered exactly — proving the subword-based encode/decode round-trip works even for unusual or novel strings.

---

## Exercise: Encoding a Nonsense String

To further demonstrate that BPE can handle *any* input — not just real words — even a random nonsense string encodes and decodes cleanly:

```python
integers = tokenizer.encode("Akwirw ier")
print(integers)
# [33901, 86, 343, 86, 220, 959]

strings = tokenizer.decode(integers)
print(strings)
# Akwirw ier
```

`"Akwirw ier"` isn't a real word or phrase, but BPE still represents it correctly by decomposing it into smaller known subword/byte-level units, then reconstructing it perfectly on decode.

---

## Cheat Sheet

- **Byte Pair Encoding (BPE)** = the tokenization scheme GPT models actually use — breaks words into subword units instead of relying on a whole-word vocabulary.
- **Key benefit:** can represent *any* text, including unseen or made-up words, without needing an `<|unk|>` token.
- **GPT's only special token:** `<|endoftext|>` — no `[BOS]`, `[EOS]`, `[PAD]`, or `<|unk|>` needed.
- **Implementation:** OpenAI's `tiktoken` library — `tiktoken.get_encoding("gpt2")` loads the exact tokenizer GPT-2 uses.
- **`encode(text, allowed_special={...})`** — converts text to token IDs, with special tokens explicitly whitelisted.
- **`decode(ids)`** — converts token IDs back to the original text, exactly.
- **Proven by example:** both a real out-of-vocabulary phrase (`"someunknownPlace"`) and a nonsense string (`"Akwirw ier"`) encode/decode correctly — demonstrating BPE's robustness compared to a fixed word-level vocabulary.