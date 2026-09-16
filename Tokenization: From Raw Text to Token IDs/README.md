# Tokenization: From Raw Text to Token IDs

## Overview

Before any text can be fed into an LLM, it has to be converted into numbers. This covers the first part of that pipeline: splitting text into tokens, assigning each token a unique ID, building a working encode/decode tokenizer, and handling words the model has never seen before using special tokens.

The example text used throughout is Edith Wharton's short story *"The Verdict"* (20,479 characters, loaded from `the-verdict.txt`).

---

## Step 1: Creating Tokens

The goal: split raw text into a list of individual words and special characters (tokens) that can later be converted into embeddings.

### Starting simple: splitting on whitespace

```python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
```
This just splits on whitespace, keeping punctuation stuck to words (`'Hello,'`, `'world.'`, etc.) — not useful on its own.

### Splitting on whitespace *and* punctuation

```python
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```
Now words and punctuation are separated into their own list entries, and empty/whitespace-only entries are filtered out.

**Note on whitespace:** whether to keep whitespace as its own token depends on the application. Removing it saves memory and compute, but for text where spacing is meaningful (e.g., Python code), you'd want to preserve it. This walkthrough removes whitespace for simplicity.

### Handling more punctuation types

Extending the regex to also catch question marks, quotes, parentheses, and double-dashes:

```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

### Applying it to the full short story

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
```
Result: **4,690 tokens** total from the story.

---

## Step 2: Creating Token IDs

Once text is split into tokens, each unique token needs a unique integer ID.

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)  # 1,130
vocab = {token: integer for integer, token in enumerate(all_words)}
```

The vocabulary maps each unique token (sorted alphabetically) to an integer — e.g., `'!'` → 0, `'"'` → 1, `'A'` → 11, `'HAD'` → 44, and so on.

### Building a tokenizer class (`SimpleTokenizerV1`)

A complete tokenizer needs both directions:
- **`encode`** — text → token IDs (uses the string-to-int vocab)
- **`decode`** — token IDs → text (uses an inverse int-to-string vocab)

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # remove space before punctuation
        return text
```

Tested on a snippet from the story, this successfully encodes to IDs and decodes back to (nearly) the original text.

### The problem: unknown words

Running the tokenizer on text **not** in the training set fails:

```python
text = "Hello, do you like tea?"
tokenizer.encode(text)
# KeyError: 'Hello'
```

"Hello" never appeared in the short story, so it's not in the vocabulary. This illustrates why LLMs need **large and diverse training sets** — vocabulary coverage depends entirely on what the model has seen.

---

## Adding Special Context Tokens

To handle unknown words and mark boundaries between documents, two special tokens are added to the vocabulary:

| Token | Purpose |
|---|---|
| `<\|unk\|>` | Stands in for any word not found in the vocabulary |
| `<\|endoftext\|>` | Marks the boundary between unrelated texts (e.g., between two separate books/documents) |

```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
```
New vocabulary size: **1,132** (up from 1,130).

### `SimpleTokenizerV2`

Same as V1, but any token not found in the vocabulary is replaced with `<|unk|>` before encoding:

```python
preprocessed = [
    item if item in self.str_to_int else "<|unk|>"
    for item in preprocessed
]
```

Example: joining two texts with `<|endoftext|>` between them, then encoding/decoding, correctly reveals that "Hello" and "palace" are out-of-vocabulary — both come back as `<|unk|>`.

### Other special tokens (used by some LLMs, though not GPT)

| Token | Meaning |
|---|---|
| `[BOS]` | Beginning of sequence — marks where a piece of content starts |
| `[EOS]` | End of sequence — marks where a text ends (similar role to `<\|endoftext\|>`) |
| `[PAD]` | Padding — extends shorter texts in a batch to match the length of the longest one |

**Note:** GPT models don't use any of `[BOS]`, `[EOS]`, or `[PAD]`, and don't use `<|unk|>` either — they only use `<|endoftext|>`, and handle unknown words differently, using **byte pair encoding** (covered in the next set of notes).

---

## Cheat Sheet

- **Tokenization pipeline (so far):** raw text → split into tokens (regex) → assign token IDs via vocabulary → handle unknowns.
- **`SimpleTokenizerV1`**: basic encode/decode using a fixed vocabulary; fails (`KeyError`) on unseen words.
- **`SimpleTokenizerV2`**: adds `<|unk|>` (unknown word placeholder) and `<|endoftext|>` (document boundary marker) to handle those cases gracefully.
- **Vocabulary size** grows from 1,130 unique tokens to 1,132 once `<|endoftext|>` and `<|unk|>` are added.
- **Special tokens** `[BOS]`, `[EOS]`, `[PAD]` exist in some tokenizer designs but **GPT does not use them** — GPT only uses `<|endoftext|>`.
- **Key limitation of this approach:** a fixed word-level vocabulary can't represent words it's never seen — this is exactly what byte pair encoding solves.