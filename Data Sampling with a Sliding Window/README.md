# Data Sampling with a Sliding Window

## Overview

Once text is tokenized (via byte pair encoding), it needs to be turned into the actual input/target pairs an LLM trains on. This covers the **sliding window** technique for generating next-word-prediction examples, and the PyTorch `Dataset`/`DataLoader` implementation that batches those examples efficiently for training.

---

## The Core Idea: Input/Target Pairs

An LLM is trained to predict the **next token** given some preceding context. To generate training examples for this, take a chunk of tokens as the input (`x`), and the *same chunk shifted one position to the right* as the target (`y`).

```python
enc_text = tokenizer.encode(raw_text)  # 5,145 tokens total
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")
# x: [290, 4920, 2241, 287]
# y:      [4920, 2241, 287, 257]
```

`context_size` (also called the context window or context length) determines how many prior tokens the model sees when predicting the next one.

### Stepping through it one token at a time

```python
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
```

```
[290] ----> 4920
[290, 4920] ----> 2241
[290, 4920, 2241] ----> 287
[290, 4920, 2241, 287] ----> 257
```

Decoded back into words, this makes the next-word-prediction task intuitive — a sentence being built up one word at a time:

```
 and ---->  established
 and established ---->  himself
 and established himself ---->  in
 and established himself in ---->  a
```

This shifted-by-one relationship between input and target **is** the training signal for a GPT-style model.

---

## Implementing a Data Loader

For real training, thousands of these (input, target) pairs need to be generated and served in batches. This is done with a custom PyTorch `Dataset` plus the standard `DataLoader`.

### `GPTDatasetV1`

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

**Key parameters:**

| Parameter | Meaning |
|---|---|
| `max_length` | Number of tokens in each input/target chunk (the context window) |
| `stride` | How far the sliding window moves forward each step |

- `stride == max_length` → chunks don't overlap at all.
- `stride < max_length` → chunks overlap, producing more (somewhat redundant) training examples from the same text.

### `create_dataloader_v1`

Wraps the dataset in a standard PyTorch `DataLoader`, which handles batching, shuffling, and parallel loading:

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                          stride=128, shuffle=True, drop_last=True,
                          num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
```

---

## Trying It Out

### Single example, stride = 1

```python
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
```

Each target is simply the input shifted one token to the right — consistent with the sliding window logic above.

```python
second_batch = next(data_iter)
print(second_batch)
# [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
```

With `stride=1`, consecutive batches overlap by all but one token.

### Full batch, no overlap (stride = max_length)

```python
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
```

```
Inputs:
 tensor([[   40,   367,  2885,  1464],
         [ 1807,  3619,   402,   271],
         [10899,  2138,   257,  7026],
         [15632,   438,  2016,   257],
         [  922,  5891,  1576,   438],
         [  568,   340,   373,   645],
         [ 1049,  5975,   284,   502],
         [  284,  3285,   326,    11]])

Targets:
 tensor([[  367,  2885,  1464,  1807],
         [ 3619,   402,   271, 10899],
         [ 2138,   257,  7026, 15632],
         [  438,  2016,   257,   922],
         [ 5891,  1576,   438,   568],
         [  340,   373,   645,  1049],
         [ 5975,   284,   502,   284],
         [ 3285,   326,    11,   287]])
```

Here, with `stride=4` (equal to `max_length`), each row is a completely fresh, non-overlapping chunk of the text — and each row's target is, again, that row's input shifted one position to the right.

---

## Cheat Sheet

- **Training signal for next-word prediction:** input = a chunk of `context_size` tokens; target = the same chunk shifted right by one token.
- **`context_size` / `max_length`** — how many tokens of context the model sees before predicting the next one.
- **`stride`** — how far the sliding window advances each step:
  - `stride == max_length` → non-overlapping chunks.
  - `stride < max_length` → overlapping chunks (more training samples, more redundancy).
- **`GPTDatasetV1`** — a PyTorch `Dataset` that tokenizes raw text once, then slices it into `(input_chunk, target_chunk)` pairs using the sliding window.
- **`create_dataloader_v1`** — wraps `GPTDatasetV1` in a PyTorch `DataLoader` for batching, shuffling, and efficient loading during training.
- **Output shape:** each batch returns `(inputs, targets)`, both of shape `(batch_size, max_length)`, ready to be turned into embeddings next.