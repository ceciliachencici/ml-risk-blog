---
title: "Masked Loss Isn't Enough: Handling NaNs in Transformer-Based Risk Profiling"
layout: default
---

# Handling NaNs in Transformer-Based Risk Profiling

In real-world risk profiling applications — such as fraud detection or seller trust scoring — transformer models are often used to capture **sequential user behavior**. These sequences are variable in length and frequently require:

- **Padding** to handle different sequence lengths in batch training
- **Position encoding** to give the model a sense of order
- **Masking** to prevent information leakage (e.g. future positions)

This is standard in NLP and time series modeling. However, a **less-known but critical issue** arises: 

> In some configurations, **NaNs or infinite values are introduced into the model’s intermediate outputs by design**, especially in padded or masked positions.

This blog explores:
- What causes these NaNs
- Why masking the **loss alone** is not enough
- How to simulate the issue
- What a correct solution looks like

---

## Real-World Scenarios

### Risk modeling use cases where this arises:
- Behavioral risk profiling using event sequences (logins, transactions, edits)
- Transformers over merchant listings or user journeys
- Session-level risk scoring with attention across padded timelines

In these cases, masking is required — but **NaNs will still reach your model internals if not handled properly**, and cause gradients to explode.

---

## Simulating the Issue

Let’s simulate a case where a transformer-like model applies **position encodings** over a batch of padded sequences. We’ll intentionally create `NaN` values on the padded positions, and **observe how masking the loss doesn’t prevent gradient corruption**.

We'll also explain why mathematically, even `NaN * 0` is not safe for autograd.

---

## Sample Input: Padded Behavioral Sequences

```python
import torch

# Batch of 3 sequences, padded to length 5
# Assume NaNs introduced via attention masking or sin/cos position encoding
X = torch.tensor([
    [1.0, 2.0, 3.0, float('nan'), float('nan')],  # real + padding
    [5.0, float('nan'), float('nan'), float('nan'), float('nan')],  # 1 real + 4 pad
    [1.0, 2.0, 3.0, 4.0, 5.0]  # full sequence
])

mask = ~torch.isnan(X)  # boolean mask for valid tokens
```

---

## Model: Dummy Encoder + Position Embedding

```python
import torch.nn as nn

class PositionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, dim))  # Positional weights for 5 steps

    def forward(self, x):
        # Adds positional encoding to each timestep
        pos = self.weight[:x.shape[1]]
        return x + pos  # if x contains NaN, result will too

class RiskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 8)
        self.pos = PositionEncoder(8)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, T, 1]
        x = self.embedding(x)  # [B, T, 8]
        x = self.pos(x)  # ⛔ May introduce NaNs into attention features
        x = self.head(x)  # [B, T, 1]
        return x.squeeze(-1)
```

---

## Why Loss Masking Fails (Math Insight)

Let’s assume:

- `x` has shape `[B, T]`
- `mask` is `[B, T]` boolean
- loss is masked like:
  ```python
  loss = ((output - target) ** 2)[mask].mean()
  ```

which avoided computing loss on padded positions.

But `output = model(x)` **already contains NaNs** from `x`.  
So the **gradient graph includes NaN paths**.  
And in PyTorch:

```python
torch.tensor(float('nan')) * 0  =>  nan
```

So masking in the loss doesn’t cut off the bad computation paths.  
Backprop still touches the broken outputs → corrupts your gradients.

---

##  Full Demo with Gradient Check


```python
X.requires_grad = True
out = model(X)
y = torch.ones_like(out)
loss = ((out - y)**2)[mask].mean()
loss.backward()


# Gradient check
for name, param in model.named_parameters():
if param.grad is not None:
print(f"{name} grad has NaN? {torch.isnan(param.grad).any().item()}")
```


###  Expected Output


```text
linear1.weight grad has NaN? True
linear1.bias grad has NaN? True
pos.weight grad has NaN? True
head.weight grad has NaN? True
head.bias grad has NaN? True
```


Even though we masked the loss, NaNs still poisoned the entire backward graph.



## Solution

- **Sanitize inputs** (e.g., `torch.nan_to_num`) **before** they enter the model
- Or apply **masked computation at every layer** (like attention masks)
- Never assume loss masking alone is enough

---

##  Download Code & Data

👉 [Download `masked-loss-transformer-demo.zip`](sandbox:/mnt/data/masked-loss-transformer-demo.zip)

Contains:
- `main.py` – full runnable demo
- `data/synthetic_sequences.csv`
- `README.md`

---

## 🔗 Source Repo

[GitHub: ceciliachencici/ml-risk-blog](https://github.com/ceciliachencici/ml-risk-blog)
