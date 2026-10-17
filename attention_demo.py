"""Step 3: understand self-attention in three small steps (nothing is trained here)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
T = 4  # sequence length (4 tokens) so the matrices are easy to read

tril = torch.tril(torch.ones(T, T))  # 1s on and below the diagonal, 0s above

# ---------- Part 1: average of all past tokens via matrix multiply ----------
x = torch.randn(1, T, 2)  # (batch, time, channels)
print("x (4 tokens, 2 numbers each):\n", x[0], "\n")

wei = tril / tril.sum(dim=1, keepdim=True)  # each row sums to 1
print("Averaging weights (row t averages tokens 0..t):\n", wei, "\n")
print("Output (each row = mean of the tokens so far):\n", (wei @ x)[0], "\n")

# ---------- Part 2: the same weights built with mask + softmax ----------
wei2 = torch.zeros(T, T)
wei2 = wei2.masked_fill(tril == 0, float("-inf"))  # forbid looking at the future
wei2 = F.softmax(wei2, dim=-1)
print("Part 2 weights equal Part 1 weights:", torch.allclose(wei, wei2), "\n")

# ---------- Part 3: one head of real self-attention ----------
C = 8            # embedding size of each token
head_size = 4    # size of the query/key/value vectors
x = torch.randn(1, T, C)

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k, q, v = key(x), query(x), value(x)          # each (1, T, head_size)
scores = q @ k.transpose(-2, -1)              # (1, T, T): how much token i likes token j
scores = scores * head_size ** -0.5           # scale so softmax doesn't get too peaky
scores = scores.masked_fill(tril == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)           # (1, T, T), each row sums to 1
out = weights @ v                             # (1, T, head_size)

print("Attention weights (learned-style, row t only looks at 0..t):\n", weights[0].detach(), "\n")
print("Row sums:", weights[0].sum(dim=-1).detach())
print("Output shape:", tuple(out.shape))