"""Step 2: batching + the simplest possible language model (bigram baseline)."""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ---- settings (we'll reuse these ideas in the real GPT) ----
batch_size = 32      # how many sequences we train on in parallel
block_size = 8       # context length: how many characters the model can see
max_iters = 3000     # training steps
eval_interval = 300  # how often we print loss
eval_iters = 200     # batches averaged when measuring loss
learning_rate = 1e-2
device = "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- load data made by prepare.py ----
with open(os.path.join(HERE, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size, itos = meta["vocab_size"], meta["itos"]


def load(name):
    arr = np.fromfile(os.path.join(HERE, name), dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))


train_data, val_data = load("train.bin"), load("val.bin")


def get_batch(split):
    """Return x (inputs) and y (targets = inputs shifted by one character)."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # A lookup table: row = current char, columns = score for every possible next char
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # shape (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                 # only the last position matters
            probs = F.softmax(logits, dim=-1)         # scores -> probabilities
            nxt = torch.multinomial(probs, num_samples=1)  # sample one char
            idx = torch.cat([idx, nxt], dim=1)        # append and repeat
        return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


model = BigramLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Peek at what a batch looks like
xb, yb = get_batch("train")
print("x shape:", tuple(xb.shape), "| y shape:", tuple(yb.shape))
print("x[0]:", xb[0].tolist())
print("y[0]:", yb[0].tolist(), "(same, shifted by one)")
print(f"Loss expected from pure guessing: ln({vocab_size}) = {np.log(vocab_size):.3f}\n")

for step in range(max_iters + 1):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n--- generated text ---")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(start, max_new_tokens=300)[0].tolist()
print("".join(itos[i] for i in out))