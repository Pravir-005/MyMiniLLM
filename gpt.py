"""Step 4: a small GPT (decoder-only transformer) trained on TinyShakespeare."""
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# ---------------- config (sized for a CPU; scale up on a GPU) ----------------
batch_size = 32
block_size = 128      # context length
n_embd = 128          # embedding size
n_head = 4            # attention heads (head size = 128 / 4 = 32)
n_layer = 4           # transformer blocks
dropout = 0.1
max_iters = 3000
eval_interval = 250
eval_iters = 50
max_lr = 1e-3
min_lr = 1e-4
warmup_iters = 100
grad_clip = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "ckpt.pt")

# ---------------- data ----------------
with open(os.path.join(HERE, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size, itos = meta["vocab_size"], meta["itos"]


def load(name):
    arr = np.fromfile(os.path.join(HERE, name), dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))


train_data, val_data = load("train.bin"), load("val.bin")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------- model ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)  # q, k, v in one layer
        self.proj = nn.Linear(n_embd, n_embd, bias=False)     # mixes the heads together
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        hs = C // n_head
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size): each head works independently
        q = q.view(B, T, n_head, hs).transpose(1, 2)
        k = k.view(B, T, n_head, hs).transpose(1, 2)
        v = v.view(B, T, n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * hs ** -0.5            # (B, n_head, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = att @ v                                              # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # glue heads back together
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual connection around attention
        x = x + self.mlp(self.ln2(x))   # residual connection around MLP
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                              # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                   # never exceed the context length
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
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


def get_lr(it):
    if it < warmup_iters:                                     # linear warmup
        return max_lr * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)  # cosine decay


if __name__ == "__main__":
    model = GPT().to(device)
    print(f"device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

    best_val = float("inf")
    t0 = time.time()
    for it in range(max_iters + 1):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.5f} | {time.time() - t0:.0f}s elapsed")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict()}, CKPT_PATH)

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    print("\n--- sample from the final model ---")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=500, temperature=0.8)[0].tolist()
    print("".join(itos[i] for i in out))