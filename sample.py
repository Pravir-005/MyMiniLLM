"""Generate text from the checkpoint saved by gpt.py.

Examples:
    python3 sample.py
    python3 sample.py --prompt "ROMEO:" --temp 0.7 --top_k 20 --n 600
"""
import argparse

import torch
import torch.nn.functional as F

# Reuses the model definition and config from gpt.py (training only runs when gpt.py is executed directly)
from GPT import CKPT_PATH, GPT, block_size, device, meta

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="\n", help="text to start from")
parser.add_argument("--n", type=int, default=500, help="characters to generate")
parser.add_argument("--temp", type=float, default=0.8, help="below 1 = safer, above 1 = wilder")
parser.add_argument("--top_k", type=int, default=0, help="keep only the k likeliest chars (0 = off)")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

stoi, itos = meta["stoi"], meta["itos"]

model = GPT().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Encode the prompt, skipping any character the tokenizer has never seen
ids = [stoi[c] for c in args.prompt if c in stoi] or [0]
idx = torch.tensor([ids], dtype=torch.long, device=device)

with torch.no_grad():
    for _ in range(args.n):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / max(args.temp, 1e-6)
        if args.top_k > 0:
            kth_best = torch.topk(logits, min(args.top_k, logits.size(-1)))[0][:, [-1]]
            logits[logits < kth_best] = float("-inf")  # forbid everything outside the top k
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)

print("".join(itos[i] for i in idx[0].tolist()))