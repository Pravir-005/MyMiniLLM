"""Day 4: train a BPE tokenizer on TinyShakespeare and write train_bpe.bin / val_bpe.bin.

Run prepare.py first (it creates input.txt and meta.pkl).
Usage:  python3 prepare_bpe.py [vocab_size]     (default 512)
"""
import os
import pickle
import sys
import time

import numpy as np

from bpe import build_vocab, encode, train_bpe

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 512

with open(os.path.join(HERE, "input.txt"), "r", encoding="utf-8") as f:
    text = f.read()
with open(os.path.join(HERE, "meta.pkl"), "rb") as f:
    char_meta = pickle.load(f)
stoi, char_itos = char_meta["stoi"], char_meta["itos"]
base_vocab = char_meta["vocab_size"]  # 65 characters

# Same 90/10 split as prepare.py, done on the raw text BEFORE learning merges, so the
# tokenizer never sees validation text (otherwise val loss would be slightly flattered).
n = int(0.9 * len(text))
train_text, val_text = text[:n], text[n:]
train_ids = [stoi[c] for c in train_text]
val_ids = [stoi[c] for c in val_text]

n_merges = VOCAB_SIZE - base_vocab
print(f"Learning {n_merges} merges (vocab {base_vocab} -> {VOCAB_SIZE}) on {len(train_ids):,} characters...")
t0 = time.time()
train_tok, merges = train_bpe(train_ids, base_vocab, n_merges)
print(f"Done in {time.time() - t0:.0f}s")
val_tok = encode(val_ids, merges, base_vocab)

itos = build_vocab(char_itos, merges, base_vocab)
vocab_size = base_vocab + len(merges)

# --- what did it learn? ---
print(f"\nTrain: {len(train_text):,} chars -> {len(train_tok):,} tokens "
      f"({len(train_text) / len(train_tok):.2f} chars per token)")
print(f"Val:   {len(val_text):,} chars -> {len(val_tok):,} tokens")
print("\nFirst 15 merges (most frequent pairs first):")
for k in range(min(15, len(merges))):
    a, b = merges[k]
    print(f"  {itos[a]!r} + {itos[b]!r} -> {itos[base_vocab + k]!r}")
print("\nLast 10 merges (rarer, longer chunks):")
for k in range(max(0, len(merges) - 10), len(merges)):
    print(f"  -> {itos[base_vocab + k]!r}")
demo = "ROMEO: Wherefore art thou"
demo_tok = encode([stoi[c] for c in demo], merges, base_vocab)
print(f"\n{demo!r} becomes {len(demo_tok)} tokens: {[itos[i] for i in demo_tok]}")

# --- save ---
np.array(train_tok, dtype=np.uint16).tofile(os.path.join(HERE, "train_bpe.bin"))
np.array(val_tok, dtype=np.uint16).tofile(os.path.join(HERE, "val_bpe.bin"))
with open(os.path.join(HERE, "meta_bpe.pkl"), "wb") as f:
    pickle.dump(
        {
            "vocab_size": vocab_size,
            "itos": itos,
            "stoi": stoi,                  # character -> id (used to encode prompts)
            "merges": merges,
            "base_vocab": base_vocab,
            "val_tokens_per_char": len(val_tok) / len(val_text),  # for bits-per-character
        },
        f,
    )
print("\nSaved train_bpe.bin, val_bpe.bin, meta_bpe.pkl")