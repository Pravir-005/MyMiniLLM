"""Step 1: get data, build a character-level tokenizer, split into train/val."""
import os
import pickle
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(HERE, "input.txt")

# 1) Download the raw text (about 1 MB) if we don't have it yet
if not os.path.exists(TEXT_PATH):
    print("Downloading TinyShakespeare...")
    urllib.request.urlretrieve(URL, TEXT_PATH)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()
print(f"Characters in dataset: {len(text):,}")

# 2) Tokenizer: every unique character becomes one token id
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # string -> integer
itos = {i: ch for ch, i in stoi.items()}      # integer -> string


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


print(f"Vocab size: {vocab_size}")
print("Sample encode/decode:", encode("Hello"), "->", decode(encode("Hello")))

# 3) Turn the whole text into one long array of token ids
data = np.array(encode(text), dtype=np.uint16)

# 4) 90% train, 10% validation (validation = text the model never trains on)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
train_data.tofile(os.path.join(HERE, "train.bin"))
val_data.tofile(os.path.join(HERE, "val.bin"))

# 5) Save tokenizer so later scripts can reuse it
with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")