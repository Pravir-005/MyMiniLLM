"""A minimal BPE (byte-pair encoding) tokenizer, written from scratch.

Idea: start with one token per character. Repeatedly find the most frequent pair of
neighbouring tokens and replace every occurrence with a brand-new token. Each such
replacement is called a "merge". After N merges the vocabulary is (characters + N).
"""
from collections import Counter


def merge_pair(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    a, b = pair
    out = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(ids, base_vocab, n_merges, log_every=50):
    """Learn `n_merges` merges from a list of token ids. Returns (merged ids, merges)."""
    merges = []
    for k in range(n_merges):
        counts = Counter(zip(ids, ids[1:]))   # how often does each neighbouring pair occur?
        if not counts:
            break
        pair = max(counts, key=counts.get)    # most frequent pair
        new_id = base_vocab + k               # the next free token id
        ids = merge_pair(ids, pair, new_id)
        merges.append(pair)
        if log_every and (k + 1) % log_every == 0:
            print(f"  merge {k + 1}/{n_merges} | tokens left: {len(ids):,}")
    return ids, merges


def encode(ids, merges, base_vocab):
    """Apply the learned merges, in the order they were learned, to new text (as char ids)."""
    for k, pair in enumerate(merges):
        ids = merge_pair(ids, pair, base_vocab + k)
    return ids


def build_vocab(char_itos, merges, base_vocab):
    """Map every token id to the text it stands for (a merged token = its two parts joined)."""
    itos = dict(char_itos)
    for k, (a, b) in enumerate(merges):
        itos[base_vocab + k] = itos[a] + itos[b]
    return itos