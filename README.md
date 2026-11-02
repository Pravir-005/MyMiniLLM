# MiNigpt

A small GPT-style language model built from scratch in PyTorch, trained character-by-character on TinyShakespeare. Built for fun and to understand how LLMs work, one piece at a time.

Inspired by Andrej Karpathy's *"Let's build GPT"* video and the nanoGPT project.

## Results so far

| Model | Parameters | Val loss |
|---|---|---|
| Random guessing | n/a | 4.17 (ln 65) |
| Bigram baseline (`bigram.py`) | 4,225 | 2.49 |
| Small GPT (`gpt.py`) | 824,320 | **1.62** |

The GPT trains for 3000 steps in about 70 seconds on a single consumer GPU. Train loss ended at 1.44 against val loss of 1.62, a mild overfitting gap I plan to study when scaling up.

Sample from the small GPT (`--prompt "ROMEO:" --temp 0.7 --top_k 20`):

```
ROMEO:
All before with a purpose be made the way age.
Stirrah, what have been thee should them comest
Have away with himself my lord
```

It has learned the *shape* of a play (speaker names, line breaks, verse-length lines, mostly real words) but not meaning, which is expected at this size.

## Project layout

| File | What it does |
|---|---|
| `prepare.py` | Downloads TinyShakespeare, builds a character tokenizer, writes train/val splits |
| `bigram.py` | Batching + the simplest language model, used as a baseline |
| `attention_demo.py` | Prints causal averaging, softmax masking and one self-attention head step by step |
| `gpt.py` | Full decoder-only transformer (multi-head attention, MLP, residuals, LayerNorm) and training loop |
| `sample.py` | Generates text from a saved checkpoint with temperature, top-k and a custom prompt |

## Run it

```bash
pip install torch numpy

python prepare.py          # downloads data, writes train.bin / val.bin / meta.pkl
python bigram.py           # baseline, loss ~2.5
python GPT.py              # trains the GPT, saves ckpt.pt
python sample.py --prompt "ROMEO:" --temp 0.7 --top_k 20 --n 600
```

`gpt.py` uses CUDA automatically if available and falls back to CPU otherwise (slower, but fine for this size).

## Roadmap

- [x] Day 1: data pipeline, character tokenizer, bigram baseline
- [x] Day 2: self-attention and a full small GPT
- [ ] Day 3: scale to ~10M parameters, study overfitting, tune hyperparameters
- [ ] Day 4: train a BPE tokenizer, compare against character-level, write up findings
