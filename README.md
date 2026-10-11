# ASCII Structure Dream

Deep Dream-like process on a grid of random ASCII characters, scored by a **language model**.

A causal LM (default Qwen2.5-1.5B, or any model that fits your 12 GB 4070 Ti in 4-bit) evaluates short windows of the grid by perplexity. Random mutations are accepted only when they lower the average perplexity — i.e. when the random text starts looking more like sequences the model assigns probability to.

Watch it evolve in the terminal.

## Requirements

```bash
pip install -r requirements.txt
```

Works on CPU; uses CUDA + 4-bit automatically when a GPU is present.

## Run

```bash
# Small & fast (default model Qwen2.5-1.5B)
python dream.py

# The 500×500 grid (display center-crops to terminal)
python dream.py --size 500

# Use a model you already have locally, or a 7B/8B that fits in 4-bit
python dream.py --model Qwen/Qwen2.5-7B-Instruct --size 80 --windows 4 --mut 8

# Gemma-3-4B, Granite, etc. also work
python dream.py --model google/gemma-2-2b
```

Ctrl-C to stop.

## How it works

1. Random printable-ASCII grid.
2. Draw several random windows (default 128 chars).
3. Score each window with the LM’s perplexity.
4. Propose random cell mutations.
5. Accept only if mean perplexity falls.
6. Redraw (or center-crop) so you can watch structure emerge.

Because full 500×500 text far exceeds any context window, scoring is always done on short windows. Larger models are slower but can be more sensitive to longer-range patterns.

## GPU

Your 12 GB 4070 Ti is enough for 7B–8B models in 4-bit (the same class used in [intent-readout-search](https://github.com/kummahiih/intent-readout-search)). The script loads with bitsandbytes 4-bit by default.

## Related

- [regret-heuristic](https://github.com/kummahiih/regret-heuristic)
- [intent-readout-search](https://github.com/kummahiih/intent-readout-search)
