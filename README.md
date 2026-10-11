# ASCII Structure Dream

Deep Dream-like process on a grid of random ASCII characters, scored by a **language model**.

Mutations are accepted when they lower the LM’s perplexity on short windows (the text becomes more probable under the model).

## Faster defaults

The script now defaults to **gpt2** (~124M) and includes speed knobs:

```bash
# fastest interactive
python dream.py --fast

# still LM, but quicker
python dream.py --model gpt2 --windows 2 --score-every 4 --size 60

# larger model when you want more sensitivity (slower)
python dream.py --model Qwen/Qwen2.5-1.5B --size 48 --windows 3
```

`--fast` reduces windows, shortens them, and only scores every 4 proposals (other proposals are accepted with low probability to keep the grid moving).

## Requirements

```bash
pip install -r requirements.txt
```

4-bit is used automatically on CUDA for larger models.

## Full 500×500

```bash
python dream.py --size 500 --fast
```

Display center-crops to the terminal. Scoring stays on short windows so context length is never the limit.

## GPU

12 GB 4070 Ti easily runs gpt2, 1.5B, and 7B/8B in 4-bit. Smaller models are recommended for live watching.

## Related

- [regret-heuristic](https://github.com/kummahiih/regret-heuristic)
- [intent-readout-search](https://github.com/kummahiih/intent-readout-search)
