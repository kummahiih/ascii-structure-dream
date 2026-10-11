# ASCII Structure Dream

Deep Dream-like evolutionary process on a grid of random ASCII characters.

A small randomly-initialized CNN scores "structure" (magnitude of intermediate feature maps).  
Random mutations are proposed; they are accepted only when the structure score rises.  
The grid slowly organizes into whatever patterns the frozen network finds salient.

Watch it evolve live in the terminal.

## Requirements

- Python 3.8+
- PyTorch (CPU is fine; CUDA used automatically if available)
- ~12 GB VRAM is more than enough for any size here

```bash
pip install -r requirements.txt
```

## Run

```bash
# Terminal-friendly default (80×40)
python dream.py

# The requested 500×500 (display will center-crop to your terminal size)
python dream.py --size 500

# Longer run, more mutations per step
python dream.py --size 120 --steps 2000 --mut 40

# Force CPU
python dream.py --cpu
```

Ctrl-C to stop. The process prints the current grid (or a center crop when the grid is larger than the terminal) after each accepted mutation.

## How it works

1. Start with a uniform random printable-ASCII grid.
2. A small 3-layer CNN (fixed random weights) evaluates a structure score = mean squared activation of the feature maps.
3. Propose `N` random cell changes.
4. Accept the proposal only if the score increases.
5. Repeat. Over time local patterns the network responds to are reinforced.

This is a discrete, mutation-based cousin of the original DeepDream gradient-ascent procedure.

## GPU note

The code checks `torch.cuda.is_available()` and moves the model and tensors to GPU when present.  
Your 12 GB card easily handles a 500×500 single-channel grid and this tiny network.

## Related

Part of the broader experimental line that includes [regret-heuristic](https://github.com/kummahiih/regret-heuristic) and [intent-readout-search](https://github.com/kummahiih/intent-readout-search).
