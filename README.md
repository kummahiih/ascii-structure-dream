# ASCII Structure Dream

Fast Deep Dream-like process on a grid of random letters.

A character n-gram model (English letter / bigram / trigram statistics) scores the grid.  
Mutations are accepted when they raise the score — i.e. when the text starts looking more like English letter sequences.

No transformers, no GPU required. Fast enough to watch live even on larger grids.

## Run

```bash
# 10×10 test
python dream.py --size 10

# medium
python dream.py --size 40

# bigger
python dream.py --size 80 --mut 15
```

Ctrl-C to stop.

## What you should see

Short English-like fragments (`the`, `and`, `ing`, `er`, …) gradually appear and spread because those n-grams raise the score. It is not a full language model, but it is fast enough that the terminal actually updates and structure becomes visible.

## Why the change

Full LM scoring in Python was too slow for anything beyond tiny grids, so the display froze. This version evaluates a whole grid in milliseconds.
