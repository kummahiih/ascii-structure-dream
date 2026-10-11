#!/usr/bin/env python3
"""
ASCII Structure Dream — fast n-gram version

Deep Dream-like process on a grid of random letters.
A character n-gram model (English letter statistics) scores the grid.
Mutations are accepted when they raise the n-gram likelihood
(i.e. the text starts looking more like English letter sequences).

Fast enough to watch even on larger grids. No transformers required.

Usage:
    python dream.py                  # 40x20
    python dream.py --size 10
    python dream.py --size 80 --mut 20
"""

import argparse
import os
import sys
import time
import random
import string
import math
from collections import Counter

# English letter + space unigram / bigram / trigram counts
# (approximate, derived from common English text statistics)
ENGLISH_UNI = {
    ' ': 0.18, 'e': 0.12, 't': 0.09, 'a': 0.08, 'o': 0.075, 'i': 0.07,
    'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043, 'l': 0.040,
    'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.024, 'f': 0.022, 'g': 0.020,
    'y': 0.020, 'p': 0.019, 'b': 0.015, 'v': 0.010, 'k': 0.008, 'j': 0.0015,
    'x': 0.0015, 'q': 0.001, 'z': 0.0007,
}
# a few strong English bigrams (log-prob bonuses)
ENGLISH_BI = {
    'th': 2.5, 'he': 2.2, 'in': 1.8, 'er': 1.7, 'an': 1.6, 're': 1.5,
    'on': 1.4, 'at': 1.3, 'en': 1.3, 'nd': 1.2, 'ti': 1.2, 'es': 1.2,
    'or': 1.1, 'te': 1.1, 'of': 1.1, 'ed': 1.0, 'is': 1.0, 'it': 1.0,
    'al': 1.0, 'ar': 0.9, 'st': 0.9, 'to': 0.9, 'nt': 0.9, 'ng': 0.8,
    'se': 0.8, 'ha': 0.8, 'as': 0.7, 'ou': 0.7, 'io': 0.7, 'le': 0.7,
    've': 0.6, 'co': 0.6, 'me': 0.6, 'de': 0.6, 'hi': 0.6, 'ri': 0.6,
    'ro': 0.5, 'ic': 0.5, 'ne': 0.5, 'ea': 0.5, 'ra': 0.5, 'ce': 0.5,
    'li': 0.5, 'ch': 0.5, 'll': 0.4, 'be': 0.4, 'ma': 0.4, 'si': 0.4,
    'om': 0.4, 'ur': 0.4,
}
# strong English trigrams
ENGLISH_TRI = {
    'the': 3.0, 'and': 2.5, 'ing': 2.2, 'her': 1.8, 'hat': 1.6, 'his': 1.5,
    'tha': 1.5, 'ere': 1.4, 'for': 1.4, 'ent': 1.3, 'ion': 1.3, 'ter': 1.2,
    'was': 1.2, 'you': 1.2, 'ith': 1.1, 'ver': 1.1, 'all': 1.0, 'wit': 1.0,
    'thi': 1.0, 'tio': 1.0,
}

CHARS = string.ascii_lowercase + " "
# precompute log unigram
LOG_UNI = {c: math.log(p + 1e-6) for c, p in ENGLISH_UNI.items()}
for c in CHARS:
    if c not in LOG_UNI:
        LOG_UNI[c] = math.log(1e-4)


def random_grid(h, w):
    return [[random.choice(CHARS) for _ in range(w)] for _ in range(h)]


def mutate(grid, n_mut):
    h, w = len(grid), len(grid[0])
    new = [row[:] for row in grid]
    for _ in range(n_mut):
        y = random.randrange(h)
        x = random.randrange(w)
        if random.random() < 0.25 and x + 2 < w:
            # seed a short common English fragment
            frag = random.choice(["the", "and", "ing", "er ", "th", "he", "in", "on", "at", "to"])
            for i, c in enumerate(frag):
                if x + i < w:
                    new[y][x + i] = c
        else:
            new[y][x] = random.choice(CHARS)
    return new


def score_grid(grid):
    """Higher is better (log-likelihood under English n-grams)."""
    s = 0.0
    h, w = len(grid), len(grid[0])
    for y in range(h):
        row = grid[y]
        for x in range(w):
            c = row[x]
            s += LOG_UNI.get(c, -8.0)
            if x + 1 < w:
                bi = c + row[x + 1]
                s += ENGLISH_BI.get(bi, 0.0)
            if x + 2 < w:
                tri = c + row[x + 1] + row[x + 2]
                s += ENGLISH_TRI.get(tri, 0.0)
    return s / (h * w)  # average per cell so size doesn't dominate


def clear_screen():
    print("\033[2J\033[H", end="")


def display(grid, accepted, score, elapsed, proposals):
    clear_screen()
    print(f"ASCII n-gram Dream  acc={accepted:4d}/{proposals:<4d}  score={score:.3f}  t={elapsed:.1f}s")
    print("─" * len(grid[0]))
    for row in grid:
        print("".join(row))
    print("─" * len(grid[0]))
    print("Letters+space. Higher score = more English-like. Ctrl-C to stop.")


def main():
    p = argparse.ArgumentParser(description="ASCII Structure Dream (fast n-gram)")
    p.add_argument("--size", type=int, default=40, help="Grid width")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--steps", type=int, default=0, help="Max proposals (0=forever)")
    p.add_argument("--mut", type=int, default=6, help="Mutations per proposal")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    w = args.size
    h = args.height if args.height is not None else max(8, w // 2)
    if w <= 20:
        h = w
        args.mut = max(1, min(args.mut, 3))

    grid = random_grid(h, w)
    current = score_grid(grid)

    start = time.time()
    proposals = 0
    accepted = 0

    try:
        while args.steps == 0 or proposals < args.steps:
            display(grid, accepted, current, time.time() - start, proposals)

            cand = mutate(grid, args.mut)
            sc = score_grid(cand)
            # accept improvement; tiny chance of equal/worse so it doesn't freeze
            if sc > current or random.random() < 0.02:
                grid = cand
                current = sc
                accepted += 1

            proposals += 1
            # yield so the terminal can refresh and Ctrl-C works
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"Accepted {accepted} / {proposals} proposals. Final score: {current:.3f}")


if __name__ == "__main__":
    main()
