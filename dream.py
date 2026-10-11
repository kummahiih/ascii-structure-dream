#!/usr/bin/env python3
"""
ASCII Structure Dream

Deep Dream-like process on a grid of ASCII characters.
Starts from random printable ASCII. A small CNN scores "structure"
(activation magnitude of intermediate feature maps). Random mutations
are accepted only when they increase the structure score.

Watch it evolve in the terminal.

Usage:
    python dream.py                  # 80x40, good for terminal
    python dream.py --size 500       # full 500x500 (display crops to terminal)
    python dream.py --size 200 --steps 1000 --mut 50
"""

import argparse
import os
import sys
import time
import random
import string

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False
    print("PyTorch not found. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

# Printable ASCII excluding control chars. Space to ~
CHARS = string.printable.strip()
# Prefer denser set for visual interest
CHARS = "".join(c for c in string.printable if 32 <= ord(c) <= 126)
N_CHARS = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}


def grid_to_tensor(grid, device):
    """Convert list-of-lists of chars to (1, 1, H, W) float tensor in [0, 1]."""
    h, w = len(grid), len(grid[0])
    t = torch.zeros(1, 1, h, w, device=device)
    for y in range(h):
        for x in range(w):
            t[0, 0, y, x] = ord(grid[y][x]) / 127.0
    return t


class StructureNet(nn.Module):
    """Small CNN whose intermediate activations define 'structure'."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        # Fixed random weights — the "dream" is what this particular net finds interesting
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def structure_score(self, x):
        """Higher when intermediate maps have strong local patterns."""
        with torch.no_grad():
            feat = self.forward(x)
            # L2 norm of features + local variance encouragement
            return feat.pow(2).mean().item()


def random_grid(h, w):
    return [[random.choice(CHARS) for _ in range(w)] for _ in range(h)]


def mutate(grid, n_mut):
    """Return a new grid with n_mut random cell changes."""
    h, w = len(grid), len(grid[0])
    new = [row[:] for row in grid]
    for _ in range(n_mut):
        y = random.randrange(h)
        x = random.randrange(w)
        new[y][x] = random.choice(CHARS)
    return new


def clear_screen():
    # ANSI clear + home
    print("\033[2J\033[H", end="")


def display(grid, step, score, elapsed, crop=None):
    """Print grid (or center crop) to terminal."""
    h, w = len(grid), len(grid[0])
    if crop and (w > crop[0] or h > crop[1]):
        # center crop
        cw, ch = crop
        x0 = max(0, (w - cw) // 2)
        y0 = max(0, (h - ch) // 2)
        view = [row[x0:x0 + cw] for row in grid[y0:y0 + ch]]
        label = f" [crop {cw}x{ch} of {w}x{h}]"
    else:
        view = grid
        label = ""
    clear_screen()
    print(f"ASCII Structure Dream  step={step:5d}  score={score:.5f}  time={elapsed:.1f}s{label}")
    print("─" * min(len(view[0]) if view else 0, 120))
    for row in view:
        print("".join(row))
    print("─" * min(len(view[0]) if view else 0, 120))
    print("Ctrl-C to stop. Mutations accepted only when structure score rises.")


def main():
    p = argparse.ArgumentParser(description="ASCII Structure Dream")
    p.add_argument("--size", type=int, default=80, help="Grid side length (default 80; use 500 for full request)")
    p.add_argument("--height", type=int, default=None, help="Override height (default = size//2 or size)")
    p.add_argument("--steps", type=int, default=0, help="Max steps (0 = run forever)")
    p.add_argument("--mut", type=int, default=20, help="Mutations per proposal")
    p.add_argument("--display-every", type=int, default=1, help="Redraw every N accepted steps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    w = args.size
    h = args.height if args.height else (args.size if args.size <= 120 else args.size // 2)
    # For 500 request: square 500x500 is extreme for terminal; default height smaller unless forced
    if args.size == 500 and args.height is None:
        h = 500  # honor the request; display will crop

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}", file=sys.stderr)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)

    net = StructureNet().to(device).eval()
    grid = random_grid(h, w)

    # Terminal crop for display when grid is huge
    term_w = 120
    term_h = 40
    try:
        # rough terminal size
        term_w = min(os.get_terminal_size().columns - 2, 160)
        term_h = min(os.get_terminal_size().lines - 6, 50)
    except Exception:
        pass

    start = time.time()
    step = 0
    accepted = 0
    score = 0.0

    try:
        while args.steps == 0 or step < args.steps:
            # Evaluate current
            t = grid_to_tensor(grid, device)
            score = net.structure_score(t)

            if step % args.display_every == 0 or accepted == 0:
                display(grid, accepted, score, time.time() - start,
                        crop=(term_w, term_h) if (w > term_w or h > term_h) else None)

            # Propose mutation
            cand = mutate(grid, args.mut)
            t_cand = grid_to_tensor(cand, device)
            score_cand = net.structure_score(t_cand)

            if score_cand > score:
                grid = cand
                score = score_cand
                accepted += 1

            step += 1
            # small yield so Ctrl-C is responsive
            if step % 10 == 0:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"Accepted {accepted} mutations in {step} proposals. Final score: {score:.5f}")


if __name__ == "__main__":
    main()
