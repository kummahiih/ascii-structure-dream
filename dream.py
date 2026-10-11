#!/usr/bin/env python3
"""
ASCII Structure Dream — language-model version

Deep Dream-like process on a grid of random ASCII characters.
A causal language model scores local windows of the grid by perplexity.
Random mutations are accepted only when they lower the average perplexity
(i.e. the text starts looking more like something the model assigns probability to).

Designed for a 12 GB GPU (4-bit load). 500×500 is supported; scoring is done
on short sliding windows so context length is never exceeded.

Usage:
    python dream.py
    python dream.py --size 500 --model Qwen/Qwen2.5-1.5B
    python dream.py --model google/gemma-2-2b --windows 8 --mut 15
"""

import argparse
import os
import sys
import time
import random
import string
import math

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    print("Missing dependencies. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

CHARS = "".join(c for c in string.printable if 32 <= ord(c) <= 126)


def random_grid(h, w):
    return [[random.choice(CHARS) for _ in range(w)] for _ in range(h)]


def mutate(grid, n_mut):
    h, w = len(grid), len(grid[0])
    new = [row[:] for row in grid]
    for _ in range(n_mut):
        y = random.randrange(h)
        x = random.randrange(w)
        new[y][x] = random.choice(CHARS)
    return new


def grid_to_lines(grid):
    return ["".join(row) for row in grid]


def clear_screen():
    print("\033[2J\033[H", end="")


def display(grid, step, score, elapsed, crop=None):
    h, w = len(grid), len(grid[0])
    if crop and (w > crop[0] or h > crop[1]):
        cw, ch = crop
        x0 = max(0, (w - cw) // 2)
        y0 = max(0, (h - ch) // 2)
        view = [row[x0:x0 + cw] for row in grid[y0:y0 + ch]]
        label = f" [crop {cw}x{ch} of {w}x{h}]"
    else:
        view = grid
        label = ""
    clear_screen()
    print(f"ASCII LM Dream  accepted={step:5d}  ppl={score:.2f}  time={elapsed:.1f}s{label}")
    print("─" * min(len(view[0]) if view else 0, 140))
    for row in view:
        print("".join(row))
    print("─" * min(len(view[0]) if view else 0, 140))
    print("Mutations accepted only when language-model perplexity falls. Ctrl-C to stop.")


def load_model(name, device, use_4bit):
    print(f"Loading {name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    if use_4bit and device.type == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    if device.type == "cuda" and not use_4bit:
        model = model.to(device)
    return model, tokenizer


@torch.no_grad()
def window_perplexity(model, tokenizer, text, device):
    """Average token perplexity of a short string. Returns inf on failure."""
    if not text.strip():
        return float("inf")
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if inputs["input_ids"].size(1) < 2:
            return float("inf")
        out = model(**inputs, labels=inputs["input_ids"])
        return math.exp(out.loss.item())
    except Exception:
        return float("inf")


def score_grid(model, tokenizer, grid, n_windows, window_len, device):
    """Mean perplexity over random windows drawn from the grid."""
    lines = grid_to_lines(grid)
    full = "\n".join(lines)
    if len(full) < window_len + 1:
        return window_perplexity(model, tokenizer, full, device)

    scores = []
    for _ in range(n_windows):
        start = random.randrange(0, max(1, len(full) - window_len))
        chunk = full[start:start + window_len]
        scores.append(window_perplexity(model, tokenizer, chunk, device))
    valid = [s for s in scores if math.isfinite(s)]
    return sum(valid) / len(valid) if valid else float("inf")


def main():
    p = argparse.ArgumentParser(description="ASCII Structure Dream (language model)")
    p.add_argument("--size", type=int, default=60, help="Grid width (height defaults to size//2 or size)")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                   help="HF model id. Small models are faster; 7B/8B work in 4-bit on 12GB")
    p.add_argument("--windows", type=int, default=6, help="Number of random windows scored per proposal")
    p.add_argument("--window-len", type=int, default=128, help="Characters per scoring window")
    p.add_argument("--steps", type=int, default=0, help="Max proposals (0 = forever)")
    p.add_argument("--mut", type=int, default=12, help="Random cell changes per proposal")
    p.add_argument("--display-every", type=int, default=1)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    w = args.size
    h = args.height if args.height is not None else (w if w <= 100 else max(30, w // 4))
    if args.size >= 500 and args.height is None:
        h = 500

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}", file=sys.stderr)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)

    model, tokenizer = load_model(args.model, device, use_4bit=not args.no_4bit)

    grid = random_grid(h, w)

    try:
        term_w = min(os.get_terminal_size().columns - 2, 140)
        term_h = min(os.get_terminal_size().lines - 6, 40)
    except Exception:
        term_w, term_h = 100, 30

    start = time.time()
    proposals = 0
    accepted = 0
    best = float("inf")

    try:
        while args.steps == 0 or proposals < args.steps:
            score = score_grid(model, tokenizer, grid, args.windows, args.window_len, device)
            best = min(best, score)

            if accepted % args.display_every == 0 or proposals == 0:
                display(grid, accepted, score, time.time() - start,
                        crop=(term_w, term_h) if (w > term_w or h > term_h) else None)

            cand = mutate(grid, args.mut)
            score_cand = score_grid(model, tokenizer, cand, args.windows, args.window_len, device)

            if score_cand < score:
                grid = cand
                accepted += 1

            proposals += 1
            if proposals % 5 == 0:
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"Accepted {accepted} / {proposals} proposals. Best perplexity seen: {best:.2f}")


if __name__ == "__main__":
    main()
