#!/usr/bin/env python3
"""
generate_dataset.py

Generates MP3 audio CAPTCHAs using symbols from symbols.txt.
Each captcha is a RANDOM sequence of length 4-6 (inclusive).
Saves files to dataset/ and writes dataset/manifest.csv with columns: path,label

Behavior:
 - symbols.txt may be one-per-line or space-separated or a single line like "0123AB..."
 - TTS pronounces characters separated by spaces (e.g. "A 7 K 3") to encourage clear enunciation.
 - Filenames are safe: <LABEL>_<index>.mp3

Usage:
    python generate_dataset.py --outdir dataset --n 2000
"""

import argparse
import os
import random
import csv
import sys
from pathlib import Path
from gtts import gTTS

def load_symbols(symbols_file: Path):
    txt = symbols_file.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"symbols file {symbols_file} is empty")
    # support one-char-per-line or space separated or continuous string
    parts = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # if line contains spaces, split
        if " " in line:
            parts.extend(line.split())
        else:
            # either multiple chars e.g. "ABC123" or a single char
            # treat each character separately
            parts.extend(list(line))
    # remove duplicates and whitespace characters, preserve order
    seen = set()
    symbols = []
    for ch in parts:
        ch = ch.strip()
        if not ch:
            continue
        if ch not in seen:
            seen.add(ch)
            symbols.append(ch)
    return symbols

def generate_random_label(symbols, min_len=4, max_len=6):
    length = random.randint(min_len, max_len)
    return "".join(random.choice(symbols) for _ in range(length))

def safe_filename(label: str, idx: int):
    # Replace any path-unfriendly chars just in case, but symbols are expected to be alnum.
    safe_label = "".join(c if c.isalnum() else "_" for c in label)
    return f"{safe_label}_{idx:06d}.mp3"

def main(outdir, symbols_file, n, min_len, max_len):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    symbols_file = Path(symbols_file)
    if not symbols_file.exists():
        print(f"symbols file not found: {symbols_file}", file=sys.stderr)
        sys.exit(1)

    symbols = load_symbols(symbols_file)
    print(f"Loaded {len(symbols)} symbols. Example symbols: {symbols[:20]}")
    manifest_path = outdir / "manifest.csv"
    rows = []
    for i in range(n):
        label = generate_random_label(symbols, min_len=min_len, max_len=max_len)
        # Create TTS text with spaces between characters to force char-by-char pronunciation
        tts_text = " ".join(list(label))
        # Use gTTS to generate mp3
        tts = gTTS(text=tts_text, lang="en")
        fname = safe_filename(label, i)
        outpath = outdir / fname
        tts.save(str(outpath))
        rows.append({"path": str(outpath.resolve()), "label": label})
        if (i+1) % 100 == 0 or i == n-1:
            print(f"Generated {i+1}/{n}: {label} -> {fname}")

    # write manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {n} files to {outdir} and manifest to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="dataset")
    parser.add_argument("--symbols", type=str, default="symbols.txt")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=6)
    args = parser.parse_args()
    main(args.outdir, args.symbols, args.n, args.min_len, args.max_len)
