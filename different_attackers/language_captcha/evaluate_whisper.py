#!/usr/bin/env python3
"""
evaluate_whisper.py

Reads dataset/manifest.csv, runs whisper.medium on each file, compares prediction to ground truth.

Outputs:
 - Per-sample printout (GT vs Pred)
 - Summary: total, exact-match accuracy, character error rate (CER)

Usage:
    python evaluate_whisper.py --manifest dataset/manifest.csv --model medium
"""

import argparse
import csv
import re
import sys
from pathlib import Path
import whisper
import torch

def normalize_pred(text: str):
    # remove whitespace and non-alphanumeric, uppercase
    return re.sub(r"[^A-Za-z0-9]", "", text).upper().strip()

def cer(reference: str, hypothesis: str):
    # simple Levenshtein distance / CER = edits / len(reference)
    r = reference
    h = hypothesis
    if len(r) == 0:
        return 1.0 if len(h) > 0 else 0.0
    # DP
    m = len(r)
    n = len(h)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # deletion
                           dp[i][j-1] + 1,      # insertion
                           dp[i-1][j-1] + cost) # substitution
    edits = dp[m][n]
    return edits / m

def main(manifest_path: Path, model_name: str):
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    # load manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Loaded {len(rows)} samples from manifest")

    # load whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    total = 0
    exact_correct = 0
    cer_sum = 0.0

    for r in rows:
        path = r["path"]
        gt = r["label"].upper().strip()
        total += 1
        res = model.transcribe(path)
        pred_raw = res.get("text", "")
        pred = normalize_pred(pred_raw)
        sample_cer = cer(gt, pred)
        cer_sum += sample_cer
        if pred == gt:
            exact_correct += 1
            match_flag = "OK"
        else:
            match_flag = "ERR"
        print(f"{match_flag} | GT: {gt} | Pred: {pred} | CER: {sample_cer:.3f} | file: {Path(path).name}")

    accuracy = exact_correct / total if total > 0 else 0.0
    avg_cer = cer_sum / total if total > 0 else 0.0
    print("\n=========================")
    print(f"Total: {total}")
    print(f"Exact-match accuracy: {exact_correct}/{total} = {accuracy:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    print("=========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="dataset/manifest.csv")
    parser.add_argument("--model", type=str, default="medium", help="whisper model: tiny/base/small/medium/large")
    args = parser.parse_args()
    main(Path(args.manifest), args.model)
