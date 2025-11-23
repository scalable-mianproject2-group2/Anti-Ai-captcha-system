#!/usr/bin/env python3
"""
classify.py

Simple YAMNet ground-truth vs predicted script (Windows-friendly).

Embedded folder path so you don't need to pass it on the command line.
"""

import os
import sys
import tempfile
import csv
import numpy as np

# ========== EDIT THIS PATH: put your folder here (raw string to avoid backslash escapes) ==========
AUDIO_FOLDER = r"D:\TCD Notes and assignments\Scalable computing\MainProject2\animal_classifier\animals"
# ================================================================================================

# Try imports; if missing, inform the user
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import soundfile as sf
    from pydub import AudioSegment
except Exception as e:
    print("Missing packages. Please install requirements with:")
    print("  pip install tensorflow tensorflow-hub soundfile pydub numpy")
    raise

# Constants
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
TARGET_SR = 16000  # YAMNet expects 16 kHz mono
OUTPUT_CSV = "results.csv"

# Helpers
def export_to_temp_wav(audiosegment):
    """
    Export a pydub.AudioSegment to a temporary WAV file and return its path.
    Use delete=False on Windows to avoid permission errors when reopening.
    Caller is responsible for deleting the file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmpname = tmp.name
    tmp.close()  # close the handle so pydub can write to it on Windows
    audiosegment.export(tmpname, format="wav")
    return tmpname

def load_audio_to_numpy(path):
    """
    Convert audio file (mp3/wav/...) to 16 kHz mono float32 numpy array [-1..1].
    Uses pydub to decode and soundfile to read normalized floats.
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)
    tmp_wav = None
    try:
        tmp_wav = export_to_temp_wav(audio)
        wav, sr = sf.read(tmp_wav, dtype="float32")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
    if sr != TARGET_SR:
        raise RuntimeError(f"Unexpected sample rate: {sr} (expected {TARGET_SR})")
    # ensure 1-D
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav

def load_class_map():
    """Download/parse the YAMNet class CSV to a list of display names."""
    import urllib.request, csv
    try:
        with urllib.request.urlopen(YAMNET_CLASS_MAP_URL) as resp:
            text = resp.read().decode("utf-8").splitlines()
    except Exception:
        # offline fallback:
        return [f"class_{i}" for i in range(521)]
    reader = csv.reader(text)
    names = []
    for row in reader:
        if len(row) >= 3:
            names.append(row[2])
    return names

def filename_to_label(fname):
    """Extract ground-truth label token from filename (cleaned, lowercase)."""
    base = os.path.splitext(os.path.basename(fname))[0]
    # replace spaces with underscore, replace non-alnum/_/- with underscore
    label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in base)
    return label.lower()

def map_prediction_to_labels(topk_classnames, label_set):
    """
    Map AudioSet top-k class names to one of our filename labels by case-insensitive substring match.
    Returns matched label or None.
    """
    for cname in topk_classnames:
        c = cname.lower()
        for lab in label_set:
            if lab in c:
                return lab
    return None

def top_k_from_scores(scores_np, class_names, k=3):
    idx = np.argsort(scores_np)[::-1][:k]
    return [(class_names[i], float(scores_np[i])) for i in idx]

def main():
    folder = AUDIO_FOLDER
    if not os.path.isdir(folder):
        print("Audio folder not found:", folder)
        return

    exts = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)])
    if not files:
        print("No audio files found in folder. Supported:", exts)
        return

    print("Loading YAMNet model (this may download the model on first run)...")
    yamnet = hub.load(YAMNET_MODEL_HANDLE)
    class_names = load_class_map()

    labels = [filename_to_label(f) for f in files]
    label_set = set(labels)

    results = []

    for path, gt_label in zip(files, labels):
        print(f"\nProcessing: {os.path.basename(path)}  (ground-truth token: '{gt_label}')")
        try:
            wav = load_audio_to_numpy(path)
        except Exception as e:
            print("  ERROR converting file:", e)
            continue

        waveform = tf.convert_to_tensor(wav, dtype=tf.float32)
        # model returns (scores, embeddings, spectrogram)
        scores, embeddings, spectrogram = yamnet(waveform)
        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)

        top3 = top_k_from_scores(mean_scores, class_names, k=3)
        top3_names = [t[0] for t in top3]

        mapped = map_prediction_to_labels(top3_names, label_set)
        if mapped is None:
            # fallback: derive a clean token from top-1 classname
            mapped = top3_names[0].lower().replace(" ", "_")

        top1_name, top1_score = top3[0]

        print(f"  Predicted (mapped to your labels): {mapped}")
        print(f"  Top-1 AudioSet class: '{top1_name}'  score={top1_score:.4f}")
        print("  Top-3 AudioSet classes:")
        for n, s in top3:
            print(f"    {n:40s} {s:.4f}")

        results.append({
            "filename": os.path.basename(path),
            "ground_truth": gt_label,
            "predicted_mapped": mapped,
            "top1_class": top1_name,
            "top1_score": f"{top1_score:.4f}"
        })

    # Print summary
    if results:
        print("\n\nSummary (filename | ground_truth | predicted_mapped | top1_class | score):")
        print("{:35s} {:20s} {:20s} {:40s} {:>8s}".format("filename", "ground_truth", "predicted_mapped", "top1_class", "score"))
        print("-"*125)
        for r in results:
            print("{:35s} {:20s} {:20s} {:40s} {:>8s}".format(r['filename'], r['ground_truth'], r['predicted_mapped'], r['top1_class'][:40], r['top1_score']))
    else:
        print("\nNo successful conversions / predictions were made.")

    # Save CSV
    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["filename","ground_truth","predicted_mapped","top1_class","top1_score"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults written to {OUTPUT_CSV}")
    except Exception as e:
        print("Error writing CSV:", e)

if __name__ == "__main__":
    main()
