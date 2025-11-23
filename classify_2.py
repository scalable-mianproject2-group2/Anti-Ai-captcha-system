#!/usr/bin/env python3
"""
Train a small classifier on top of YAMNet embeddings with simple augmentations.

- Embedding: mean-pooled YAMNet embeddings (1024-d)
- Classifier: sklearn LogisticRegression (fast)
- Augmentations: pitch shift, time stretch, additive noise, small gain
- Outputs: results.csv, model_embeddings_labels.npz

Edit AUDIO_FOLDER below to your folder (raw string).
"""
import os, sys, random, csv
import numpy as np
from collections import defaultdict

# ========== EDIT THIS PATH ==========
AUDIO_FOLDER = r"D:\TCD Notes and assignments\Scalable computing\MainProject2\animal_classifier\animals"
# ====================================

# Imports (ensure you pip-install the required packages)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import librosa
    import soundfile as sf
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing package. Please run:\n  pip install tensorflow tensorflow-hub librosa scikit-learn numpy soundfile matplotlib")
    raise

# Constants
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
SR = 16000  # required by YAMNet
AUG_PER_FILE = 8   # number of augmented examples to synthesize per original file (including original)
RANDOM_SEED = 42
OUTPUT_CSV = "results.csv"
NPZ_OUT = "model_embeddings_labels.npz"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------- Augmentation helpers ----------------
def augment_waveform(y, sr):
    """
    Produce a list of augmented waveforms from input y (1D numpy float32).
    Each returned waveform has sample rate sr and is a numpy float32 1D array.
    We'll generate:
      - original
      - pitch shifts: +1, -1 semitone
      - time stretch: 0.9, 1.1
      - additive gaussian noise (small)
      - small gain changes
    We limit length by padding/trimming to original length.
    """
    outs = []
    # original (ensure float32)
    y0 = y.astype(np.float32)
    outs.append(y0)

    # pitch shifts (librosa.pitch_shift expects float, sr)
    try:
        outs.append(librosa.effects.pitch_shift(y0, sr, n_steps=+1))
        outs.append(librosa.effects.pitch_shift(y0, sr, n_steps=-1))
    except Exception:
        pass

    # time stretches
    try:
        ts1 = librosa.effects.time_stretch(y0, rate=0.9)
        ts2 = librosa.effects.time_stretch(y0, rate=1.1)
        # fix lengths
        def fixlen(a, target_len):
            if len(a) > target_len:
                return a[:target_len]
            elif len(a) < target_len:
                return np.pad(a, (0, target_len - len(a)), mode='constant')
            else:
                return a
        target_len = len(y0)
        outs.append(fixlen(ts1, target_len))
        outs.append(fixlen(ts2, target_len))
    except Exception:
        pass

    # add noise
    noise = np.random.randn(len(y0)).astype(np.float32)
    # scale noise to achieve modest SNR
    sig_rms = np.sqrt(np.mean(y0**2)) + 1e-9
    noise = noise * sig_rms * 0.05
    outs.append(y0 + noise)

    # small random gain
    gain = np.random.uniform(0.8, 1.2)
    outs.append(np.clip(y0 * gain, -1.0, 1.0))

    # Shuffle and trim/pad to original length, keep uniqueish
    final = []
    for a in outs:
        if len(a) > len(y0):
            a = a[:len(y0)]
        elif len(a) < len(y0):
            a = np.pad(a, (0, len(y0)-len(a)), mode='constant')
        final.append(a.astype(np.float32))
    # deduplicate (by hash of values) and shuffle
    uniq = []
    seen = set()
    for x in final:
        h = hash(x.tobytes())
        if h not in seen:
            seen.add(h)
            uniq.append(x)
    random.shuffle(uniq)
    # limit to AUG_PER_FILE
    return uniq[:AUG_PER_FILE]

# ---------------- Embedding extraction ----------------
print("Loading YAMNet model (TF-Hub) ... this may download on first run")
yamnet = hub.load(YAMNET_MODEL_HANDLE)

def yamnet_embedding(waveform):
    """
    waveform: 1-D numpy float32 (sr==SR)
    returns: 1024-d float32 embedding (mean pooled over frames)
    """
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spec = yamnet(waveform_tf)
    # embeddings shape: [num_frames, 1024]
    emb_np = embeddings.numpy()
    emb_mean = np.mean(emb_np, axis=0)
    return emb_mean.astype(np.float32)

# ---------------- Load files ----------------
exts = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(exts)])
if not files:
    print("No audio files found in folder:", AUDIO_FOLDER)
    sys.exit(1)

X = []
y = []
meta = []  # for reporting: (orig_filename, augmentation_index)

print(f"Found {len(files)} audio files. Creating augmentations and embeddings...")

for fname in files:
    fpath = os.path.join(AUDIO_FOLDER, fname)
    # load with librosa at SR
    try:
        y_orig, _ = librosa.load(fpath, sr=SR, mono=True)
    except Exception as e:
        print("Error loading", fpath, ":", e)
        continue
    # normalize loudness (simple RMS normalize)
    rms = np.sqrt(np.mean(y_orig**2)) + 1e-9
    if rms > 0:
        y_orig = y_orig / rms * 0.1  # scale to smaller RMS to avoid clipping in augmentation

    waves = augment_waveform(y_orig, SR)
    # label token from filename (clean)
    label = os.path.splitext(fname)[0].lower()
    # keep only alnum and underscore/dash
    # label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in label)
    label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in label).strip("_").lower()


    for i, w in enumerate(waves):
        try:
            emb = yamnet_embedding(w)
        except Exception as e:
            print("YAMNet embedding error for", fname, "aug", i, ":", e)
            continue
        X.append(emb)
        y.append(label)
        meta.append((fname, i))

X = np.vstack(X)
y = np.array(y)
print("Embeddings shape:", X.shape, "Labels:", np.unique(y))

# Save embeddings+labels for later
np.savez(NPZ_OUT, X=X, y=y, meta=meta)
print("Saved embeddings+labels to", NPZ_OUT)

# ---------------- Train classifier ----------------
# If there are very few original files per class, our synthetic augmentations create a usable training set.
# Use stratified split if possible.
if len(np.unique(y)) < 2:
    print("Need at least 2 classes to train. Found:", np.unique(y))
    sys.exit(1)

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

clf = LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nEvaluation on held-out augmented examples:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred, labels=np.unique(y)))

# ---------------- Predict original (non-augmented) files and print ground-truth vs predicted -----------
# We'll compute embeddings for the original (non-aug) files and predict them.
orig_X = []
orig_y = []
orig_names = []
for fname in files:
    fpath = os.path.join(AUDIO_FOLDER, fname)
    try:
        y_orig, _ = librosa.load(fpath, sr=SR, mono=True)
    except Exception as e:
        print("Error loading original file", fname, ":", e)
        continue
    # normalize same as earlier
    rms = np.sqrt(np.mean(y_orig**2)) + 1e-9
    y_orig = y_orig / rms * 0.1
    emb = yamnet_embedding(y_orig)
    orig_X.append(emb)
    label = os.path.splitext(fname)[0].lower()
    label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in label)
    orig_y.append(label)
    orig_names.append(fname)

orig_X = np.vstack(orig_X)
orig_y = np.array(orig_y)
orig_pred = clf.predict(orig_X)
orig_probs = clf.predict_proba(orig_X)

# Print table and write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["filename", "ground_truth", "predicted", "pred_prob", "top3"])
    for i, name in enumerate(orig_names):
        pred = orig_pred[i]
        prob = np.max(orig_probs[i])
        # top3
        top_idx = np.argsort(orig_probs[i])[::-1][:3]
        top3 = [(clf.classes_[j], float(orig_probs[i][j])) for j in top_idx]
        print(f"{name:20s}  GT={orig_y[i]:12s}  PRED={pred:12s}  prob={prob:.3f}  top3={top3}")
        writer.writerow([name, orig_y[i], pred, f"{prob:.4f}", ";".join([f"{t[0]}:{t[1]:.4f}" for t in top3])])

print("\nResults written to", OUTPUT_CSV)
print("If accuracy is still low, consider: more real examples per class, extracting more augmentations, or using PANNs (CNN14) embeddings / fine-tuning.")
