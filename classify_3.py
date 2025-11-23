#!/usr/bin/env python3
"""
classify_grouped.py

Group-aware training on YAMNet embeddings with augmentations.

- All augmentations of the same original file are kept together (GroupShuffleSplit).
- Saves results.csv and audio_clf.joblib (scaler+classifier).
- Edit AUDIO_FOLDER below to point to your folder (raw string recommended).
"""

import os
import sys
import random
import csv
import numpy as np

# ========== EDIT THIS PATH ==========
AUDIO_FOLDER = r"D:\TCD Notes and assignments\Scalable computing\MainProject2\animal_classifier\animals"
# ====================================

# Hyperparams
SR = 16000
AUG_PER_FILE = 8           # number of augmented examples per original file (including original)
TEST_SIZE = 0.2
RANDOM_SEED = 42
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
OUTPUT_CSV = "results.csv"
MODEL_OUT = "audio_clf.joblib"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Imports (require pip installs listed above)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import librosa
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
except Exception as e:
    print("Missing package. Run:\n  pip install tensorflow tensorflow-hub librosa scikit-learn numpy soundfile joblib matplotlib")
    raise

# ------------------------- augmentation helpers -------------------------
def augment_waveform(y, sr, max_aug=AUG_PER_FILE):
    """Return up to max_aug waveforms derived from y (numpy 1D)."""
    outs = []
    y0 = y.astype(np.float32)
    outs.append(y0)

    # pitch shift ±1 semitone
    try:
        outs.append(librosa.effects.pitch_shift(y0, sr, n_steps=+1))
        outs.append(librosa.effects.pitch_shift(y0, sr, n_steps=-1))
    except Exception:
        pass

    # time stretches 0.9 & 1.1 (fix length)
    try:
        def fixlen(a, L):
            if len(a) > L: return a[:L]
            if len(a) < L: return np.pad(a, (0, L - len(a)), mode='constant')
            return a
        ts1 = librosa.effects.time_stretch(y0, 0.9)
        ts2 = librosa.effects.time_stretch(y0, 1.1)
        L = len(y0)
        outs.append(fixlen(ts1, L))
        outs.append(fixlen(ts2, L))
    except Exception:
        pass

    # additive noise
    noise = np.random.randn(len(y0)).astype(np.float32)
    sig_rms = np.sqrt(np.mean(y0**2)) + 1e-9
    noise = noise * sig_rms * 0.05
    outs.append(y0 + noise)

    # small random gain
    gain = np.random.uniform(0.85, 1.15)
    outs.append(np.clip(y0 * gain, -1.0, 1.0))

    # ensure lengths and uniqueness
    final = []
    seen = set()
    for a in outs:
        if len(a) > len(y0):
            a = a[:len(y0)]
        elif len(a) < len(y0):
            a = np.pad(a, (0, len(y0)-len(a)), mode='constant')
        h = hash(a.tobytes())
        if h not in seen:
            seen.add(h)
            final.append(a.astype(np.float32))
    random.shuffle(final)
    return final[:max_aug]

# ------------------------- YAMNet embedding extraction -------------------------
print("Loading YAMNet model (TF Hub)... this may download on first run.")
yamnet = hub.load(YAMNET_HANDLE)

def yamnet_embedding(waveform):
    """
    waveform: 1D numpy float32 (sr==SR)
    returns: 1024-d mean-pooled embedding (numpy float32)
    """
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spec = yamnet(waveform_tf)
    emb_np = embeddings.numpy()
    emb_mean = np.mean(emb_np, axis=0)
    return emb_mean.astype(np.float32)

# ------------------------- utility: clean labels -------------------------
def clean_label_from_filename(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    # keep alnum, underscore, dash; replace others with underscore
    label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in base)
    # strip leading/trailing underscores and lower-case
    return label.strip("_").lower()

# ------------------------- gather files -------------------------
exts = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(exts)])
if not files:
    print("No audio files found in:", AUDIO_FOLDER)
    sys.exit(1)

print(f"Found {len(files)} audio files. Building augmented embeddings...")

X_list = []
y_list = []
groups = []   # group id per sample (original filename)
meta = []     # (orig_filename, aug_index)

for fname in files:
    fpath = os.path.join(AUDIO_FOLDER, fname)
    try:
        y_raw, _ = librosa.load(fpath, sr=SR, mono=True)
    except Exception as e:
        print("Error loading", fpath, ":", e)
        continue

    # RMS normalize to a modest level (avoids huge volume differences)
    rms = np.sqrt(np.mean(y_raw**2)) + 1e-9
    if rms > 0:
        y_raw = y_raw / rms * 0.1

    waves = augment_waveform(y_raw, SR, max_aug=AUG_PER_FILE)
    label = clean_label_from_filename(fname)
    for i, w in enumerate(waves):
        try:
            emb = yamnet_embedding(w)
        except Exception as e:
            print("YAMNet embedding error for", fname, "aug", i, ":", e)
            continue
        X_list.append(emb)
        y_list.append(label)
        groups.append(fname)   # group by original filename (not label) to avoid leakage
        meta.append((fname, i))

X = np.vstack(X_list)
y = np.array(y_list)
groups = np.array(groups)
print("Created embeddings:", X.shape, "Unique labels:", np.unique(y))

# ------------------------- group-aware split -------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]
groups_test = groups[test_idx]

print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])
print("Train groups (unique originals):", len(np.unique(groups_train)),
      "Test groups (unique originals):", len(np.unique(groups_test)))

# ------------------------- train classifier (pipeline) -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs'))
])
pipeline.fit(X_train, y_train)

# evaluate on grouped test set
y_pred = pipeline.predict(X_test)
print("\nEvaluation on group-held-out test set:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
labels_sorted = np.unique(y)
print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

# ------------------------- predict on original, non-augmented files -------------------------
orig_names = []
orig_labels = []
orig_embeddings = []
for fname in files:
    fpath = os.path.join(AUDIO_FOLDER, fname)
    try:
        y_orig, _ = librosa.load(fpath, sr=SR, mono=True)
    except Exception as e:
        print("Error loading original file", fname, ":", e)
        continue
    rms = np.sqrt(np.mean(y_orig**2)) + 1e-9
    y_orig = y_orig / rms * 0.1
    emb = yamnet_embedding(y_orig)
    orig_embeddings.append(emb)
    lab = clean_label_from_filename(fname)
    orig_labels.append(lab)
    orig_names.append(fname)

orig_X = np.vstack(orig_embeddings)
orig_pred = pipeline.predict(orig_X)
orig_probs = pipeline.predict_proba(orig_X)

# Print and save results
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["filename", "ground_truth", "predicted", "pred_prob", "top3"])
    for i, name in enumerate(orig_names):
        pred = orig_pred[i]
        prob = float(np.max(orig_probs[i]))
        top_idx = np.argsort(orig_probs[i])[::-1][:3]
        top3 = [(pipeline.named_steps["clf"].classes_[j], float(orig_probs[i][j])) for j in top_idx]
        print(f"{name:25s}  GT={orig_labels[i]:12s}  PRED={pred:12s}  prob={prob:.3f}  top3={top3}")
        writer.writerow([name, orig_labels[i], pred, f"{prob:.4f}", ";".join([f"{t[0]}:{t[1]:.4f}" for t in top3])])

print("\nResults written to", OUTPUT_CSV)

# ------------------------- save pipeline -------------------------
try:
    joblib.dump(pipeline, MODEL_OUT)
    print("Saved model pipeline to", MODEL_OUT)
except Exception as e:
    print("Failed to save model:", e)

print("\nDone.")
