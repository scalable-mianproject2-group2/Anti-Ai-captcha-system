# app.py
from flask import Flask, jsonify, send_file, request, render_template
from pathlib import Path
import csv
import random
import re
import torch
import whisper
import threading

# ---------- Config ----------
DATASET_DIR = Path("dataset")
MANIFEST = DATASET_DIR / "manifest.csv"
MODEL_NAME = "medium"
# ----------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")

# load manifest into memory
if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")
with open(MANIFEST, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    MANIFEST_ROWS = list(reader)

# simple running accuracy counters (in-memory)
lock = threading.Lock()
session_total = 0
session_correct = 0

def normalize_text(s: str):
    return re.sub(r"[^A-Za-z0-9]", "", s).upper().strip()

# choose device and load model
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device_str)
model = whisper.load_model(MODEL_NAME, device=device_str)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/challenge", methods=["GET"])
def challenge():
    # return one random manifest entry (id will be index in manifest)
    entry = random.choice(MANIFEST_ROWS)
    filename = Path(entry["path"]).name
    return jsonify({
        "id": filename,
        "label": entry["label"]  # frontend shouldn't trust it for challenge UX, but it's useful for display/testing
    })

@app.route("/audio/<path:filename>", methods=["GET"])
def audio(filename):
    path = DATASET_DIR / filename
    if not path.exists():
        return jsonify({"error": "file not found"}), 404
    return send_file(str(path), mimetype="audio/mpeg")

@app.route("/predict", methods=["POST"])
def predict():
    global session_total, session_correct
    data = request.get_json() or {}
    if "id" not in data:
        return jsonify({"error": "missing id"}), 400
    filename = data["id"]
    file_path = DATASET_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "file not found"}), 404

    # get ground truth from manifest (best-effort)
    gt = None
    for row in MANIFEST_ROWS:
        if Path(row["path"]).name == filename:
            gt = row["label"].upper().strip()
            break

    # transcribe using whisper
    # whisper's transcribe accepts a path; it will use ffmpeg internally (ensure ffmpeg installed)
    try:
        result = model.transcribe(str(file_path))
    except Exception as e:
        return jsonify({"error": f"transcription failed: {e}"}), 500

    pred_raw = result.get("text", "")
    pred = normalize_text(pred_raw)

    is_correct = (gt is not None and pred == gt)

    # update running counters
    with lock:
        session_total += 1
        if is_correct:
            session_correct += 1
        running_accuracy = session_correct / session_total if session_total > 0 else 0.0

    return jsonify({
        "prediction": pred,
        "prediction_raw": pred_raw,
        "ground_truth": gt,
        "correct": is_correct,
        "running_accuracy": running_accuracy
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
