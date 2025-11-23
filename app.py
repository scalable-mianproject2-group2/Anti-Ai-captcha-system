# import os
# import random
# import threading
# from flask import Flask, render_template, request, jsonify, send_from_directory, session

# # ------------ CONFIG -------------
# AUDIO_FOLDER = r"D:\TCD Notes and assignments\Scalable computing\MainProject2\animal_classifier\animals"
# POLL_KEY = "latest_ai"   # key for storing latest ai result in memory
# # ---------------------------------

# app = Flask(__name__)

# # In-memory store for AI results (simple; resets when server restarts)
# # Structure: {"id": <int increment>, "predicted": "...", "ground_truth": "...", "result": "Success"/"Failure"}
# ai_store_lock = threading.Lock()
# ai_store = {"id": 0, "payload": None}

# def clean_label(fname):
#     base = os.path.splitext(os.path.basename(fname))[0]
#     label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in base)
#     return label.strip("_").lower()

# # load audio filenames and unique labels
# audio_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith((".mp3", ".wav"))])
# unique_labels = sorted(list(set([clean_label(f) for f in audio_files])))

# @app.route('/')
# def index():
#     # pick a random audio file for captcha
#     selected_file = random.choice(audio_files)
#     ground_truth = clean_label(selected_file)
#     return render_template("index.html",
#                            audio_file=selected_file,
#                            ground_truth=ground_truth,
#                            options=unique_labels)

# @app.route('/animals/<filename>')
# def send_audio(filename):
#     return send_from_directory(AUDIO_FOLDER, filename)

# @app.route('/check', methods=['POST'])
# def check():
#     # Human submission endpoint
#     selected = request.form.get('selected')
#     ground_truth = request.form.get('ground_truth')
#     result = "Success" if selected == ground_truth else "Failure"
#     return jsonify({"result": result, "predicted": selected, "ground_truth": ground_truth})

# @app.route('/ai_check', methods=['POST'])
# def ai_check():
#     """
#     AI posts here to register its prediction.
#     The server stores the payload in memory and returns acknowledgement.
#     Frontend polls /ai_result to fetch and display it.
#     """
#     predicted = request.form.get('predicted')
#     ground_truth = request.form.get('ground_truth')
#     result = "Success" if predicted == ground_truth else "Failure"
#     payload = {"predicted": predicted, "ground_truth": ground_truth, "result": result}
#     with ai_store_lock:
#         ai_store["id"] += 1
#         ai_store["payload"] = {"id": ai_store["id"], "data": payload}
#         saved_id = ai_store["id"]
#     return jsonify({"status": "ok", "id": saved_id})

# @app.route('/ai_result', methods=['GET'])
# def ai_result():
#     """
#     Frontend polls this endpoint to see if there's a new AI result.
#     The client should send a last_seen_id (int). If server has newer id, return it.
#     Once returned, the server clears the stored payload so it won't be delivered again.
#     """
#     try:
#         last_seen = int(request.args.get('last_id', 0))
#     except Exception:
#         last_seen = 0

#     with ai_store_lock:
#         payload = ai_store.get("payload")
#         # If there's a payload and it's newer than last_seen, return it and consume it
#         if payload and payload.get("id", 0) > last_seen:
#             out = {"has_new": True, "id": payload["id"], "data": payload["data"]}
#             # consume it (so next client/poll won't get it again)
#             ai_store["payload"] = None
#             return jsonify(out)
#         else:
#             return jsonify({"has_new": False})


# if __name__ == '__main__':
#     app.run(debug=True)






# app.py -- improved with /current_captcha and enriched AI payload storage
import os
import random
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

# ========== CONFIG (edit this) ==========
AUDIO_FOLDER = r"D:\TCD Notes and assignments\Scalable computing\MainProject2\animal_classifier\animals"
# Optional: local path to logo image you uploaded earlier.
# The developer environment stores uploaded files at /mnt/data/...; include that path if you'd like it passed to template.
LOGO_LOCAL_PATH = "/mnt/data/90e31aee-1f9d-4aad-8d1e-028b74330272.png"
# =========================================

app = Flask(__name__)

# In-memory store for AI results (simple; resets when server restarts)
ai_store_lock = threading.Lock()
ai_store = {"id": 0, "payload": None}

# store last-served captcha so AI can reliably ask the server which audio is current
last_served_lock = threading.Lock()
last_served = {"audio_file": None, "ground_truth": None}

def clean_label(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in base)
    return label.strip("_").lower()

# gather audio files and unique labels
exts = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
audio_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(exts)])
unique_labels = sorted(list(set([clean_label(f) for f in audio_files])))

@app.route('/')
def index():
    # pick a random audio file to display on this page
    if not audio_files:
        return "No audio files found in AUDIO_FOLDER.", 500

    selected_file = random.choice(audio_files)
    ground_truth = clean_label(selected_file)

    # record the last-served captcha (so ai_solver can query /current_captcha)
    with last_served_lock:
        last_served["audio_file"] = selected_file
        last_served["ground_truth"] = ground_truth

    # render template; pass an optional logo_url if you want to use server-side path
    return render_template("index.html",
                           audio_file=selected_file,
                           ground_truth=ground_truth,
                           options=unique_labels,
                           logo_url=LOGO_LOCAL_PATH)

@app.route('/animals/<path:filename>')
def send_audio(filename):
    # serve audio files (make sure AUDIO_FOLDER is correct)
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/check', methods=['POST'])
def check():
    # Human submission endpoint
    selected = request.form.get('selected', '')
    ground_truth = request.form.get('ground_truth', '')
    result = "Success" if selected == ground_truth else "Failure"
    return jsonify({"result": result, "predicted": selected, "ground_truth": ground_truth})

@app.route('/ai_check', methods=['POST'])
def ai_check():
    """
    AI posts here to register its prediction.
    Accepts form fields:
      - predicted
      - ground_truth
      - confidence (optional)
      - top3 (optional)
    The server stores the payload in memory for the frontend to poll and display.
    """
    predicted = request.form.get('predicted', '')
    ground_truth = request.form.get('ground_truth', '')
    confidence = request.form.get('confidence', '')
    top3 = request.form.get('top3', '')

    result = "Success" if predicted == ground_truth else "Failure"
    payload = {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "result": result,
        "confidence": confidence,
        "top3": top3
    }

    with ai_store_lock:
        ai_store["id"] += 1
        ai_store["payload"] = {"id": ai_store["id"], "data": payload}
        saved_id = ai_store["id"]

    return jsonify({"status": "ok", "id": saved_id})

@app.route('/ai_result', methods=['GET'])
def ai_result():
    """
    Frontend polls this endpoint to see if there's a new AI result.
    The client sends last_id; the server returns newer payload (if any) and consumes it.
    Returns JSON:
      - has_new: bool
      - id: int
      - data: payload dict (predicted, ground_truth, result, confidence, top3)
    """
    try:
        last_seen = int(request.args.get('last_id', 0))
    except Exception:
        last_seen = 0

    with ai_store_lock:
        payload = ai_store.get("payload")
        if payload and payload.get("id", 0) > last_seen:
            # consume payload so it's delivered only once
            out = {"has_new": True, "id": payload["id"], "data": payload["data"]}
            ai_store["payload"] = None
            return jsonify(out)
    return jsonify({"has_new": False})

@app.route('/current_captcha', methods=['GET'])
def current_captcha():
    """
    Returns the last-served captcha info so AI can download the exact file.
    Response JSON:
      - ok: bool
      - audio_file: filename
      - ground_truth: label
      - audio_url: relative URL to the file (use FLASK_BASE + audio_url to download)
    """
    with last_served_lock:
        if last_served.get("audio_file") is None:
            return jsonify({"ok": False, "message": "no captcha served yet"}), 404
        audio_file = last_served["audio_file"]
        ground_truth = last_served["ground_truth"]
    # build audio URL relative to server root
    audio_url = url_for('send_audio', filename=audio_file)
    return jsonify({"ok": True, "audio_file": audio_file, "ground_truth": ground_truth, "audio_url": audio_url})

if __name__ == '__main__':
    # helpful log of server config
    print("Audio folder:", AUDIO_FOLDER)
    print("Number of audio files discovered:", len(audio_files))
    app.run(debug=True)
