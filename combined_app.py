import os
import random
import threading
import json, time
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect, make_response

app = Flask(__name__)

# ---------- Arrow CAPTCHA ----------
@app.route("/", methods=["GET"])
def arrow_captcha():
    return render_template("arrow.html")

@app.route("/log", methods=["POST"])
def log_arrow():
    data = request.get_json()
    print("Arrow CAPTCHA log:", data)
    # after logging, redirect user to audio CAPTCHA
    return jsonify({"status": "ok"})

# ---------- Slider CAPTCHA ----------
@app.route("/slider", methods=["GET"])
def slider_captcha():
    
    return render_template("slider.html")


@app.route("/slider/log", methods=["POST"])
def log_slider():
    data = request.get_json()
    print("Slider CAPTCHA log:", data)
    
    return jsonify({"status": "ok"})

# ---------- Cat Litter CAPTCHA ----------
@app.route("/cat", methods=["GET"])
def cat_captcha():
    return render_template("cat.html")

@app.route("/cat/log", methods=["POST"])
def log_cat():
    data = request.get_json()
    print("Cat CAPTCHA log:", data)
    # 只返回 ok，前端可以在验证成功后跳转到 /audio
    return jsonify({"status": "ok"})

# ---------- Audio CAPTCHA ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = os.path.join(BASE_DIR, "animals")

audio_play_lock = threading.Lock()
audio_play_flag = {"play": False}

ai_store_lock = threading.Lock()
ai_store = {"id": 0, "payload": None}

last_served_lock = threading.Lock()
last_served = {"audio_file": None, "ground_truth": None}

def clean_label(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    base = base.split("_")[0]
    label = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in base)
    return label.strip("_").lower()

exts = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
audio_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(exts)])
unique_labels = sorted(list(set([clean_label(f) for f in audio_files])))

@app.route("/audio", methods=["GET"])
def audio_captcha():
    if not audio_files:
        return "No audio files found in AUDIO_FOLDER.", 500

    selected_file = random.choice(audio_files)
    ground_truth = clean_label(selected_file)

    with last_served_lock:
        last_served["audio_file"] = selected_file
        last_served["ground_truth"] = ground_truth

    return render_template("audio.html",
                           audio_file=selected_file,
                           ground_truth=ground_truth,
                           options=unique_labels,
                           result=None,
                           logo_url=url_for('static', filename='images/audio.jpg'))

@app.route('/check', methods=['POST'])
def check():
    selected = request.form.get('selected', '')
    ground_truth = request.form.get('ground_truth', '')

    # Determine result message
    if selected == ground_truth:
        result_msg = "Success!"
        # Serve a new random audio file
        selected_file = random.choice(audio_files)
        ground_truth = clean_label(selected_file)
        with last_served_lock:
            last_served["audio_file"] = selected_file
            last_served["ground_truth"] = ground_truth
    else:
        result_msg = "Incorrect! Try again."
        with last_served_lock:
            selected_file = last_served["audio_file"]
            ground_truth = last_served["ground_truth"]

    return render_template("audio.html",
                           audio_file=selected_file,
                           ground_truth=ground_truth,
                           options=unique_labels,
                           result=result_msg,
                           logo_url=url_for('static', filename='images/audio.jpg'))


@app.route('/ai_check', methods=['POST'])
def ai_check():
    # AI posts predicted result, confidence, top3
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
    return jsonify({"status": "ok", "id": ai_store["id"]})

@app.route('/ai_result', methods=['GET'])
def ai_result():
    # Frontend polls this; returns and consumes stored AI payload
    try:
        last_seen = int(request.args.get('last_id', 0))
    except Exception:
        last_seen = 0
    with ai_store_lock:
        payload = ai_store.get("payload")
        if payload and payload.get("id", 0) > last_seen:
            out = {"has_new": True, "id": payload["id"], "data": payload["data"]}
            ai_store["payload"] = None   # consume
            return jsonify(out)
    return jsonify({"has_new": False})

# endpoints to let AI request the browser to play audio
@app.route('/ai_trigger_audio', methods=['POST'])
def ai_trigger_audio():
    with last_served_lock:
        if not last_served.get("audio_file"):
            # No captcha yet in this process – create one
            selected_file = random.choice(audio_files)
            ground_truth = clean_label(selected_file)
            last_served["audio_file"] = selected_file
            last_served["ground_truth"] = ground_truth
            audio_file = selected_file
        else:
            audio_file = last_served["audio_file"]
            ground_truth = last_served["ground_truth"]

    with audio_play_lock:
        audio_play_flag["play"] = True
        audio_play_flag["id"] = time.time()

    audio_url = url_for('send_audio', filename=audio_file, _external=True)
    return jsonify({
        "ok": True,
        "audio_file": audio_file,
        "audio_url": audio_url,
        "id": audio_play_flag["id"],
        "ground_truth": ground_truth
    })


@app.route('/should_play_audio', methods=['GET'])
def should_play_audio():
    with audio_play_lock:
        play = audio_play_flag.get("play", False)
        aid = audio_play_flag.get("id", None)
        
        audio_play_flag["play"] = False
        audio_play_flag["id"] = None
    return jsonify({"play": bool(play), "id": aid})


@app.route('/current_captcha', methods=['GET'])
def current_captcha():
    with last_served_lock:
        if last_served.get("audio_file") is None:
            return jsonify({"ok": False, "message": "no captcha served yet"}), 404
        audio_file = last_served["audio_file"]
        ground_truth = last_served["ground_truth"]
    audio_url = url_for('send_audio', filename=audio_file)
    return jsonify({"ok": True, "audio_file": audio_file, "ground_truth": ground_truth, "audio_url": audio_url})


@app.route('/bot_signal', methods=['POST'])
def bot_signal():
    """Receive a compact JSON summary from the client and return suspicious score."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "bad json"}), 400

    
    score = 0.0
    
    t_submit = data.get('time_to_submit_ms', 999999)
    mouse_count = data.get('mouse_samples_count', 0)
    audio_delay = data.get('audio_play_delay_ms', -1)
    focus_changes = data.get('focus_changes', 0)
    avg_speed = data.get('mouse_avg_speed', 0.0)

    if t_submit < 300:          
        score += 2.0
    if mouse_count < 3:         
        score += 1.0
    if 0 <= audio_delay < 100:  
        score += 1.0
    if focus_changes > 2:
        score += 0.8
    if avg_speed < 0.01:
        score += 0.6


    if data.get("via_ai_flag"):
        suspect = True
    else:
        suspect = score >= 1.5

    
    try:
        out = {
            "ts": time.time(),
            "score": score,
            "suspect": bool(suspect),
            "payload": data
        }
        with open("bot_signals.ndjson", "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
    except Exception as e:
        # logging failure shouldn't break the flow
        print("Failed to write bot_signals:", e)

   
    return jsonify({"suspect": bool(suspect), "score": float(score)})


@app.route('/animals/<path:filename>')
def send_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == '__main__':
    print("Audio folder:", AUDIO_FOLDER)
    print("Number of audio files discovered:", len(audio_files))
    app.run(debug=True)
