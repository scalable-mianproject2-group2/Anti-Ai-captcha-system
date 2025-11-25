import os
import random
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect

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
    # 第二个验证码：拼图滑块
    return render_template("slider.html")


@app.route("/slider/log", methods=["POST"])
def log_slider():
    data = request.get_json()
    print("Slider CAPTCHA log:", data)
    # 只返回 ok，前端可以在验证成功后跳转到 /audio
    return jsonify({"status": "ok"})


# ---------- Cat Litter CAPTCHA ----------
@app.route("/cat", methods=["GET"])
def cat_captcha():
    return render_template("cat.html")

@app.route("/cat/log", methods=["POST"])
def log_cat():
    data = request.get_json()
    print("Cat CAPTCHA log:", data)
    return jsonify({"status": "ok"})





# ---------- Audio CAPTCHA ----------
#AUDIO_FOLDER = r"C:\Users\bvija\Documents\TCD_Subjects\Scalable Computing\project_02\animals"

#zzw made changes to this path
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
        result_msg = "✅ Success!"
        # Serve a new random audio file
        selected_file = random.choice(audio_files)
        ground_truth = clean_label(selected_file)
        with last_served_lock:
            last_served["audio_file"] = selected_file
            last_served["ground_truth"] = ground_truth
    else:
        result_msg = "❌ Incorrect! Try again."
        with last_served_lock:
            selected_file = last_served["audio_file"]
            ground_truth = last_served["ground_truth"]

    return render_template("audio.html",
                           audio_file=selected_file,
                           ground_truth=ground_truth,
                           options=unique_labels,
                           result=result_msg,
                           logo_url=url_for('static', filename='images/audio.jpg'))

@app.route('/animals/<path:filename>')
def send_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == '__main__':
    print("Audio folder:", AUDIO_FOLDER)
    print("Number of audio files discovered:", len(audio_files))
    app.run(debug=True)
