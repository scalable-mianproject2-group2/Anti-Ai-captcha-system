
# ai_solver.py
import requests
import os
import librosa
import numpy as np
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import time

# ---------- CONFIG ----------
FLASK_BASE = "http://127.0.0.1:5000"
CURRENT_CAPTCHA = FLASK_BASE + "/current_captcha"
AI_CHECK = FLASK_BASE + "/ai_check"
AI_TRIGGER = FLASK_BASE + "/ai_trigger_audio"   # endpoint to tell browser to play audio
TMP_AUDIO = "temp_audio_download.wav"
SR = 16000
MODEL_FILE = "audio_clf.joblib"

# ---------- load model & yamnet ----------
pipeline = joblib.load(MODEL_FILE)
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def get_yamnet_embedding(y):
    waveform_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform_tf)
    emb_mean = np.mean(embeddings.numpy(), axis=0)
    return emb_mean.astype(np.float32)

def download_file(url, outpath):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    total = 0
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    return total

def main():
    # Ask server what captcha is currently shown
    try:
        resp = requests.get(CURRENT_CAPTCHA, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print("Failed to get current captcha:", e)
        return

    info = resp.json()
    if not info.get("ok"):
        print("Server did not return current captcha.")
        return

    audio_file = info.get("audio_file")
    ground_truth = info.get("ground_truth")
    audio_url = FLASK_BASE.rstrip('/') + info.get("audio_url", "")

    print("Server says current captcha:", audio_file)
    print("Ground truth (server):", ground_truth)
    print("Audio URL:", audio_url)

    # --- NEW: notify browser to play the audio for this captcha ---
    # This triggers the frontend polling (should_play_audio) and the browser will execute audio.play()
    try:
        # best-effort notify: ignore failures (we still proceed to download/process)
        requests.post(AI_TRIGGER, timeout=2)
    except Exception as e:
        # not fatal; continue
        print("Warning: failed to post ai_trigger_audio:", e)

    # tiny pause to give the browser a chance to pick up the flag and start playback
    time.sleep(0.3)

    # 2) download the exact file (as before)
    try:
        size = download_file(audio_url, TMP_AUDIO)
        print(f"Downloaded audio: {TMP_AUDIO} ({size} bytes)")
    except Exception as e:
        print("Failed to download audio:", e)
        return

    # 3) load and preprocess (same as training/testing)
    try:
        y, _ = librosa.load(TMP_AUDIO, sr=SR, mono=True)
    except Exception as e:
        print("Failed to load audio with librosa:", e)
        return

    if y.size == 0:
        print("Downloaded audio is empty!")
        return

    # normalize RMS as used in training
    rms = np.sqrt(np.mean(y**2)) + 1e-9
    y = y / rms * 0.1

    # 4) compute embedding and predict
    emb = get_yamnet_embedding(y).reshape(1, -1)
    probs = pipeline.predict_proba(emb)[0]
    classes = pipeline.named_steps['clf'].classes_ if hasattr(pipeline, "named_steps") else pipeline.classes_
    top_idx = probs.argsort()[::-1][:5]
    top3 = [(classes[i], float(probs[i])) for i in top_idx[:3]]
    predicted = classes[top_idx[0]]
    confidence = float(probs[top_idx[0]])

    print("Top-3 predictions:", top3)
    print("Predicted:", predicted, "Confidence:", confidence)

    # 5) post the AI result (with confidence and top3) to server so frontend can show them
    post_payload = {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "confidence": f"{confidence:.4f}",
        "top3": ";".join([f"{c}:{p:.4f}" for c,p in top3])
    }
    try:
        p = requests.post(AI_CHECK, data=post_payload, timeout=5)
        p.raise_for_status()
        print("AI solved CAPTCHA: posted result to server. Predicted:", predicted, "Ground truth:", ground_truth)
    except Exception as e:
        print("Failed to POST AI result:", e)

    # cleanup (optional)
    try:
        os.remove(TMP_AUDIO)
    except Exception:
        pass

if __name__ == "__main__":
    main()
