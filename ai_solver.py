
# # ai_solver.py
# import requests
# import os
# import librosa
# import numpy as np
# import joblib
# import tensorflow as tf
# import tensorflow_hub as hub
# import time

# FLASK_BASE = "http://127.0.0.1:5000"
# AI_TRIGGER = FLASK_BASE + "/ai_trigger_audio"
# AI_CHECK = FLASK_BASE + "/ai_check"
# TMP_AUDIO = "temp_audio_download.wav"
# SR = 16000
# MODEL_FILE = "audio_clf.joblib"

# pipeline = joblib.load(MODEL_FILE)
# yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# def get_yamnet_embedding(y):
#     waveform_tf = tf.convert_to_tensor(y, dtype=tf.float32)
#     _, embeddings, _ = yamnet(waveform_tf)
#     return np.mean(embeddings.numpy(), axis=0).astype(np.float32)

# def download_file(url, outpath):
#     r = requests.get(url, stream=True, timeout=15)
#     r.raise_for_status()
#     with open(outpath, "wb") as f:
#         for chunk in r.iter_content(8192):
#             if chunk:
#                 f.write(chunk)
#     return os.path.getsize(outpath)

# def main():


#     try:
#         resp = requests.post(AI_TRIGGER, timeout=4)
#         resp.raise_for_status()
#     except Exception as e:
#         print("ERROR: ai_trigger_audio failed:", e)
#         return

#     data = resp.json()
#     if not data.get("ok"):
#         print("Server returned error:", data)
#         return

#     audio_url = data["audio_url"]
#     audio_file = data["audio_file"]
#     ground_truth = data["ground_truth"]

#     print("Server says current captcha:", audio_file)
#     print("Ground truth:", ground_truth)
#     print("Audio URL:", audio_url)

#     # small wait so browser plays audio
#     time.sleep(0.4)


#     try:
#         size = download_file(audio_url, TMP_AUDIO)
#         print(f"Downloaded audio: {TMP_AUDIO} ({size} bytes)")
#     except Exception as e:
#         print("Failed to download:", e)
#         return


#     y, _ = librosa.load(TMP_AUDIO, sr=SR, mono=True)
#     if y.size == 0:
#         print("Audio empty!")
#         return

#     rms = np.sqrt(np.mean(y**2)) + 1e-9
#     y = y / rms * 0.1

#     emb = get_yamnet_embedding(y).reshape(1, -1)
#     probs = pipeline.predict_proba(emb)[0]

#     classes = pipeline.named_steps['clf'].classes_ \
#         if hasattr(pipeline, "named_steps") else pipeline.classes_

#     idx = probs.argsort()[::-1][:3]
#     top3 = [(classes[i], float(probs[i])) for i in idx]

#     predicted = top3[0][0]
#     confidence = top3[0][1]

#     print("Top-3 predictions:", top3)
#     print("Predicted:", predicted, "Confidence:", confidence)


#     payload = {
#         "predicted": predicted,
#         "ground_truth": ground_truth,
#         "confidence": f"{confidence:.4f}",
#         "top3": ";".join([f"{c}:{p:.4f}" for c,p in top3])
#     }

#     try:
#         r = requests.post(AI_CHECK, data=payload, timeout=5)
#         r.raise_for_status()
#         print("AI solved CAPTCHA, sent result to server.")
#     except Exception as e:
#         print("Failed to POST AI result:", e)

#     try:
#         os.remove(TMP_AUDIO)
#     except:
#         pass

# if __name__ == "__main__":
#     main()





# ai_solver.py
import requests
import os
import librosa
import numpy as np
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import time
import argparse

TMP_AUDIO = "temp_audio_download.wav"
SR = 16000
MODEL_FILE = "audio_clf.joblib"

# Load model + YAMNet once
pipeline = joblib.load(MODEL_FILE)
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")


def get_base_url():
    """
    Resolve the CAPTCHA base URL from:
    1. command-line argument --url / --base-url
    2. environment variable CAPTCHA_BASE_URL
    3. default http://127.0.0.1:5000
    """
    parser = argparse.ArgumentParser(description="AI attacker for audio CAPTCHA")
    parser.add_argument(
        "--url", "--base-url",
        dest="base_url",
        type=str,
        help="Base URL of the CAPTCHA system (e.g. http://127.0.0.1:5000)",
    )
    args = parser.parse_args()

    base_url = (
        args.base_url
        or os.environ.get("CAPTCHA_BASE_URL")
        or "http://127.0.0.1:5000"
    )
    # Normalise: no trailing slash
    return base_url.rstrip("/")


def get_yamnet_embedding(y):
    waveform_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform_tf)
    return np.mean(embeddings.numpy(), axis=0).astype(np.float32)


def download_file(url, outpath):
    r = requests.get(url, stream=True, timeout=15)
    r.raise_for_status()
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    return os.path.getsize(outpath)


def main():
    base_url = get_base_url()
    print(f"[INFO] Using CAPTCHA base URL: {base_url}")

    ai_trigger_url = base_url + "/ai_trigger_audio"
    ai_check_url = base_url + "/ai_check"

    # 1) Ask server to trigger audio + return current captcha metadata
    try:
        resp = requests.post(ai_trigger_url, timeout=4)
        resp.raise_for_status()
    except Exception as e:
        print("ERROR: ai_trigger_audio failed:", e)
        return

    data = resp.json()
    if not data.get("ok"):
        print("Server returned error:", data)
        return

    audio_url = data["audio_url"]
    audio_file = data["audio_file"]
    ground_truth = data["ground_truth"]

    print("Server says current captcha:", audio_file)
    print("Ground truth:", ground_truth)
    print("Audio URL:", audio_url)

    # small wait so browser plays audio
    time.sleep(0.4)

    # 2) Download audio file
    try:
        size = download_file(audio_url, TMP_AUDIO)
        print(f"Downloaded audio: {TMP_AUDIO} ({size} bytes)")
    except Exception as e:
        print("Failed to download:", e)
        return

    # 3) Preprocess + embedding
    y, _ = librosa.load(TMP_AUDIO, sr=SR, mono=True)
    if y.size == 0:
        print("Audio empty!")
        return

    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    y = y / rms * 0.1

    emb = get_yamnet_embedding(y).reshape(1, -1)

    # 4) Model prediction + top-3
    probs = pipeline.predict_proba(emb)[0]

    classes = (
        pipeline.named_steps["clf"].classes_
        if hasattr(pipeline, "named_steps")
        else pipeline.classes_
    )

    idx = probs.argsort()[::-1][:3]
    top3 = [(classes[i], float(probs[i])) for i in idx]

    predicted = top3[0][0]
    confidence = top3[0][1]

    print("Top-3 predictions:", top3)
    print("Predicted:", predicted, "Confidence:", confidence)

    # 5) Send result back to server
    payload = {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "confidence": f"{confidence:.4f}",
        "top3": ";".join([f"{c}:{p:.4f}" for c, p in top3]),
    }

    try:
        r = requests.post(ai_check_url, data=payload, timeout=5)
        r.raise_for_status()
        print("AI solved CAPTCHA, sent result to server.")
    except Exception as e:
        print("Failed to POST AI result:", e)

    # 6) Cleanup temp file
    try:
        os.remove(TMP_AUDIO)
    except Exception:
        pass


if __name__ == "__main__":
    main()




