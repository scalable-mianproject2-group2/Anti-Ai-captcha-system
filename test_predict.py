import sys, librosa, numpy as np, joblib, tensorflow as tf, tensorflow_hub as hub

SR = 16000
model = joblib.load("audio_clf.joblib")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def get_emb(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    rms = (np.sqrt(np.mean(y**2)) + 1e-9)
    y = y / rms * 0.1
    waveform_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform_tf)
    emb = embeddings.numpy().mean(axis=0)
    return emb.reshape(1,-1)

if __name__ == "__main__":
    path = sys.argv[1]
    emb = get_emb(path)
    probs = model.predict_proba(emb)[0]
    classes = model.named_steps['clf'].classes_ if hasattr(model, 'named_steps') else model.classes_
    top_idx = probs.argsort()[::-1][:5]
    print("Top predictions:")
    for i in top_idx:
        print(classes[i], probs[i])
