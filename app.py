from flask import Flask, render_template, request, send_from_directory
import os, random
from pydub import AudioSegment

app = Flask(__name__)

# -------------------------------
# ANIMAL SOUND FILES
# -------------------------------
ANIMALS = {
    "dog": "dog.mp3",
    "cat": "cat.mp3",
    "cow": "cow.mp3",
    "lion": "lion.mp3",
    "sheep": "sheep.mp3",
    "donkey": "donkey.mp3",
    "monkey": "monkey.mp3",
    "rooster": "rooster.mp3",
    "horse": "horse.mp3"
}

ANIMAL_DIR = "static/animal"
NOISE_DIR = "static/noise"
OUTPUT_DIR = "static/audio"


# -------------------------------
# CHANGE AUDIO SPEED (0.85x – 1.25x)
# -------------------------------
def change_speed(audio, factor):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * factor)}
    ).set_frame_rate(audio.frame_rate)


# -------------------------------
# ADD BACKGROUND NOISE
# -------------------------------
def add_background_noise(audio):
    if not os.path.exists(NOISE_DIR):
        return audio

    noise_files = [
        f for f in os.listdir(NOISE_DIR)
        if f.lower().endswith(".wav") or f.lower().endswith(".mp3")
    ]
    if not noise_files:
        print("⚠ No noise files found")
        return audio

    selected = random.sample(noise_files, min(2, len(noise_files)))
    print("🎧 Using noise:", selected)

    combined = AudioSegment.silent(duration=len(audio))

    for nf in selected:
        noise_path = os.path.join(NOISE_DIR, nf)
        noise = AudioSegment.from_file(noise_path)

        # Loop noise to match audio duration
        while len(noise) < len(audio):
            noise *= 2

        noise = noise[:len(audio)]
        noise = noise + random.randint(4, 12)  # louder noise
        combined = combined.overlay(noise)

    return audio.overlay(combined)


# -------------------------------
# GENERATE ANIMAL CAPTCHA
# -------------------------------
def generate_animal_captcha():
    animal = random.choice(list(ANIMALS.keys()))
    original_file = os.path.join(ANIMAL_DIR, ANIMALS[animal])

    print(f"🐾 Selected animal: {animal}")

    audio = AudioSegment.from_file(original_file)

    # Random speed change (defense)
    audio = change_speed(audio, random.uniform(0.85, 1.25))

    # Add noise
    audio = add_background_noise(audio)

    # Save processed output
    output_filename = f"{animal}_{random.randint(1000,9999)}.mp3"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    audio.export(output_path, format="mp3")

    return output_filename, animal


# -------------------------------
# ROUTES
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    audio_file = None
    correct_animal = None
    result = None

    if request.method == "POST":

        # If verifying answer
        if "submitted_answer" in request.form:
            user = request.form.get("submitted_answer", "")
            correct = request.form.get("correct_animal", "")

            if user == correct:
                result = "Correct!"
            else:
                result = f"Wrong! It was {correct}"

        # Always generate new challenge
        audio_file, correct_animal = generate_animal_captcha()

    return render_template(
        "index.html",
        audio_file=audio_file,
        correct_animal=correct_animal,
        animals=list(ANIMALS.keys()),
        result=result
    )


# -------------------------------
# SERVE FILES
# -------------------------------
@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANIMAL_DIR, exist_ok=True)
    os.makedirs(NOISE_DIR, exist_ok=True)

    app.run(host="0.0.0.0", port=8080, debug=True)
