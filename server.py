from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import datetime

app = Flask(__name__)
CORS(app)

# Serve the frontend
@app.route("/")
def home():
    return render_template("index.html")

# Logging endpoint
@app.route("/log", methods=["POST"])
def log():
    data = request.json
    data["timestamp"] = str(datetime.datetime.now())
    print("User action:", data)  # prints in terminal
    return jsonify({"status": "logged"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
