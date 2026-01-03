from flask import Flask, request, jsonify
import joblib
import re
from urllib.parse import urlparse
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/phishing_rf.pkl")

# ---------- Feature helpers (SAME AS TRAINING) ----------

def has_ip(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0

def count_special(url):
    return len(re.findall(r'[!@#$%^&*(),=?":{}|<>]', url))

def suspicious_words(url):
    words = [
        'login', 'verify', 'update', 'secure',
        'account', 'bank', 'confirm', 'signin',
        'paypal', 'ebay'
    ]
    return sum(word in url.lower() for word in words)

def extract(url):
    p = urlparse(url)

    features = {
        "furl_length": len(url),
        "hostname_length": len(p.netloc),
        "has_ip": has_ip(url),
        "count_dots": url.count('.'),
        "count_hyphen": url.count('-'),
        "count_at": url.count('@'),
        "count_question": url.count('?'),
        "count_slash": url.count('/'),
        "count_digits": sum(c.isdigit() for c in url),
        "count_special": count_special(url),
        "https": 1 if p.scheme == "https" else 0,
        "suspicious_words": suspicious_words(url)
    }

    return pd.DataFrame([features])

# ---------- Routes ----------

@app.route("/")
def home():
    return "Phishing URL Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "URL not provided"}), 400

    url = data["url"]

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    features = extract(url)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    result = "Phishing" if prediction == 1 else "Legitimate"

    return jsonify({
        "url": url,
        "prediction": result,
        "confidence": round(float(probability), 4)
    })

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
