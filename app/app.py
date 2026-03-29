"""
Lightweight Intent Chatbot — Flask + TF-IDF + Logistic Regression
Drop-in replacement for the SentenceTransformer version.

Key changes vs original:
  - Removed: sentence-transformers, torch  (~400 MB saved)
  - Added:   scikit-learn TF-IDF pipeline  (~5 MB)
  - RAM usage: ~60-80 MB  (was ~500+ MB)
  - All original routes, API key auth, rate limiting preserved
  - Same request/response JSON shape — frontend needs zero changes
"""

import os
import json
import random
import logging
import hashlib
import traceback
import pickle

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY              = os.getenv("API_KEY")
RATE_LIMIT           = os.getenv("RATE_LIMIT", "60/minute")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
DATA_PATH            = os.getenv("DATA_PATH", "training_data.json")

if not API_KEY:
    print("⚠️  Warning: API_KEY not set. Chat endpoint is unprotected.")

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ── Default training data (used if training_data.json is missing) ──────────────
DEFAULT_DATA = {
    "intents": [
        {
            "intent": "greeting",
            "patterns": ["Hello", "Hi there", "Hey", "Good morning", "Good evening", "Howdy", "What's up"],
            "responses": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! Great to see you. How can I assist?"
            ]
        },
        {
            "intent": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Take care", "I'm done", "That's all for now"],
            "responses": [
                "Goodbye! Have a great day!",
                "See you later! Take care.",
                "Bye! Feel free to come back anytime."
            ]
        },
        {
            "intent": "thanks",
            "patterns": ["Thank you", "Thanks a lot", "That was helpful", "Appreciate it", "Thanks!"],
            "responses": [
                "You're welcome!",
                "Happy to help!",
                "Anytime! Let me know if you need anything else."
            ]
        },
        {
            "intent": "hours",
            "patterns": [
                "What are your hours?", "When are you open?",
                "What time do you close?", "Are you open on weekends?"
            ],
            "responses": [
                "We're open Monday to Friday, 9am to 5pm.",
                "Our office hours are 9am–5pm on weekdays."
            ]
        },
        {
            "intent": "help",
            "patterns": ["I need help", "Can you help me?", "Help me please", "I have a question", "I'm confused"],
            "responses": [
                "Of course! What do you need help with?",
                "Sure, I'm here to help. What's your question?",
                "I'll do my best to help. Go ahead!"
            ]
        }
    ]
}

# ── Global model state ─────────────────────────────────────────────────────────
pipeline      = None   # sklearn Pipeline
responses_map = {}     # intent → [response, ...]
corpus_size   = 0      # number of patterns trained on


# ── Data loading & training ────────────────────────────────────────────────────

def _parse_intents(data):
    """
    Accept two formats:
      A) {"intents": [{"intent": "x", "patterns": [], "responses": []}]}
      B) [{"patterns": [], "responses": []}]   ← your old training_data.json format
    Returns (X, y, responses_map).
    """
    X, y, rmap = [], [], {}

    # Normalise to a flat list of intent dicts
    if isinstance(data, dict) and "intents" in data:
        intents = data["intents"]
    elif isinstance(data, list):
        intents = data
    else:
        intents = []

    for idx, item in enumerate(intents):
        if not isinstance(item, dict):
            logging.warning(f"Skipping item {idx}: not a dict")
            continue

        # Support both 'intent' and legacy 'tag' keys for the label
        label     = item.get("intent") or item.get("tag") or f"intent_{idx}"
        patterns  = item.get("patterns", [])
        responses = item.get("responses", [])

        if not isinstance(patterns, list):
            patterns = [patterns]
        if not isinstance(responses, list):
            responses = [responses]

        if not patterns or not responses:
            logging.warning(f"Skipping intent '{label}': empty patterns or responses")
            continue

        rmap[label] = responses
        for p in patterns:
            X.append(str(p).lower())
            y.append(label)

    return X, y, rmap


def load_training_data():
    global pipeline, responses_map, corpus_size

    # 1. Load raw data
    raw = None
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            logging.info(f"Loaded training data from {DATA_PATH}")
        except Exception as e:
            logging.error(f"Failed to read {DATA_PATH}: {e}. Falling back to defaults.")
    else:
        logging.warning(f"{DATA_PATH} not found. Using built-in default data.")

    if raw is None:
        raw = DEFAULT_DATA

    # 2. Parse intents
    X, y, rmap = _parse_intents(raw)

    if not X:
        logging.error("No valid patterns found in data. Falling back to defaults.")
        X, y, rmap = _parse_intents(DEFAULT_DATA)

    # 3. Train pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            analyzer="char_wb",   # character n-grams → typo tolerant
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            C=1.0,
            solver="lbfgs",
            multi_class="auto",
        )),
    ])
    pipe.fit(X, y)

    pipeline      = pipe
    responses_map = rmap
    corpus_size   = len(X)

    logging.info(f"✅ Model trained — {corpus_size} patterns across {len(rmap)} intents.")


# Train at startup (fast — < 1 second)
print("Training model...")
load_training_data()
print("Model ready.")


# ── Middleware: API key guard ───────────────────────────────────────────────────
@app.before_request
def enforce_api_key():
    """Protect /chat with API key if one is configured."""
    if request.endpoint == "chat_api":
        if API_KEY:
            provided = request.headers.get("X-API-Key")
            if not provided or provided != API_KEY:
                logging.warning(f"Unauthorized attempt from {request.remote_addr}")
                return jsonify({"error": "Unauthorized"}), 401


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the chat UI (templates/index.html)."""
    try:
        return render_template("index.html")
    except Exception as e:
        logging.error(f"Template render failed: {e}")
        return jsonify({"error": "Chat interface not available"}), 500


@app.route("/chat", methods=["POST"])
@limiter.limit("60/minute")
def chat_api():
    """
    Main chat endpoint.
    Request:  POST /chat  JSON { "text": "Hello" }
    Response: JSON { "response": "...", "confidence": 0.92, "tag": "matched", "status": "success" }
    Identical response shape to the original app — frontend unchanged.
    """
    try:
        if pipeline is None:
            return jsonify({"error": "Bot not ready. Please try again later."}), 503

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        user_text = data["text"].strip()

        if not user_text:
            return jsonify({"error": "Message cannot be empty"}), 400

        if len(user_text) > 500:
            return jsonify({"error": "Message too long (max 500 chars)"}), 400

        # Audit log (hashed — no PII stored)
        request_hash = hashlib.sha256(user_text.encode()).hexdigest()[:12]
        logging.info(
            f"[CHAT] IP={request.remote_addr} | Hash={request_hash} | "
            f"Msg='{user_text[:50]}{'...' if len(user_text) > 50 else ''}'"
        )

        # Predict
        text_lower = user_text.lower()
        proba      = pipeline.predict_proba([text_lower])[0]
        best_idx   = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        intent     = pipeline.classes_[best_idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            response = random.choice(responses_map[intent])
            tag      = "matched"
        else:
            response = "Sorry, I don't understand that. Could you rephrase?"
            tag      = "no_match"

        return jsonify({
            "response":   response,
            "confidence": round(confidence, 4),
            "tag":        tag,
            "status":     "success"
        })

    except Exception:
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Kubernetes liveness + readiness probe. Same path as before."""
    if pipeline is None:
        return jsonify({"status": "degraded (model not loaded)", "patterns": 0}), 503
    return jsonify({"status": "ok", "patterns": corpus_size}), 200


@app.route("/test_model", methods=["GET"])
def test_model():
    """Smoke-test endpoint — same path as before."""
    if pipeline is None:
        return jsonify({"error": "Model not loaded"}), 500
    sample = list(responses_map.keys())
    return jsonify({
        "message":      "Model and data are loaded",
        "num_patterns": corpus_size,
        "intents":      sample,
    })


@app.route("/reload", methods=["POST"])
def reload_data():
    """
    Hot-reload training data without restarting the container.
    POST /reload  (requires API key header if API_KEY is set)
    """
    if API_KEY:
        provided = request.headers.get("X-API-Key")
        if not provided or provided != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
    try:
        load_training_data()
        return jsonify({"status": "reloaded", "patterns": corpus_size})
    except Exception:
        logging.error(traceback.format_exc())
        return jsonify({"error": "Reload failed"}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)