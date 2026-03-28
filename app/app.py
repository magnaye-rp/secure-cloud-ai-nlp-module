import json
import random
import logging
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentence_transformers import SentenceTransformer, util
import torch
import traceback
from dotenv import load_dotenv
import hashlib

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # tighten in prod

# ====================== CIA CONFIG ======================
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("❌ Set API_KEY in .env for Confidentiality!")

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[os.getenv("RATE_LIMIT", "60/minute")]
)

logging.basicConfig(level=logging.INFO)
model = SentenceTransformer('all-MiniLM-L6-v2')   # lightweight & fast

# ====================== CIA MIDDLEWARE ======================
@app.before_request
def enforce_api_key():
    """CONFIDENTIALITY: Require API key on every protected route"""
    if request.endpoint in ['predict', 'embed']:
        provided_key = request.headers.get('X-API-Key')
        if not provided_key or provided_key != API_KEY:
            logging.warning(f"Unauthorized attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized – Invalid or missing X-API-Key"}), 401

# ====================== ROUTES ======================
@app.route("/")
def home():
    """AVAILABILITY: Simple health check"""
    return jsonify({"status": "healthy", "message": "🔒 CIA-secured NLP service running (ngrok dev)"})

@app.route("/predict", methods=["POST"])
@limiter.limit("60/minute")   # AVAILABILITY + CONFIDENTIALITY
def predict():
    """Example semantic similarity endpoint"""
    try:
        data = request.get_json()
        if not data or "text1" not in data or "text2" not in data:
            return jsonify({"error": "Missing text1 or text2"}), 400

        text1 = data["text1"].strip()
        text2 = data["text2"].strip()

        # INTEGRITY: basic input validation
        if len(text1) > 500 or len(text2) > 500:
            return jsonify({"error": "Text too long (max 500 chars)"}), 400

        # INTEGRITY: log hashed request for audit
        request_hash = hashlib.sha256(f"{text1}{text2}".encode()).hexdigest()[:12]
        logging.info(f"[INTEGRITY] Request hash={request_hash} | IP={request.remote_addr}")

        # Your SentenceTransformers logic
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        cosine_score = util.cos_sim(emb1, emb2)[0][0].item()

        return jsonify({
            "similarity_score": round(cosine_score, 4),
            "message": "CIA triad applied ✓"
        })
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal error"}), 500

@app.route("/health", methods=["GET"])
def health():
    """AVAILABILITY: Kubernetes readiness/liveness probe"""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
