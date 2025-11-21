import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

app = Flask(__name__, static_folder=str(ROOT / "static"), template_folder=str(ROOT / "templates"))
if CORS_AVAILABLE:
    CORS(app)

# Globals for loaded models
_baseline_clf = None
_baseline_vec = None
_bert_tokenizer = None
_bert_model = None
_label_names = None


def discover_label_names():
    # Try to load label names from CSVs if present
    bert_csv = MODELS_DIR / "bert_per_class_metrics.csv"
    comp = MODELS_DIR / "compare_per_class_f1.csv"
    if bert_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(bert_csv)
            if "label" in df.columns:
                return list(df["label"])
        except Exception:
            pass
    if comp.exists():
        try:
            import pandas as pd
            df = pd.read_csv(comp)
            if "label" in df.columns:
                return list(df["label"])
        except Exception:
            pass
    # fallback
    return [f"label_{i}" for i in range(14)]


def load_baseline():
    global _baseline_clf, _baseline_vec
    if _baseline_clf is not None and _baseline_vec is not None:
        return _baseline_clf, _baseline_vec
    clf_path = MODELS_DIR / "logreg_scotus_tfidf.pkl"
    vec_path = MODELS_DIR / "tfidf_vectorizer_scotus.pkl"
    if clf_path.exists() and vec_path.exists():
        _baseline_clf = joblib.load(clf_path)
        _baseline_vec = joblib.load(vec_path)
        return _baseline_clf, _baseline_vec
    return None, None


def load_bert():
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is not None and _bert_model is not None:
        return _bert_tokenizer, _bert_model
    bert_dir = MODELS_DIR / "legalbert_scotus"
    if bert_dir.exists():
        try:
            print(f"[DEBUG] Loading BERT from: {bert_dir}")
            _bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_dir), use_fast=True)
            _bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir), local_files_only=True)
            print("[DEBUG] BERT loaded successfully!")
            return _bert_tokenizer, _bert_model
        except Exception as e:
            print(f"[DEBUG] BERT loading error (attempt 1): {e}")
            try:
                # fallback without local_files_only
                _bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_dir), use_fast=True)
                _bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
                print("[DEBUG] BERT loaded successfully (fallback)!")
                return _bert_tokenizer, _bert_model
            except Exception as e2:
                print(f"[DEBUG] BERT loading error (attempt 2): {e2}")
                return None, None
    else:
        print(f"[DEBUG] BERT directory not found: {bert_dir}")
    return None, None


@app.route("/")
def index():
    return render_template("splash.html")


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    model = data.get("model", "bert")
    text = data.get("text", "")
    top_k = int(data.get("top_k", 3))
    use_gpu = bool(data.get("use_gpu", False))

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    if model == "baseline":
        clf, vec = load_baseline()
        if clf is None or vec is None:
            return jsonify({"error": "Baseline model or vectorizer not found on server"}), 500
        X = vec.transform([text])
        pred = int(clf.predict(X)[0])
        probs = None
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0].tolist()
        return jsonify({"model": "baseline", "pred_id": pred, "label": _label_names[pred] if pred < len(_label_names) else str(pred), "probs": probs})

    else:
        tokenizer, model_obj = load_bert()
        if tokenizer is None or model_obj is None:
            return jsonify({"error": "Legal-BERT model not found on server"}), 500
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        model_obj.to(device)
        model_obj.eval()
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model_obj(**inputs)
            logits = outputs.logits.cpu().numpy().squeeze(0)
            probs = np.exp(logits) / np.exp(logits).sum()
            top_idx = probs.argsort()[-top_k:][::-1]
            preds = []
            for i in top_idx:
                preds.append({"id": int(i), "label": _label_names[int(i)] if int(i) < len(_label_names) else str(i), "prob": float(probs[int(i)])})
            return jsonify({"model": "bert", "preds": preds})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    # list available CSVs and figure PNGs
    csvs = []
    figures = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.csv"):
            csvs.append(f.name)
    if FIGURES_DIR.exists():
        for f in FIGURES_DIR.glob("*.png"):
            figures.append(f.name)
    return jsonify({"csvs": csvs, "figures": figures})


@app.route("/figures/<path:fname>")
def serve_figure(fname):
    return send_from_directory(str(FIGURES_DIR), fname)


@app.route("/csvs/<path:fname>")
def serve_csv(fname):
    return send_from_directory(str(MODELS_DIR), fname)


if __name__ == "__main__":
    # load label names at startup
    _label_names = discover_label_names()
    # attempt to pre-load lightweight models (baseline)
    try:
        load_baseline()
    except Exception:
        pass
    app.run(host="0.0.0.0", port=8501, debug=True)
