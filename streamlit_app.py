import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ML libraries
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Paths (assume this file lives at project root or run from project root)
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

# Helper: discover available artifacts
available = {
    "tfidf": (MODELS_DIR / "logreg_scotus_tfidf.pkl", MODELS_DIR / "tfidf_vectorizer_scotus.pkl"),
    "bert_dir": MODELS_DIR / "legalbert_scotus",
    "csvs": {
        "baseline": MODELS_DIR / "baseline_per_class_metrics.csv",
        "bert": MODELS_DIR / "bert_per_class_metrics.csv",
        "compare": MODELS_DIR / "compare_per_class_f1.csv",
    }
}

# Load label names (prefer CSVs if available)
def load_label_names():
    # Try bert CSV first
    bert_csv = available["csvs"]["bert"]
    if bert_csv.exists():
        try:
            df = pd.read_csv(bert_csv)
            if "label" in df.columns:
                return list(df["label"])  # assumes ordered by id
        except Exception:
            pass
    # Try compare csv
    comp = available["csvs"]["compare"]
    if comp.exists():
        try:
            df = pd.read_csv(comp)
            if "label" in df.columns:
                return list(df["label"])
        except Exception:
            pass
    # Fallback: generic placeholder labels
    return [f"label_{i}" for i in range(14)]

LABEL_NAMES = load_label_names()
NUM_LABELS = len(LABEL_NAMES)

st.set_page_config(page_title="Legal Models demo", layout="wide")

st.title("Legal models — demo & presentation")
st.write("Load models from the local `models/` folder, paste legal text, and get predictions. Use the Performance page to inspect metrics and figures.")

# Sidebar controls
st.sidebar.header("Model & device")
model_choice = st.sidebar.selectbox("Select model:", ["Legal-BERT", "TF-IDF + LogisticRegression"], index=0)
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)

# Tabs: Predict | Performance
tab1, tab2 = st.tabs(["Predict", "Performance"])

# Prediction helpers
@st.cache_resource
def load_baseline():
    clf_path, vec_path = available["tfidf"]
    if clf_path.exists() and vec_path.exists():
        clf = joblib.load(clf_path)
        vec = joblib.load(vec_path)
        return clf, vec
    return None, None

@st.cache_resource
def load_bert():
    bert_path = available["bert_dir"]
    if bert_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(bert_path), use_fast=True)
        # Use safe device mapping
        try:
            model = AutoModelForSequenceClassification.from_pretrained(str(bert_path))
        except Exception:
            # Attempt with local files (safetensors/model)
            model = AutoModelForSequenceClassification.from_pretrained(str(bert_path), local_files_only=True)
        return tokenizer, model
    return None, None


def predict_baseline(text, clf, vec):
    if clf is None or vec is None:
        return None
    x = vec.transform([text])
    pred = int(clf.predict(x)[0])
    probs = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x)[0]
    return pred, probs


def predict_bert(text, tokenizer, model, device="cpu", top_k=5):
    if tokenizer is None or model is None:
        return None
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy().squeeze(0)
        probs = np.exp(logits) / np.exp(logits).sum()
        top_idx = probs.argsort()[-top_k:][::-1]
        return list(zip(top_idx.tolist(), probs[top_idx].tolist()))

# Predict tab
with tab1:
    st.header("Make a prediction")
    sample_text = """
Add or paste a legal text (court opinion excerpt or case summary) here. Keep it under the model maximum length if possible (512 tokens for the demo)."""
    text = st.text_area("Input text:", value=sample_text, height=300)

    col1, col2 = st.columns([1,1])
    with col1:
        top_k = st.number_input("Top K results (for BERT):", min_value=1, max_value=NUM_LABELS, value=3)
    with col2:
        run = st.button("Predict")

    if run:
        if model_choice == "TF-IDF + LogisticRegression":
            clf, vec = load_baseline()
            if clf is None:
                st.error("Baseline model or vectorizer not found in models/ (expected files: logreg_scotus_tfidf.pkl and tfidf_vectorizer_scotus.pkl)")
            else:
                res = predict_baseline(text, clf, vec)
                if res is None:
                    st.error("Failed to run baseline prediction")
                else:
                    pred_id, probs = res
                    label = LABEL_NAMES[pred_id] if pred_id < len(LABEL_NAMES) else f"label_{pred_id}"
                    st.success(f"Predicted id {pred_id} → {label}")
                    if probs is not None:
                        st.write("Top probabilities:")
                        top = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:top_k]
                        df = pd.DataFrame([{"id":int(i), "label":(LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i)), "prob":float(p)} for i,p in top])
                        st.table(df)
        else:
            tokenizer, model = load_bert()
            if tokenizer is None or model is None:
                st.error("Legal-BERT model not found under models/legalbert_scotus. Make sure the folder exists and contains the tokenizer/config/model files.")
            else:
                device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
                res = predict_bert(text, tokenizer, model, device=device, top_k=top_k)
                if res is None:
                    st.error("Failed to run BERT prediction")
                else:
                    st.success("Top predictions (id → label) with probabilities:")
                    df = pd.DataFrame([{"id":int(i), "label":(LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i)), "prob":float(p)} for i,p in res])
                    st.table(df)

# Performance tab
with tab2:
    st.header("Performance & artifacts")
    st.write("Below are available metrics CSVs and saved figure images from the training/evaluation runs. Missing files will be skipped.")

    # Show CSVs if present
    for key, p in available["csvs"].items():
        if p.exists():
            st.subheader(f"{key.capitalize()} metrics")
            try:
                df = pd.read_csv(p)
                st.dataframe(df)
            except Exception as e:
                st.write(f"Failed to read {p.name}: {e}")

    # Show figures (list images in figures/)
    if FIGURES_DIR.exists():
        imgs = sorted(FIGURES_DIR.glob("*.png"))
        if imgs:
            st.subheader("Saved Figures")
            for img in imgs:
                st.markdown(f"**{img.name}**")
                try:
                    im = Image.open(img)
                    st.image(im, use_column_width=True)
                except Exception as e:
                    st.write(f"Could not open image {img.name}: {e}")
        else:
            st.write("No images found in figures/ folder.")
    else:
        st.write("No figures/ folder found in project root.")

    st.markdown("---")
    st.write("If you want to show other artifacts (saved model checkpoints, trainer logs), place them under `models/` or `figures/` and reload this app.")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built for local presentation. Ensure the `models/` and `figures/` folders sit next to this script.")
