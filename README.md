<!--
	Polished README for the DL_CapstoneProject (Legal Models Presentation App)
	Purpose: provide a GitHub-style, presentable project README that explains the
	project, how to run it (Flask + Streamlit), model artifacts, UI components,
	development workflow, and troubleshooting notes.
-->

# Legal Document Classification (LegalBERT + Baseline)

[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()

One-stop demo & presentation repo for legal document classification. This project contains:

- A Flask-backed HTML frontend (presentation-ready UI) for serving model predictions and visualizations.
- A standalone Streamlit demo app for quick interactive exploration.
- Trained model artifacts (Legal-BERT packaged folder and TF-IDF baseline), checkpoints, logs, and saved evaluation figures.

This README gives clear, copy-paste instructions for installing, running, and extending the project, and documents the UI components, model locations, and developer workflow.

---

Table of contents

- About
- Features
- UI components (what to look for)
- Quickstart (install & run)
	- Flask UI (default)
	- Streamlit demo
- Models & data
- Inference (how the app runs models)
- Training & checkpoints
- File / folder layout
- Managing large files
- Development, testing & contribution
- Troubleshooting
- License & citations

---

About

Legal document classification using a fine-tuned Legal-BERT model and a TF-IDF + Logistic Regression baseline. The repo is intended for demos, evaluation, and quick local inference; it is not optimized as a production-ready inference service (but can be adapted for production).

Features

- Two user-facing UIs:
	- Flask HTML frontend (rich, animated UI with templates, splash, and visualizations).
	- Streamlit demo app (single-file interactive demo).
- Local model inference using Hugging Face Transformers and scikit-learn baseline.
- Saved evaluation figures and CSV metrics to inspect model performance.
- Training checkpoints (full trainer state) for resuming or reproducing experiments.

UI components (extracted from `templates/`, `static/` and docs)

- Splash screen: animated splash in `templates/splash.html` (auto-redirect to main page).
- Navigation & header: sticky header with brand, page navigation, and theme toggle.
- Model selection cards: interactive cards to choose between `Legal-BERT` and `TF-IDF + LogisticRegression`.
- Input area: large textarea for pasting legal text, GPU toggle, and Top-K selector (for BERT results).
- Results panel: animated result cards showing top predictions and confidence bars.
- Visualizations gallery: `templates/visualizations.html` displays training/eval PNGs from `figures/` with modal full-screen view and keyboard navigation.
- Performance tables: show CSV metrics (per-class metrics, compare F1) loaded from `models/*.csv`.
- Theme (Dark/Light): CSS variables and toggle (in `static/css/style.css`).
- Client-side JS: `static/js/app.js` implements interactive behaviors (modal, keyboard shortcuts, theme memory in localStorage).

Quickstart (install & run)

Prerequisites

- Windows or Linux with Python 3.8+
- Optional: CUDA-capable GPU + appropriate drivers and CUDA toolkit for GPU inference with PyTorch

Recommended: create a virtual environment before installing dependencies.

PowerShell (Windows) — minimal steps

```powershell
cd "d:\HARSH_DL\deep learning\deep learning"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you only want the UI and baseline (no BERT GPU inference):

```powershell
pip install Flask scikit-learn pandas joblib Pillow streamlit
```

Running the Flask demo (default)

```powershell
python .\app.py
```

Open the URL shown (default: `http://127.0.0.1:8501`).

Running the Streamlit demo (alternative)

```powershell
streamlit run .\streamlit_app.py
```

Streamlit is self-contained and runs independently; it expects the same `models/` and `figures/` folders to be next to the script.

Models & data (Models added in Releases)

- Packaged model for inference: `models/legalbert_scotus/` — contains `config.json`, `tokenizer.json`, `model.safetensors` (or HF-compatible model weights).
- Baseline artifacts: `models/logreg_scotus_tfidf.pkl` and `models/tfidf_vectorizer_scotus.pkl` (scikit-learn artifacts used by baseline).
- Metrics CSVs: `models/baseline_per_class_metrics.csv`, `models/bert_per_class_metrics.csv`, and `models/compare_per_class_f1.csv` used to populate performance tables and label names.
- Figures: `figures/*.png` for confusion matrices, F1 plots, training curves, etc.

Inference details (code pointers)

- Streamlit app (`streamlit_app.py`):
	- `load_baseline()` loads baseline pickles.
	- `load_bert()` loads the HF tokenizer & model from `models/legalbert_scotus`.
	- `predict_baseline()` and `predict_bert()` run inference and return top-k probabilities.

- Flask UI (`app.py`):
	- Serves templates and static files.
	- Routes load models from `models/` and return HTML pages or JSON responses depending on implementation.

Example: load HF model in Python (used by the apps)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./models/legalbert_scotus", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("./models/legalbert_scotus")

# Move model to device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

Training & checkpoints

- Full training checkpoints are under `legalbert_out/checkpoint-*/` and include optimizer, scheduler, scaler, and trainer state — useful for resuming training.
- Use the Trainer API or your existing training scripts to resume from these checkpoints.

Managing large files

- Model checkpoints and `*.safetensors` are large. Best practices:
	- Do not commit binary checkpoint files to standard Git history.
	- Use Git LFS for large tracked files if you must keep them in the repo.
	- Prefer uploading to cloud storage (S3, GDrive) and include download links in the README.
	- Use `upload_large_assets.sh` (present in the repo) to help move large files to external storage.

File / folder layout (short)

```
deep learning/
├─ app.py                     # Flask presentation app
├─ streamlit_app.py           # Streamlit demo (standalone)
├─ requirements.txt           # Project dependencies
├─ models/                    # Packaged models + CSV metrics
├─ legalbert_out/             # Full training checkpoints (large)
├─ figures/                   # Evaluation images (PNGs)
├─ logs/                      # run summaries and trainer logs
├─ static/                    # CSS, JS
└─ templates/                 # HTML templates (index, splash, visualizations)
```

Development & contribution

- Branching workflow: create a feature branch, add tests/screenshots in PR, and request review.
- Keep model weights out of commits; include small unit tests for core logic.

Troubleshooting

- Missing tokenizer/model: ensure `models/legalbert_scotus/` contains `config.json`, tokenizer files, and a model weight file.
- Model load errors: try `local_files_only=True` in `from_pretrained()` if you do not want HF to fetch remote files.
- GPU not detected: verify GPU drivers and install PyTorch with the correct CUDA version. Check `torch.cuda.is_available()`.


Acknowledgements & credits

- Legal-BERT (and the authors)
- Hugging Face `transformers`
- scikit-learn, Streamlit, Flask
