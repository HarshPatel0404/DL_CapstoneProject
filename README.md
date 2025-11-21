# Legal Models Presentation App (Flask + HTML frontend)

This repository now contains a small Flask backend and a static HTML/CSS/JS frontend for presenting and testing the TF-IDF+LogisticRegression baseline and the Legal-BERT model saved in this project folder. The web UI allows pasting legal text, choosing a model, and receiving predictions. The Performance page displays available CSV metrics and saved figure PNGs from the training run.

Prerequisites
- Python 3.8+
- PowerShell (Windows) or other shell

Install dependencies (recommended in a virtualenv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Run the Flask app (from the repository root):

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open http://127.0.0.1:8501 in your browser.

What the app expects
- `models/` folder containing:
  - `legalbert_scotus/` — tokenizer & model files for Legal-BERT (exists in this repo)
  - `logreg_scotus_tfidf.pkl` and `tfidf_vectorizer_scotus.pkl` — baseline artifacts saved by the training script (if present)
  - CSVs: `baseline_per_class_metrics.csv`, `bert_per_class_metrics.csv`, `compare_per_class_f1.csv` (used to populate labels and tables)
- `figures/` folder containing PNGs saved by the training/eval script (confusion matrices, per-class f1, training/eval plots, etc.)

Notes
- For BERT inference the server will attempt to use CUDA if `use_gpu` is requested and a GPU is available. Ensure `torch` is installed with CUDA support for GPU inference.
- If model artifacts are missing, the server will report errors for those endpoints and continue serving the site (place missing files in `models/` or `figures/`).

If you'd like an alternative UI (Gradio or a packaged Docker/one-click launcher), tell me which and I'll add it.
