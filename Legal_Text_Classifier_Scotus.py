# ================================================================
# Full notebook — TDIDF + Logistic_regression Model Legal-BERT fine-tune on lex_glue scotus
# ================================================================

# 0) INSTALLS
# ================================================================
# !pip install -U transformers datasets accelerate scikit-learn seaborn matplotlib joblib

# 1) IMPORTS
# ================================================================
import os, re, random, time
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
)

# 1.a Mount Google Drive and set SAVE_DIR
# ================================================================
from google.colab import drive
drive.mount('/content/gdrive')   # will prompt for authorization

# Global save directory (user-specified)
SAVE_DIR = "/content/gdrive/My Drive/Colab Notebooks/Sem-7/DL/Project"
os.makedirs(SAVE_DIR, exist_ok=True)
# Create subfolders
os.makedirs(f"{SAVE_DIR}/models", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/figures", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/logs", exist_ok=True)

print("All outputs will be saved under:", SAVE_DIR)

# 2) Reproducibility
# ================================================================
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 3) Load dataset
# ================================================================
print("Loading lex_glue/scotus dataset...")
dataset = load_dataset("lex_glue", "scotus")
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]
print("Splits:", {k: len(v) for k, v in dataset.items()})

# 4) Label map & inspection (auto-remap handled if needed)
# ================================================================
original_label_map = {
  0:"Criminal Procedure",1:"Civil Rights",2:"First Amendment",3:"Due Process",
  4:"Privacy",5:"Attorneys",6:"Unions",7:"Economic Activity",8:"Judicial Power",
  9:"Federalism",10:"Interstate Relations",11:"Federal Taxation",12:"Miscellaneous",
  13:"Private Action"
}

print("\nSample train labels (first 20):", train_ds["label"][:20])
label_counts = Counter(train_ds["label"])
print("\nLabel counts (train):")
pprint(label_counts)

# Remap labels to compact index order used earlier in your code (if mapping necessary)
present = sorted(set(train_ds["label"]) | set(val_ds["label"]) | set(test_ds["label"]))
orig_to_compact = {orig:i for i,orig in enumerate(present)}
compact_to_orig = {i:orig for orig,i in orig_to_compact.items()}
label_map = {i: original_label_map[orig] for i,orig in compact_to_orig.items()}
NUM_LABELS = len(label_map)
print("\nNUM_LABELS:", NUM_LABELS)
print("label_map (id -> name):")
pprint(label_map)

def remap(ex):
    ex["label"] = orig_to_compact[int(ex["label"])]
    return ex

train_ds = train_ds.map(remap)
val_ds   = val_ds.map(remap)
test_ds  = test_ds.map(remap)

# 5) Cleaning function (light)
# ================================================================
def clean_text(t):
    if t is None: return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Prepare arrays for baseline
X_train_texts = [clean_text(t) for t in train_ds["text"]]
X_val_texts   = [clean_text(t) for t in val_ds["text"]]
X_test_texts  = [clean_text(t) for t in test_ds["text"]]

y_train = np.array(train_ds["label"])
y_val   = np.array(val_ds["label"])
y_test  = np.array(test_ds["label"])

# 6) TF-IDF baseline (original simple logistic) + analysis
# ================================================================
print("\nTraining TF-IDF + LogisticRegression baseline...")
vectorizer = TfidfVectorizer(max_features=45000, ngram_range=(1,3))
X_train = vectorizer.fit_transform(X_train_texts)
X_test  = vectorizer.transform(X_test_texts)

clf = LogisticRegression(max_iter=800, n_jobs=-1, C=3.0)
clf.fit(X_train, y_train)

base_preds = clf.predict(X_test)
base_micro = f1_score(y_test, base_preds, average="micro")
base_macro = f1_score(y_test, base_preds, average="macro")
print(f"\nBaseline TF-IDF results — Micro F1: {base_micro:.4f}  Macro F1: {base_macro:.4f}")

print("\nBaseline classification report:")
print(classification_report(y_test, base_preds, target_names=[label_map[i] for i in range(NUM_LABELS)], zero_division=0))

# Baseline confusion matrices
def plot_confusion_matrix(true, pred, labels, normalize=False, title="Confusion matrix", figsize=(12,10), cmap="Blues", savepath=None):
    cm = confusion_matrix(true, pred, labels=labels)
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=figsize)
    sns.heatmap(np.nan_to_num(cm), annot=True, fmt=".2f" if normalize else "d",
                xticklabels=[label_map[i] for i in labels],
                yticklabels=[label_map[i] for i in labels], cmap=cmap)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
        print("Saved confusion matrix to:", savepath)
    plt.show()

plot_confusion_matrix(y_test, base_preds, labels=list(range(NUM_LABELS)), normalize=False, title="Baseline Confusion Matrix (counts)", cmap="Blues", savepath=f"{SAVE_DIR}/figures/baseline_confmat_counts.png")
plot_confusion_matrix(y_test, base_preds, labels=list(range(NUM_LABELS)), normalize=True, title="Baseline Confusion Matrix (normalized by true row)", cmap="Blues", savepath=f"{SAVE_DIR}/figures/baseline_confmat_norm.png")

# Per-class metrics table for baseline
prec, rec, f1s, sup = precision_recall_fscore_support(y_test, base_preds, labels=list(range(NUM_LABELS)), zero_division=0)
df_baseline_metrics = pd.DataFrame({
    "label_id": list(range(NUM_LABELS)),
    "label": [label_map[i] for i in range(NUM_LABELS)],
    "precision": prec, "recall": rec, "f1": f1s, "support": sup
}).sort_values("f1", ascending=False)
display(df_baseline_metrics)

plt.figure(figsize=(10,6))
sns.barplot(data=df_baseline_metrics, x="f1", y="label")
plt.title("Baseline per-class F1 (TF-IDF + LR)")
plt.xlabel("F1 score")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/figures/baseline_per_class_f1.png", bbox_inches="tight")
plt.show()

# Save baseline artifacts to Google Drive
joblib.dump(clf, f"{SAVE_DIR}/models/logreg_scotus_tfidf.pkl")
joblib.dump(vectorizer, f"{SAVE_DIR}/models/tfidf_vectorizer_scotus.pkl")
df_baseline_metrics.to_csv(f"{SAVE_DIR}/models/baseline_per_class_metrics.csv", index=False)
print("Saved baseline models and metrics to", f"{SAVE_DIR}/models/")

# 7) LEGAL-BERT PREP: tokenizer, tokenization, dataset formatting
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("\nDevice:", DEVICE)

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tok(batch):
    texts = [clean_text(t) for t in batch["text"]]
    return tokenizer(texts, truncation=True, padding=False, max_length=512)

print("\nTokenizing datasets for Legal-BERT (this may take a minute)...")
train_tok = train_ds.map(tok, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(tok, batched=True, remove_columns=val_ds.column_names)
test_tok  = test_ds.map(tok, batched=True, remove_columns=test_ds.column_names)

# re-add labels and set format
train_tok = train_tok.add_column("labels", train_ds["label"])
val_tok   = val_tok.add_column("labels", val_ds["label"])
test_tok  = test_tok.add_column("labels", test_ds["label"])

train_tok.set_format("torch", ["input_ids", "attention_mask", "labels"])
val_tok.set_format("torch", ["input_ids", "attention_mask", "labels"])
test_tok.set_format("torch", ["input_ids", "attention_mask", "labels"])

# 8) Model, data_collator, metrics
# ================================================================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    micro = f1_score(labels, preds, average="micro")
    macro = f1_score(labels, preds, average="macro")
    return {"eval_micro_f1": micro, "eval_macro_f1": macro}

# 9) TrainingArguments — logging and per-epoch evaluation & progress
# ================================================================
training_args = TrainingArguments(
    output_dir=f"{SAVE_DIR}/legalbert_out",    # <- checkpoints and outputs will be saved here
    eval_strategy="epoch",            # evaluate at end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_ratio=0.1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.02,
    num_train_epochs=4,
    fp16=(torch.cuda.is_available()),
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    logging_strategy="steps",          # log steps so we can see progress per logging_steps
    logging_steps=50,
    save_total_limit=2,
    report_to="none",                 # disable external reporting by default
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 10) Train Legal-BERT (trainer prints progress per logging_steps and per epoch)
# ================================================================
print("\nStarting Legal-BERT fine-tuning (trainer will print progress):")
start_time = time.time()
train_result = trainer.train()
elapsed = time.time() - start_time
print(f"Training finished in {elapsed:.1f}s")

# Save final model + tokenizer to Google Drive
trainer.save_model(f"{SAVE_DIR}/models/legalbert_scotus")
tokenizer.save_pretrained(f"{SAVE_DIR}/models/legalbert_scotus")
print("Saved Legal-BERT model and tokenizer to", f"{SAVE_DIR}/models/legalbert_scotus")

# Also save trainer state and training result into logs
try:
    # trainer.state.log_history may be large; save JSON-ish
    import json
    with open(f"{SAVE_DIR}/logs/trainer_state_log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    # Save train_result metrics
    with open(f"{SAVE_DIR}/logs/train_result.json", "w") as f:
        json.dump(train_result.metrics, f)
    print("Saved trainer logs to:", f"{SAVE_DIR}/logs/")
except Exception as e:
    print("Failed to save trainer logs:", e)

# 11) Extract Trainer logs (loss & metrics per step/epoch) and plot progress
# ================================================================
log_history = trainer.state.log_history  # list of dicts
# Convert to DataFrame for easy plotting
df_logs = pd.DataFrame(log_history)

# Show logs to inspect (filtered)
print("\nTrainer log history (sample):")
display(df_logs.tail(20))

# Plot training loss over steps (if present)
if "loss" in df_logs.columns:
    df_loss = df_logs[["step","loss"]].dropna().drop_duplicates(subset="step").sort_values("step")
    plt.figure(figsize=(8,4))
    plt.plot(df_loss["step"], df_loss["loss"], marker="o")
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss by step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/figures/training_loss_steps.png", bbox_inches="tight")
    plt.show()

# Plot eval metrics per epoch (eval_macro_f1 / eval_micro_f1)
metrics_cols = [c for c in df_logs.columns if c.startswith("eval_")]
if metrics_cols:
    # group by epoch and show last metric logged per epoch
    df_eval = df_logs.dropna(subset=metrics_cols + ["epoch"]).groupby("epoch").last().reset_index()
    plt.figure(figsize=(8,4))
    if "eval_macro_f1" in df_eval.columns:
        plt.plot(df_eval["epoch"], df_eval["eval_macro_f1"], marker="o", label="eval_macro_f1")
    if "eval_micro_f1" in df_eval.columns:
        plt.plot(df_eval["epoch"], df_eval["eval_micro_f1"], marker="o", label="eval_micro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Evaluation metrics by epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/figures/eval_metrics_by_epoch.png", bbox_inches="tight")
    plt.show()
else:
    print("No eval metrics found in trainer logs to plot.")

# 12) Evaluate on test set & print extended metrics
# ================================================================
print("\nEvaluating final Legal-BERT on test set...")
preds_out = trainer.predict(test_tok)
logits = preds_out.predictions
bert_preds = np.argmax(logits, axis=-1)

bert_micro = f1_score(y_test, bert_preds, average="micro")
bert_macro = f1_score(y_test, bert_preds, average="macro")
print(f"Legal-BERT Micro-F1: {bert_micro:.4f}  Macro-F1: {bert_macro:.4f}")

print("\nClassification report (Legal-BERT):")
print(classification_report(y_test, bert_preds, target_names=[label_map[i] for i in range(NUM_LABELS)], zero_division=0))

# Confusion matrices for BERT
plot_confusion_matrix(y_test, bert_preds, labels=list(range(NUM_LABELS)), normalize=False, title="Legal-BERT Confusion Matrix (counts)", cmap="Greens", savepath=f"{SAVE_DIR}/figures/bert_confmat_counts.png")
plot_confusion_matrix(y_test, bert_preds, labels=list(range(NUM_LABELS)), normalize=True, title="Legal-BERT Confusion Matrix (normalized)", cmap="Greens", savepath=f"{SAVE_DIR}/figures/bert_confmat_norm.png")

# Per-class metrics for BERT
prec_b, rec_b, f1_b, sup_b = precision_recall_fscore_support(y_test, bert_preds, labels=list(range(NUM_LABELS)), zero_division=0)
df_metrics_bert = pd.DataFrame({
    "label_id": list(range(NUM_LABELS)),
    "label": [label_map[i] for i in range(NUM_LABELS)],
    "precision": prec_b, "recall": rec_b, "f1": f1_b, "support": sup_b
}).sort_values("f1", ascending=False)
display(df_metrics_bert)

plt.figure(figsize=(10,6))
sns.barplot(data=df_metrics_bert, x="f1", y="label")
plt.title("Legal-BERT per-class F1")
plt.xlabel("F1 score")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/figures/bert_per_class_f1.png", bbox_inches="tight")
plt.show()

# Precision & Recall table heatmap
pr_matrix = df_metrics_bert.set_index("label")[["precision","recall"]].T
plt.figure(figsize=(10,4))
sns.heatmap(pr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Precision & Recall (Legal-BERT) — rows: [precision, recall]")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/figures/bert_precision_recall_heatmap.png", bbox_inches="tight")
plt.show()

# 13) Compare baseline vs BERT per-class F1 side-by-side
# ================================================================
# Ensure same label order
baseline_f1_by_label = df_baseline_metrics.set_index("label")["f1"].reindex([label_map[i] for i in range(NUM_LABELS)]).values
bert_f1_by_label = df_metrics_bert.set_index("label").reindex([label_map[i] for i in range(NUM_LABELS)])["f1"].values

df_compare = pd.DataFrame({
    "label": [label_map[i] for i in range(NUM_LABELS)],
    "baseline_f1": baseline_f1_by_label,
    "bert_f1": bert_f1_by_label
}).sort_values("bert_f1", ascending=False)

plt.figure(figsize=(12,8))
df_melt = df_compare.melt(id_vars="label", value_vars=["baseline_f1","bert_f1"], var_name="model", value_name="f1")
sns.barplot(data=df_melt, x="f1", y="label", hue="model")
plt.title("Baseline vs Legal-BERT per-class F1")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/figures/compare_per_class_f1.png", bbox_inches="tight")
plt.show()

# 14) Predict helpers (baseline and BERT)
# ================================================================
def predict_and_describe_baseline(text, clf, vectorizer, label_map, clean_fn=clean_text):
    cleaned = clean_fn(text)
    X = vectorizer.transform([cleaned])
    pred_label = int(clf.predict(X)[0])
    prob = None
    if hasattr(clf, "predict_proba"):
        prob = float(clf.predict_proba(X)[0][pred_label])
    return pred_label, label_map.get(pred_label, "UNKNOWN"), prob

def predict_and_describe_bert(text, tokenizer, model, label_map, device="cpu", top_k=3):
    model.to(device).eval()
    with torch.no_grad():
        inputs = tokenizer(clean_text(text), truncation=True, max_length=512, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0).cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum()
        top_idx = probs.argsort()[-top_k:][::-1]
        return [(int(i), label_map[int(i)], float(probs[i])) for i in top_idx]

# Sample text from test set
sample_idx = 5
sample_text = test_ds["text"][sample_idx]
print("\n=== SAMPLE TEXT (truncated) ===")
print(sample_text[:800], "...\n")

print("Baseline prediction (TF-IDF + LR):")
b_id, b_name, b_prob = predict_and_describe_baseline(sample_text, clf, vectorizer, label_map)
print(f"Predicted id {b_id} → {b_name} (prob={b_prob:.4f})" if b_prob is not None else f"Predicted id {b_id} → {b_name}")

print("\nLegal-BERT top-3 predictions:")
for pid, pname, pscore in predict_and_describe_bert(sample_text, tokenizer, model, label_map, device=DEVICE, top_k=3):
    print(f"{pid} → {pname} (score={pscore:.4f})")

# 15) Save metrics & artifacts to Google Drive
# ================================================================
df_metrics_bert.to_csv(f"{SAVE_DIR}/models/bert_per_class_metrics.csv", index=False)
df_compare.to_csv(f"{SAVE_DIR}/models/compare_per_class_f1.csv", index=False)

# Optionally save a small run-summary text file
summary_txt = f"""
Run summary:
- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Model: {MODEL_NAME}
- Num labels: {NUM_LABELS}
- TF-IDF baseline micro/macro: {base_micro:.4f}/{base_macro:.4f}
- Legal-BERT micro/macro: {bert_micro:.4f}/{bert_macro:.4f}
- SAVE_DIR: {SAVE_DIR}
"""
with open(f"{SAVE_DIR}/logs/run_summary.txt", "w") as f:
    f.write(summary_txt)

print("\nSaved metrics CSVs and run summary under models/logs in", SAVE_DIR)

print("\nDone. Baseline and Legal-BERT trained, evaluated, visualized, and saved to Google Drive.")
