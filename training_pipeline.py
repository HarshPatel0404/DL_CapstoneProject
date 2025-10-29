# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
import re
from datasets import load_dataset

# Preprocessing steps
def preprocess_text(text):
    """
    Preprocess the text by removing non-textual elements, lowercasing, tokenizing, and lemmatizing.
    """
    # Remove non-textual elements (e.g., metadata, references)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercasing
    text = text.lower()
    
    # Tokenization using Legal-BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    
    # For simplicity, we skip stopword removal and lemmatization here.
    return " ".join(tokens)

# Encode labels
def encode_labels(labels):
    """
    Encode the multi-label concepts using MultiLabelBinarizer.
    """
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(labels)
    return encoded_labels, mlb

# Main pipeline
def prepare_pipeline():
    """
    Prepare the training pipeline for the Eur-Lex 57K dataset.
    """
    # Load dataset from Hugging Face
    dataset = load_dataset("coastalcph/lex_glue", "eurlex")
    
    # Preprocess text
    dataset = dataset.map(lambda x: {"text": preprocess_text(x["text"])})
    
    # Encode labels
    dataset = dataset.map(lambda x: {"concepts": x["concepts"].split(",")})
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform([x["concepts"] for x in dataset["train"]])
    
    # Add encoded labels to the dataset
    dataset = dataset.map(
        lambda x, idx: {**x, **dict(zip(mlb.classes_, encoded_labels[idx]))},
        with_indices=True
    )
    
    # Save splits
    dataset["train"].to_csv("train.csv", index=False)
    dataset["validation"].to_csv("val.csv", index=False)
    dataset["test"].to_csv("test.csv", index=False)
    
    print("Pipeline completed. Training, validation, and testing sets saved.")

# Run the pipeline
if __name__ == "__main__":
    prepare_pipeline()