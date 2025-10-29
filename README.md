# AI-Based Legal Document Classification using Deep Learning and NLP

## Project Overview
This project focuses on building an **AI-based system for legal document classification** using **Deep Learning (DL)** and **Natural Language Processing (NLP)** techniques. The system is designed to classify legal documents into multiple relevant categories (multi-label classification) based on their content. 

The project leverages **Legal-BERT**, a transformer-based model fine-tuned for legal text, to achieve state-of-the-art performance in the legal NLP domain. The system is modular, explainable, and designed for real-world applicability in the LegalTech industry.

---

## Objectives
1. **Automate Legal Document Classification**: Reduce manual effort in categorizing legal documents by leveraging AI.
2. **Leverage Deep Learning**: Use transformer-based architectures (Legal-BERT) for multi-label classification.
3. **Ensure Explainability**: Integrate explainability methods (e.g., SHAP, LIME) to make the model's predictions interpretable.
4. **Evaluate Performance**: Use robust evaluation metrics such as Micro-F1, Macro-F1, Precision@K, and Subset Accuracy.
5. **Benchmark Comparisons**: Compare the deep learning model with traditional baselines (e.g., TF-IDF + Logistic Regression).

---

## Workflow and System Design
The project workflow is designed to be modular and scalable:
1. **Data Preprocessing**:
   - Remove non-textual elements, normalize text, and tokenize using Legal-BERT tokenizer.
   - Encode multi-label tags using `MultiLabelBinarizer`.
   - Split the dataset into training, validation, and testing sets (80/10/10).
2. **Model Training**:
   - Fine-tune the Legal-BERT model on the Eur-Lex 57K dataset.
   - Use multi-label classification loss functions (e.g., Binary Cross-Entropy).
3. **Explainability**:
   - Apply SHAP and LIME to explain model predictions and ensure transparency.
4. **Evaluation**:
   - Evaluate the model using Micro-F1, Macro-F1, Precision@K, and Subset Accuracy.
5. **Baseline Comparison**:
   - Compare the Legal-BERT model with traditional ML baselines (e.g., TF-IDF + Logistic Regression).

---

## Tools and Technologies
- **Deep Learning Frameworks**:
  - [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) (Transformer-based model for legal text)
  - Hugging Face `transformers` library
- **NLP Libraries**:
  - `datasets` (for loading and preprocessing the Eur-Lex 57K dataset)
  - `scikit-learn` (for label encoding and baseline models)
- **Explainability Tools**:
  - SHAP (SHapley Additive exPlanations)
  - LIME (Local Interpretable Model-Agnostic Explanations)
- **Evaluation Metrics**:
  - Micro-F1, Macro-F1, Precision@K, Subset Accuracy

---
## Contributors
This project is developed collaboratively by a team of 5 members:
- **Alay Patel**
- **Apurva Deodhar**
- **Aum Bosmiya**
- **Maharshi Patel**
- **Harsh Patel**

---

## Acknowledgments
- **Professor Chintan Shah** for his valuable feedback and guidance.
- **Hugging Face** for providing the `transformers` library and Legal-BERT model.
- **LexGLUE Benchmark** for the Eur-Lex 57K dataset.
