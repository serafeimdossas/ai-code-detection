# AI Code Detection

Creation of classifiers able to distinguish between human-written and AI-generated code.

## Project Overview

```
project-root/
├── data/
│   ├── raw/                      # Raw CSV splits (train.csv, validation.csv, test.csv)
│   └── processed/
│       ├── tfidf/                # Precomputed TF-IDF vectors & label pickles
│       ├── features/             # Extracted Python code features 
│       └── codebert/             # Precomputed embedding arrays & label files
│
├── notebooks/                    # Exploratory notebooks
│   └── 01-exploration.ipynb
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py           # Download and split HF dataset
│   ├── features/
│   │   ├── build_code_features.py    # Generate Python code features
│   │   ├── build_codebert.py         # Generate codebert embeddings
│   │   ├── build_tfidf.py            # Generate TF-IDF embeddings
│   │   ├── FEATURES_REFERENCE.md     # MD file listing included code features
│   │   └── python_code_features.py   # Includes methods for code features extraction
│   └── models/
│       ├── xgb_tfidf/
│       │   ├── train_xgb.py          # Train XGBoost on TF-IDF features using also Python code features
│       │   ├── predict_xgb.py        # Batch prediction with XGB - TF-IDF trained model
│       │   └── predict_xgb.py        # One off prediction with XGB - TF-IDF trained model
│       ├── xgb_codebert/
│       │   ├── train_xgb_emb.py      # Train XGBoost on codebert embedding features
│       │   ├── predict_xgb_emb.py    # Batch prediction with codebert embedding-based XGB
│       │   └── predict_xgb_emb.py    # One off prediction with codebert embedding-based XGB
│       └── mlp_codebert/
│           ├── train_mlp_emb.py      # Train MLP on codebert embedding features
│           ├── predict_mlp_emb.py    # Batch prediction with MLP model
│           └── predict_mlp_emb.py    # One off prediction with MLP model
│
├── models/                           # Saved artifacts for each model
│   ├── xgb_tfidf/
│   ├── xgb_codebert/
│   └── mlp_codebert/
│
├── requirements.txt
└── README.md
```

## Setup

1. **Clone & navigate**

   ```bash
   git clone https://github.com/serafeimdossas/ai-code-detection.git
   cd ai-code-detection
   ```
2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 1. Download and Split Dataset

Downloads the **serafeimdossas/ai-code-detection** dataset from Hugging Face and splits into **train/validation/test**.

```bash
python src/data/make_dataset.py
```

This produces:

```
data/raw/
├── train.csv
├── validation.csv
└── test.csv
```

## 2. Preprocess for TF-IDF

Cleans code snippets, fits a TF-IDF vectorizer on the training split, transforms all splits, and saves feature-label pickles.

```bash
python src/features/build_tfidf.py
```

Outputs under `data/processed/tfidf/`:

```
tfidf_vectorizer.pkl
train.pkl
validation.pkl
test.pkl
```

## 3. Generate CodeBERT Embeddings

Encodes code snippets using a pretrained SentenceTransformer model and saves `.npy` arrays.

```bash
pip install sentence-transformers torch
python src/features/build_codebert.py
```

Produces under `data/processed/codebert/`:

```
train_emb.npy
train_labels.npy
validation_emb.npy
validation_labels.npy
test_emb.npy
test_labels.npy
```

## 4. Train XGBoost (TF-IDF)

```bash
python src/models/xgb_tfidf/train_xgb.py
```

This saves:

```
models/xgb_tfidf/xgb_tfidf_dense_features.json
models/xgb_tfidf/xgb_tfidf_label_encoder.pkl
models/xgb_tfidf/xgb_tfidf_metrics_{timestamp}.json
models/xgb_tfidf/xgb_tfidf_scaler.pkl
models/xgb_tfidf/xgb_tfidf.json
```

## 5. Train XGBoost (CodeBERT Embeddings)

```bash
python src/models/xgb_codebert/train_xgb_emb.py
```

This saves:

```
models/xgb_codebert/xgb_with_emb.json
models/xgb_codebert/xgb_with_emb_label_encoder.pkl
```

## 6. Predictions (Batch)

```bash
python src/models/xgb_tfidf/predict_xgb.py \
  --input examples_to_score.csv \
  --output xgb_tfidf_predictions.csv
```

## 7. Predictions (Single Snippet)

```python
python src/models/xgb_tfidf/predict_xgb_oneoff.py
```

*Note: Adjust in code the selected model and the snippet to be tested*

---

**Notes**

* To experiment with other algorithms or embedding types, add new subfolders under `src/models/`.
