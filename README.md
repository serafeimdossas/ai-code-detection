# AI Code Detection

Creation of classifiers able to distinguish between human-written and AI-generated code.

## Project Overview

```
project-root/
├── data/
│   ├── raw/                      # Raw CSV splits (train.csv, validation.csv, test.csv)
│   └── processed/
│       ├── tfidf/                # Precomputed TF-IDF vectors & label pickles
│       └── codebert/             # Precomputed embedding arrays & label files
│
├── notebooks/                    # Exploratory notebooks
│   └── 01-exploration.ipynb
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py           # Download and split HF dataset
│   ├── features/
│   │   ├── build_codebert.py         # Generate codebert embeddings
│   │   └── build_tfidf.py            # Generate TF-IDF embeddings
│   └── models/
│       ├── xgb_tfidf/
│       │   ├── train_xgb.py          # Train XGBoost on TF-IDF features
│       │   └── predict_xgb.py        # Batch prediction with XGB baseline
│       ├── xgb_codebert/
│       │   ├── train_xgb_emb.py      # Train XGBoost on codebert embedding features
│       │   └── predict_xgb_emb.py    # Batch prediction with codebert embedding-based XGB
│       └── mlp_codebert/
│           ├── train_mlp_emb.py      # Train MLP on codebert embedding features
│           └── predict_mlp_emb.py    # Batch prediction with MLP model
│
├── models/                           # Saved model artifacts
│   ├── xgb_tfidf/
│   │   ├── xgb_baseline.json
│   │   └── xgb_baseline_label_encoder.pkl
│   ├── xgb_codebert/
│   │   ├── xgb_with_emb.json
│   │   └── xgb_with_emb_label_encoder.pkl
│   └── mlp_codebert/
│       ├── mlp_emb.pt                # Best MLP model weights
│       └── mlp_emb_label_encoder.pkl
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

Downloads the H-AIRosettaMP dataset from Hugging Face, filters for Python snippets, and splits into **train/validation/test**.

```bash
python src/data/make_dataset.py \
  --dataset_name isThisYouLLM/H-AIRosettaMP \
  --output_dir data/raw/H-AIRosettaMP \
  --train_ratio 0.8
```

This produces:

```
data/raw/H-AIRosettaMP/
├── train.csv
├── validation.csv
└── test.csv
```

## 2. Preprocess for TF-IDF

Cleans code snippets, fits a TF-IDF vectorizer on the training split, transforms all splits, and saves feature-label pickles.

```bash
python src/features/build_tfidf.py \
  --input_dir data/raw/H-AIRosettaMP \
  --output_dir data/processed/tfidf \
  --code_col code \
  --label_col target \
  --clean
```

Outputs under `data/processed/tfidf/`:

```
tfidf_vectorizer.pkl
train.pkl
validation.pkl
test.pkl
```

## 3. Train XGBoost (TF-IDF)

```bash
python src/models/xgb_tfidf/train_xgb.py \
  --data_dir data/processed/tfidf \
  --model_out models/xgb_tfidf/xgb_baseline.json \
  --n_estimators 500 \
  --learning_rate 0.1 \
  --max_depth 6 \
  --early_stopping_rounds 20
```

This saves:

```
models/xgb_tfidf/xgb_baseline.json
models/xgb_tfidf/xgb_baseline_label_encoder.pkl
```

## 4. Generate Embeddings

Encodes code snippets using a pretrained SentenceTransformer model and saves `.npy` arrays.

```bash
pip install sentence-transformers torch
python src/features/build_codebert.py \
  --input_dir data/raw/H-AIRosettaMP \
  --output_dir data/processed/{embeddings-type} \
  --model_name microsoft/codebert-base
```

Produces under `data/processed/{embeddings-type}/`:

```
train_emb.npy
train_labels.npy
validation_emb.npy
validation_labels.npy
test_emb.npy
test_labels.npy
```

## 5. Train XGBoost (Embeddings)

```bash
python src/models/xgb_codebert/train_xgb_emb.py \
  --data_dir data/processed/{embeddings-type} \
  --model_out models/xgb_codebert/xgb_with_emb.json \
  --n_estimators 500 \
  --learning_rate 0.1 \
  --max_depth 6 \
  --early_stopping_rounds 20
```

This saves:

```
models/xgb_codebert/xgb_with_emb.json
models/xgb_codebert/xgb_with_emb_label_encoder.pkl
```

## 6. Usage (Batch)

```bash
python src/models/xgb_tfidf/predict_xgb.py \
  --model models/xgb_tfidf/xgb_baseline.json \
  --vectorizer data/processed/tfidf/tfidf_vectorizer.pkl \
  --label_encoder models/xgb_tfidf/xgb_baseline_label_encoder.pkl \
  --input examples_to_score.csv \
  --output predictions_xgb.csv
```

## 7. Usage (Single Snippet)

```python
python src/models/xgb_tfidf/predict_xgb_oneoff.py
```

*Note: Adjust in code the selected model and the snippet to be tested*

---

**Notes**

* To experiment with other algorithms or embedding types, add new subfolders under `src/models/`.
