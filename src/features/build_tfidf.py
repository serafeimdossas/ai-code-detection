# src/features/build_tfidf.py

import os
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def preprocess_and_vectorize(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed/tfidf",
    code_col: str = "code",
    label_col: str = "label",
    ngram_range=(3,6),
    max_features=50000
):
    """
    Loads CSVs from input_dir, cleans code snippets, vectorizes with TF-IDF,
    and saves feature-label pickles and the vectorizer to output_dir.

    Handles missing code entries by filling with empty strings.
    """
    # Ensure output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load CSVs
    dfs = {}
    for split in ("train", "validation", "test"):
        path = os.path.join(input_dir, f"{split}.csv")
        df = pd.read_csv(path)
        # Ensure code_col exists
        if code_col not in df.columns:
            raise KeyError(f"Column '{code_col}' not found in {path}")
        # Fill missing code with empty string
        df[code_col] = df[code_col].fillna("")
        dfs[split] = df

    # Cleaning function
    def clean_code(snippet: str) -> str:
        if not isinstance(snippet, str):
            return ""
        # normalize whitespace
        return " ".join(snippet.split())

    # Apply cleaning
    for split, df in dfs.items():
        dfs[split]["code_clean"] = df[code_col].apply(clean_code)

    # Fit TF-IDF on train
    vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features
    )
    X_train = vect.fit_transform(dfs["train"]["code_clean"])
    
    # Save vectorizer
    joblib.dump(vect, os.path.join(output_dir, "tfidf_vectorizer.pkl"))

    # Transform val & test
    X_val = vect.transform(dfs["validation"]["code_clean"])
    X_test = vect.transform(dfs["test"]["code_clean"])

    # Extract labels
    if label_col not in dfs["train"].columns:
        raise KeyError(f"Label column '{label_col}' not found in train split")
    y_train = dfs["train"][label_col]
    y_val = dfs["validation"][label_col]
    y_test = dfs["test"][label_col]

    # Save feature-label pickles
    joblib.dump((X_train, y_train), os.path.join(output_dir, "train.pkl"))
    joblib.dump((X_val, y_val), os.path.join(output_dir, "validation.pkl"))
    joblib.dump((X_test, y_test), os.path.join(output_dir, "test.pkl"))

    print("Preprocessing and vectorization complete.")


if __name__ == "__main__":
    preprocess_and_vectorize()
