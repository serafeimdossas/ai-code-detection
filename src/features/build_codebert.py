# src/features/build_codebert.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed/codebert"
MODEL_NAME = "microsoft/codebert-base"

# Function to clean and normalize code snippets
def clean_code(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    return " ".join(snippet.split())

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Process each split
    for split in ["train", "validation", "test"]:
        csv_path = os.path.join(INPUT_DIR, f"{split}.csv")
        df = pd.read_csv(csv_path)
        
        # Create clean code column by normalizing formatting
        df['code_clean'] = df['code'].fillna('').apply(clean_code)

        # Generate embeddings
        embeddings = model.encode(
            df['code_clean'].tolist(),
            batch_size=32,
            show_progress_bar=True
        )
        embeddings = np.array(embeddings)

        # Save embeddings and labels
        np.save(os.path.join(OUTPUT_DIR, f"{split}_emb.npy"), embeddings)
        np.save(os.path.join(OUTPUT_DIR, f"{split}_labels.npy"), df['label'].values) # type: ignore
        print(f"Saved {split} embeddings: {embeddings.shape}")

    print("Embedding generation complete.")

if __name__ == '__main__':
    main()
