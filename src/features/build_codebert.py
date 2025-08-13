# src/features/build_codebert.py

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for code snippets using a pretrained model"
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/raw",
        help="Directory containing train.csv, validation.csv, test.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed/codebert",
        help="Directory to save embedding arrays and label files"
    )
    parser.add_argument(
        "--model_name", type=str, default="microsoft/codebert-base",
        help="HuggingFace model name or path for embeddings"
    )
    return parser.parse_args()


def clean_code(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    return " ".join(snippet.split())


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize embedding model
    model = SentenceTransformer(args.model_name)

    # Process each split
    for split in ["train", "validation", "test"]:
        csv_path = os.path.join(args.input_dir, f"{split}.csv")
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
        np.save(os.path.join(args.output_dir, f"{split}_emb.npy"), embeddings)
        np.save(os.path.join(args.output_dir, f"{split}_labels.npy"), df['label'].values) # type: ignore
        print(f"Saved {split} embeddings: {embeddings.shape}")

    print("Embedding generation complete.")

if __name__ == '__main__':
    main()
