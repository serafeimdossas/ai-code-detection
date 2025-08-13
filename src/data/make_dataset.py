# src/data/make_dataset.py

from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def download_and_split(
    dataset_name: str = "serafeimdossas/ai-code-detection",
    output_dir: str = "data/raw/",
):
    """
    Downloads the serafeimdossas/ai-code-detection dataset,
    and splits into train/validation/test and writes each to CSV under output_dir.

    Splitting ratios: train, validation, test = 80%, 10%, 10% by default.
    """

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Let's assume your dataset is already processed into a single split
    df = dataset["train"].to_pandas() # type: ignore

    # Step 1: Split into 80% train and 20% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"] # type: ignore
    )

    # Step 2: Split temp into 50% validation, 50% test (10% each of total)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, shuffle=True, stratify=temp_df["label"]
    )

    # Save as CSV
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    # Print confirmation 
    print("Dataset download and splitting complete.")


if __name__ == '__main__':
    download_and_split()
