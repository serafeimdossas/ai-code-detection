# src/data/make_dataset.py

import os
from pathlib import Path
from datasets import load_dataset, DatasetDict


def download_and_split(
    dataset_name: str = "isThisYouLLM/H-AIRosettaMP",
    output_dir: str = "data/raw//H-AIRosettaMP",
    train_ratio: float = 0.80,
    seed: int = 42
):
    """
    Downloads the H-AIRosettaMP dataset, filters to only Python entries,
    then splits into train/validation/test and writes each to CSV under output_dir.

    Only rows where 'language' is 'Python' are kept.
    Splitting ratios: train, validation, test = 80%, 10%, 10% by default.
    """

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load entire dataset (merge all splits if needed)
    ds_all = load_dataset(dataset_name)

    if isinstance(ds_all, DatasetDict):
        # concatenate all splits into one
        keys = list(ds_all.keys())
        ds = ds_all[keys[0]]
        for split in keys[1:]:
            ds = ds.concatenate(ds_all[split]) # type: ignore
    else:
        ds = ds_all

    # Filter to only Python-language snippets
    ds = ds.filter(lambda example: example.get('language_name', '') == 'Python')

    # Calculate test ratio (10%) (10%)
    test_ratio = (1.0 - train_ratio) / 2

    # Split off test set
    split1 = ds.train_test_split(test_size=test_ratio, seed=seed) # type: ignore
    remainder = split1['train']
    test_set = split1['test']

    # Split remainder into train and validation
    # validation ratio relative to remainder = (1 - train_ratio) / (train_ratio + (1 - train_ratio)) = test_ratio * 2?
    val_ratio = test_ratio
    split2 = remainder.train_test_split(test_size=val_ratio, seed=seed)
    train_set = split2['train']
    val_set = split2['test']

    splits = {
        'train': train_set,
        'validation': val_set,
        'test': test_set
    }

    # Save splits to CSV
    for name, dataset in splits.items():
        out_path = os.path.join(output_dir, f"{name}.csv")
        print(f"Writing {name} ({len(dataset)} samples) to {out_path}")
        dataset.to_csv(out_path, index=False)

    print("Dataset download, filter, and splitting complete.")


if __name__ == '__main__':
    download_and_split()
