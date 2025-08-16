import os, pandas as pd, joblib
from pathlib import Path
from python_code_features import python_code_features

def build_code_features(input_dir="data/raw", output_dir="data/processed/features", code_col="code"):
    """
    Reads raw code data from CSV files (train/validation/test),
    extracts features from the code using `python_code_features`,
    and saves the dense numerical features as compressed .pkl files.
    """

    # create output directory if it doesn't already exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # parse dataset splits (train, validation, and test)
    for split in ("train","validation","test"):
        # load the CSV for current split
        df = pd.read_csv(os.path.join(input_dir, f"{split}.csv"))

        # use `python_code_features` function for each row in the code column
        # and convert the resulting series into a DataFrame using apply(pd.Series)
        feats = df[code_col].fillna("").map(python_code_features).apply(pd.Series)

        # save features as pickle file 
        joblib.dump(feats.astype("float32"), os.path.join(output_dir, f"{split}_dense_features.pkl"), compress=3)
        print(split, feats.shape)

if __name__ == "__main__":
    build_code_features()
