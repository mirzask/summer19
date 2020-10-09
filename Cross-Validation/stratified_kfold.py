import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Applies Stratified K-fold cross validation using sci-kit learn. Adds a k-fold column to the provided CSV file."
    )

    parser.add_argument(
        "-i", "--input_csv", type=str, required=True, help="Input csv file."
    )

    parser.add_argument(
        "--label_column_name", type=str, required=True, help="Column name in CSV file that contains labels."
    )

    parser.add_argument(
        "-k", "--num_folds", type=int, default=5, help="Number of folds (default=5)."
    )

    args = parser.parse_args()

    # Read CSV file
    df = pd.read_csv(args.input_csv)

    # Create a new column called 'kfold'. Initialize with all values = -1
    df["kfold"] = -1
    # Randomize the rows of data
    df = df.sample(frac=1).reset_index(drop=True)

    # Extract target ('y') values
    y = df[args.label_column_name].values

    # Initiate K-fold
    kf = StratifiedKFold(n_splits=args.num_folds)

    # Fill in the 'kfold' column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, "kfold"] = fold

    file_ = Path(args.input_csv)
    p = file_.parent / (file_.stem+'_folds.csv')
    # Save the csv file with the new 'kfold' column
    df.to_csv(p, index=False)
