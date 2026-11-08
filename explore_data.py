"""
explore_data.py  -  Stage 1: look at the Dry Bean dataset before modelling.

What it does
1. Reads the original Excel file (data/raw/Dry_Bean_Dataset.xlsx) with pandas.
2. Prints shape, column names, data types, missing values, duplicates, class counts.
3. Saves a clean CSV copy to data/dry_bean.csv (train_models.py reads this CSV).

Run:  python explore_data.py
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_XLSX = BASE_DIR / "data" / "raw" / "Dry_Bean_Dataset.xlsx"
CLEAN_CSV = BASE_DIR / "data" / "dry_bean.csv"
TARGET = "Class"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main():
    banner("1. Reading the Excel file")
    beans = pd.read_excel(RAW_XLSX, engine="openpyxl")
    print(f"Loaded: {RAW_XLSX.name}")

    banner("2. Shape (rows, columns)")
    print(beans.shape)

    banner("3. Column names")
    for position, name in enumerate(beans.columns, start=1):
        print(f"{position:>2}. {name}")

    banner("4. Data types")
    print(beans.dtypes.to_string())

    banner("5. Missing values per column")
    print(beans.isnull().sum().to_string())
    print(f"Total missing cells: {int(beans.isnull().sum().sum())}")

    banner("6. Duplicate rows")
    n_dup = int(beans.duplicated().sum())
    print(f"Fully duplicated rows: {n_dup}")

    banner("7. Class counts (target = 'Class')")
    print(beans[TARGET].value_counts().to_string())

    banner("8. Quick statistics of the features")
    print(beans.drop(columns=TARGET).describe().T.round(3).to_string())

    # Duplicates are removed BEFORE the train/test split. Otherwise the same
    # row could land in both train and test, and the test score would look
    # better than it really is (a form of data leakage).
    if n_dup > 0:
        banner("9. Removing duplicates")
        beans = beans.drop_duplicates().reset_index(drop=True)
        print(f"Removed {n_dup} duplicate rows. New shape: {beans.shape}")
        print(beans[TARGET].value_counts().to_string())

    beans.to_csv(CLEAN_CSV, index=False)
    banner("Saved")
    print(f"Clean CSV written to: {CLEAN_CSV.relative_to(BASE_DIR)}  shape={beans.shape}")


if __name__ == "__main__":
    main()
