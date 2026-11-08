# explore_data.py
# Step 1: have a look at the Dry Bean data before training anything.
#
# What this file does:
# 1. reads the excel file in data/raw
# 2. prints shape, column names, data types, missing values, duplicates, class counts
# 3. removes duplicate rows and saves a clean csv (data/dry_bean.csv)
#
# Run it with:  python explore_data.py

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_XLSX = BASE_DIR / "data" / "raw" / "Dry_Bean_Dataset.xlsx"
CLEAN_CSV = BASE_DIR / "data" / "dry_bean.csv"
TARGET = "Class"


def show_title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main():
    show_title("1. Reading the excel file")
    beans = pd.read_excel(RAW_XLSX, engine="openpyxl")
    print("Loaded:", RAW_XLSX.name)

    show_title("2. Shape (rows, columns)")
    print(beans.shape)

    show_title("3. Column names")
    for number, name in enumerate(beans.columns, start=1):
        print(f"{number:>2}. {name}")

    show_title("4. Data types")
    print(beans.dtypes.to_string())

    show_title("5. Missing values in each column")
    print(beans.isnull().sum().to_string())
    print("Total missing cells:", int(beans.isnull().sum().sum()))

    show_title("6. Duplicate rows")
    duplicates = int(beans.duplicated().sum())
    print("Duplicate rows:", duplicates)

    show_title("7. Class counts")
    print(beans[TARGET].value_counts().to_string())

    show_title("8. Basic statistics of the features")
    print(beans.drop(columns=TARGET).describe().T.round(3).to_string())

    # Remove duplicates BEFORE the train/test split. If we don't, the same
    # row can end up in both train and test, and the test score looks
    # better than it really is.
    if duplicates > 0:
        show_title("9. Removing duplicates")
        beans = beans.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates} duplicate rows. New shape: {beans.shape}")
        print(beans[TARGET].value_counts().to_string())

    beans.to_csv(CLEAN_CSV, index=False)
    show_title("Saved")
    print(f"Clean csv saved to data/dry_bean.csv, shape = {beans.shape}")


if __name__ == "__main__":
    main()
