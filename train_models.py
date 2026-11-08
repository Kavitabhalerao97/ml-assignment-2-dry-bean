# train_models.py
# ML Assignment 2 - Dry Bean classification.
# Run with:  python train_models.py   (after: pip install -r requirements.txt)
#
# What this file does:
# 1. reads data/dry_bean.csv
# 2. turns the text class names into numbers (LabelEncoder)
# 3. splits the data 80% train / 20% test (stratified)
# 4. builds 5 models (scaling is inside a Pipeline)
# 5. trains on the train part, checks scores on the test part
# 6. saves the models, test_data.csv and the result tables

import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# newer scikit-learn prints a warning about SVC(probability=True).
# The assignment needs probability=True for AUC, so I hide only this warning.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*",
                        category=FutureWarning)

RANDOM_SEED = 7          # same seed everywhere so results are the same every run
TEST_FRACTION = 0.20
TARGET = "Class"

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "dry_bean.csv"
MODEL_DIR = BASE_DIR / "model"
RESULT_DIR = BASE_DIR / "results"
TEST_CSV = BASE_DIR / "test_data.csv"


def build_models():
    # returns {name: (file name, pipeline)}
    return {
        # StandardScaler is the first step. The pipeline fits it on the
        # training rows only, so nothing from the test data leaks in.
        "Logistic Regression": (
            "logistic_regression.joblib",
            Pipeline([("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=2000))]),
        ),
        # A tree does not care about the scale of the features.
        # max_depth=8 so the tree cannot just remember the training rows.
        "Decision Tree": (
            "decision_tree.joblib",
            Pipeline([("model", DecisionTreeClassifier(
                max_depth=8, random_state=RANDOM_SEED))]),
        ),
        # KNN works on distances, so scaling is a must.
        # k=7: odd number so fewer ties. k=1 follows noise, a big k blurs classes.
        "KNN": (
            "knn.joblib",
            Pipeline([("scale", StandardScaler()),
                      ("model", KNeighborsClassifier(n_neighbors=7))]),
        ),
        "Naive Bayes": (
            "naive_bayes.joblib",
            Pipeline([("model", GaussianNB())]),
        ),
        # probability=True is needed for AUC, but it makes SVC slower.
        "Support Vector Machine": (
            "svm.joblib",
            Pipeline([("scale", StandardScaler()),
                      ("model", SVC(kernel="rbf", probability=True,
                                    random_state=RANDOM_SEED))]),
        ),
    }


def score_model(fitted, X_test, y_test):
    # the 6 metrics, on the test data only
    y_pred = fitted.predict(X_test)
    y_proba = fitted.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        # macro = work out the score for each class, then take the plain
        # average. So every bean type counts the same, even small BOMBAY.
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def to_markdown_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [str(row[cols[0]])] + [f"{row[c]:.4f}" for c in cols[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    started = time.time()
    line = "=" * 72
    print(line)
    print("ML Assignment 2 – Dry Bean Classification – training run")
    print(line)
    print(f"Date/time        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Python version   : {platform.python_version()} ({platform.system()})")
    print(f"scikit-learn     : {sklearn.__version__}")
    print(f"pandas           : {pd.__version__}")
    print(f"Random seed      : {RANDOM_SEED}")
    print(line)

    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    # 1. load data
    print("\n[1/5] Loading data ...")
    beans = pd.read_csv(DATA_CSV)
    feature_columns = [c for c in beans.columns if c != TARGET]
    print(f"      rows={len(beans)}, features={len(feature_columns)}, "
          f"classes={beans[TARGET].nunique()}")

    # 2. encode the class names
    print("[2/5] Changing class names to numbers ...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(beans[TARGET])
    X = beans[feature_columns]
    print(f"      classes (in order): {list(label_encoder.classes_)}")

    # 3. split
    print("[3/5] Stratified 80/20 train/test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=RANDOM_SEED)
    print(f"      train rows={len(X_train)}, test rows={len(X_test)}")

    # test_data.csv keeps the original column names and the text class names,
    # so it is easy to read and easy to upload in the app
    test_frame = X_test.copy()
    test_frame[TARGET] = label_encoder.inverse_transform(y_test)
    test_frame.to_csv(TEST_CSV, index=False)
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "feature_columns.json", "w") as handle:
        json.dump(feature_columns, handle, indent=2)

    # 4. train and score each model
    print("[4/5] Training and testing the 5 models ...")
    rows = []
    for name, (file_name, pipeline) in build_models().items():
        t0 = time.time()
        print(f"      -> {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)                   # train data only
        joblib.dump(pipeline, MODEL_DIR / file_name)
        scores = score_model(pipeline, X_test, y_test)   # test data only
        scores = {k: round(float(v), 4) for k, v in scores.items()}
        rows.append({"ML Model Name": name, **scores})
        print(f"done in {time.time() - t0:.1f}s  (accuracy {scores['Accuracy']:.4f})")

    # 5. save the tables
    print("[5/5] Saving result tables and files ...")
    results = pd.DataFrame(rows)
    results.to_csv(RESULT_DIR / "metrics_comparison.csv", index=False)
    (RESULT_DIR / "metrics_comparison.md").write_text(to_markdown_table(results))

    print("\n" + line)
    print("FINAL COMPARISON (test set, macro averages, one-vs-rest AUC)")
    print(line)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(line)
    print(f"Saved to model/   : {', '.join(sorted(p.name for p in MODEL_DIR.iterdir()))}")
    print("Saved to results/ : metrics_comparison.csv, metrics_comparison.md")
    print("Saved            : test_data.csv")
    print(f"Total run time   : {time.time() - started:.1f} seconds")
    print(line)


if __name__ == "__main__":
    main()
