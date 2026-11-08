"""
train_models.py  -  Stage 2: train, evaluate and save 5 classifiers.

ML Assignment 2 - Dry Bean Classification.
Run with:  python train_models.py     (after: pip install -r requirements.txt)

Steps
1. Read data/dry_bean.csv (made by explore_data.py)
2. Encode the text labels (Class) into numbers with LabelEncoder
3. Stratified 80/20 train/test split
4. Build 5 models (scaling lives INSIDE a Pipeline -> no data leakage)
5. Fit on the training part only, score on the test part only
6. Save models, preprocessing artifacts, test_data.csv and metric tables
"""
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
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# scikit-learn 1.9 announces that SVC(probability=True) will go away in v1.11.
# The assignment asks for probability=True (needed for AUC), so we hide only
# this one notice to keep the terminal output clean.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*",
                        category=FutureWarning)

RANDOM_SEED = 7          # one seed, used everywhere, so results repeat exactly
TEST_FRACTION = 0.20
TARGET = "Class"

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "dry_bean.csv"
MODEL_DIR = BASE_DIR / "model"
RESULT_DIR = BASE_DIR / "results"
TEST_CSV = BASE_DIR / "test_data.csv"


def build_models():
    """Return {display name: (file name, unfitted pipeline)}."""
    return {
        # Scaler first, then the model. The Pipeline fits the scaler on the
        # training rows only, and re-uses those same numbers on new data.
        "Logistic Regression": (
            "logistic_regression.joblib",
            Pipeline([("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=2000))]),
        ),
        # Trees split on thresholds, so feature scale does not matter.
        # max_depth=8 stops the tree from memorising the training rows.
        "Decision Tree": (
            "decision_tree.joblib",
            Pipeline([("model", DecisionTreeClassifier(
                max_depth=8, random_state=RANDOM_SEED))]),
        ),
        # KNN uses distances, so scaling is essential. k=7: odd number
        # (fewer ties), not so small that it follows noise (k=1), not so
        # large that it blurs the neighbouring classes.
        "KNN": (
            "knn.joblib",
            Pipeline([("scale", StandardScaler()),
                      ("model", KNeighborsClassifier(n_neighbors=7))]),
        ),
        "Naive Bayes": (
            "naive_bayes.joblib",
            Pipeline([("model", GaussianNB())]),
        ),
        # probability=True is needed for AUC but makes SVC slower
        # (it runs an extra internal cross-validation).
        "Support Vector Machine": (
            "svm.joblib",
            Pipeline([("scale", StandardScaler()),
                      ("model", SVC(kernel="rbf", probability=True,
                                    random_state=RANDOM_SEED))]),
        ),
    }


def score_model(fitted, X_test, y_test):
    """The 6 required metrics, on test data only, rounded to 4 decimals."""
    y_pred = fitted.predict(X_test)
    y_proba = fitted.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        # macro = compute per class, then plain average: every bean type
        # counts equally, even the small BOMBAY class.
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

    # 1. data ---------------------------------------------------------------
    print("\n[1/5] Loading data ...")
    beans = pd.read_csv(DATA_CSV)
    feature_columns = [c for c in beans.columns if c != TARGET]
    print(f"      rows={len(beans)}, features={len(feature_columns)}, "
          f"classes={beans[TARGET].nunique()}")

    # 2. encode target ------------------------------------------------------
    print("[2/5] Encoding the target labels ...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(beans[TARGET])
    X = beans[feature_columns]
    print(f"      classes (in order): {list(label_encoder.classes_)}")

    # 3. split --------------------------------------------------------------
    print("[3/5] Stratified 80/20 train/test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=RANDOM_SEED)
    print(f"      train rows={len(X_train)}, test rows={len(X_test)}")

    # readable held-out file: original column names + original text labels
    test_frame = X_test.copy()
    test_frame[TARGET] = label_encoder.inverse_transform(y_test)
    test_frame.to_csv(TEST_CSV, index=False)
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "feature_columns.json", "w") as handle:
        json.dump(feature_columns, handle, indent=2)

    # 4 + 5. train and evaluate --------------------------------------------
    print("[4/5] Training and evaluating the 5 models ...")
    rows = []
    for name, (file_name, pipeline) in build_models().items():
        t0 = time.time()
        print(f"      -> {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)               # training data only
        joblib.dump(pipeline, MODEL_DIR / file_name)
        scores = score_model(pipeline, X_test, y_test)  # test data only
        scores = {k: round(float(v), 4) for k, v in scores.items()}
        rows.append({"ML Model Name": name, **scores})
        print(f"done in {time.time() - t0:.1f}s  (accuracy {scores['Accuracy']:.4f})")

    # 6. save tables --------------------------------------------------------
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
