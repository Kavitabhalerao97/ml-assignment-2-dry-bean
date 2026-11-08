"""
app.py  -  Streamlit app: Dry Bean classifier evaluation board.

The app only LOADS the models saved by train_models.py. It never trains
(this file contains no training code at all). The user uploads test data that
contains the 16 feature columns plus the true 'Class' column, picks a model,
and the app shows metrics, a confusion matrix and a classification report.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

BEAN_CMAP = LinearSegmentedColormap.from_list(
    "bean", ["#FBF6EC", "#E8C79A", "#C9733A", "#7A2E12"])
BAR_COLOURS = ["#A8481F", "#2F6F6B", "#D9A441", "#6B4E8A", "#4C7A34", "#B23A48"]

st.set_page_config(page_title="Bean Sorter Lab", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(120deg, #3B2415 0%, #7A2E12 55%, #A8481F 100%);
           padding: 1.6rem 2rem; border-radius: 18px; color: #FBF6EC; margin-bottom: 1.2rem;}
    .hero h1 {margin: 0; font-size: 2.1rem; color: #FBF6EC;}
    .hero p {margin: .3rem 0 0 0; font-size: 1.05rem; opacity: .9;}
    .stat-card {background: #F0E4CF; border-left: 6px solid #A8481F; border-radius: 10px;
                padding: .8rem 1rem; margin-bottom: .5rem;}
    .stat-card .label {font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7;}
    .stat-card .value {font-size: 1.8rem; font-weight: 700;}
    .step {background: #FFFFFF; border: 1px solid #E3D3B6; border-radius: 12px; padding: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading ---
@st.cache_resource(show_spinner="Loading saved models ...")
def load_assets():
    """Load every saved model and preprocessing artifact (done once).

    Safety note: joblib files are pickles, so only load files you trust. These
    are produced by this repo's own train_models.py - never by uploaded data.
    """
    models = {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "feature_columns.json") as handle:
        feature_columns = json.load(handle)
    return models, encoder, feature_columns


# ------------------------------------------------------------- evaluation ---
@st.cache_data(show_spinner=False)
def evaluate(model_name, features, y_true, _model, n_classes):
    """Score one saved model on the uploaded data. Nothing is trained here."""
    y_pred = _model.predict(features)
    present = np.unique(y_true)

    auc, auc_note = np.nan, ""
    if len(present) < n_classes:
        auc_note = (f"AUC needs every one of the {n_classes} classes to appear in the "
                    f"uploaded file; only {len(present)} do.")
    else:
        try:
            proba = _model.predict_proba(features)
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        except ValueError as problem:
            auc_note = f"AUC could not be computed: {problem}"

    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    shown = np.unique(np.concatenate([y_true, y_pred]))
    report = classification_report(y_true, y_pred, labels=shown, output_dict=True,
                                   zero_division=0)
    return {"scores": scores, "auc_note": auc_note, "matrix": matrix,
            "report": report, "shown": shown}


def fmt(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def stat_card(label, value):
    st.markdown(
        f'<div class="stat-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ pages ---
def welcome(models_ready):
    st.markdown(
        '<div class="hero"><h1>🫘 Bean Sorter Lab</h1>'
        '<p>Upload test data, pick a saved model, and see how well it recognises '
        'seven kinds of dry beans.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><b>1 · Get data</b><br>Press <i>Download sample test '
                'data</i> in the sidebar, or use your own CSV with the same columns.</div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="step"><b>2 · Upload</b><br>Drop the CSV into the sidebar '
                'uploader. It must contain the 16 features and the <code>Class</code> '
                'column.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="step"><b>3 · Explore</b><br>Choose a model and read the '
                'metrics, confusion matrix and report. Models are never retrained.</div>',
                unsafe_allow_html=True)
    saved = BASE_DIR / "results" / "metrics_comparison.csv"
    if saved.exists():
        st.subheader("Results recorded at training time")
        st.caption("Scores of the five models on the held-out test split (from train_models.py).")
        st.dataframe(pd.read_csv(saved), hide_index=True, width="stretch")


def draw_matrix(matrix, class_names, title):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=BEAN_CMAP, cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.6, linecolor="#FBF6EC", ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return fig


def draw_comparison_bars(table):
    cols = [c for c in METRIC_NAMES if table[c].notna().any()]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(cols)
    positions = np.arange(len(table))
    for i, col in enumerate(cols):
        ax.bar(positions + i * width, table[col], width, label=col,
               color=BAR_COLOURS[i % len(BAR_COLOURS)])
    ax.set_xticks(positions + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(table.index, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(ncol=len(cols), loc="lower center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    try:
        models, encoder, feature_columns = load_assets()
    except Exception as problem:  # missing / incompatible model files
        st.error(f"Could not load the saved models: {problem}")
        st.info("Run `python train_models.py` once, then restart the app.")
        st.stop()
    class_names = list(encoder.classes_)

    with st.sidebar:
        st.markdown("## 🫘 Bean Sorter Lab")
        st.write("Five classifiers, trained once on the UCI **Dry Bean** dataset "
                 "(16 shape measurements, 7 bean types), are loaded from disk here. "
                 "Upload test data to evaluate them.")
        model_name = st.selectbox("Choose a model", list(MODEL_FILES))
        sample_path = BASE_DIR / "test_data.csv"
        if sample_path.exists():
            st.download_button("⬇ Download sample test data", sample_path.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
        upload = st.file_uploader("Upload test data (CSV)", type="csv")

    if upload is None:
        welcome(True)
        return

    # ---- read + validate ----------------------------------------------------
    try:
        upload.seek(0)  # Streamlit reruns the script; always read from the start
        raw = pd.read_csv(upload)
    except Exception as problem:
        st.error(f"That file could not be read as a CSV: {problem}")
        return
    needed = feature_columns + [TARGET]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        st.error("The uploaded file is missing required column(s): " + ", ".join(missing))
        st.info("Expected columns: " + ", ".join(needed))
        return

    data = raw[needed].copy()
    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    before = len(data)
    data = data.dropna(subset=feature_columns + [TARGET])
    if len(data) < before:
        st.warning(f"{before - len(data)} row(s) with empty or non-numeric values were ignored.")

    known = data[TARGET].astype(str).str.strip().isin(class_names)
    if (~known).any():
        unseen = sorted(data.loc[~known, TARGET].astype(str).unique())
        st.warning(f"{int((~known).sum())} row(s) have labels the models were never trained on "
                   f"({', '.join(unseen)}) and were ignored. Known classes: {', '.join(class_names)}.")
    data = data[known]
    if data.empty:
        st.error("No usable rows are left after cleaning, so nothing can be evaluated.")
        return

    features = data[feature_columns]
    y_true = encoder.transform(data[TARGET].astype(str).str.strip())
    n_classes = len(class_names)

    st.markdown(
        f'<div class="hero"><h1>🫘 {model_name}</h1><p>Evaluated on {len(data):,} uploaded rows '
        f'· {len(np.unique(y_true))} of {n_classes} classes present</p></div>',
        unsafe_allow_html=True)

    result = evaluate(model_name, features, y_true, models[model_name], n_classes)

    tab_metrics, tab_matrix, tab_report, tab_compare, tab_data = st.tabs(
        ["📊 Metrics", "🧩 Confusion matrix", "📝 Classification report",
         "⚔️ Compare all 5 models", "🔍 Data preview"])

    with tab_metrics:
        columns = st.columns(3)
        for i, metric in enumerate(METRIC_NAMES):
            with columns[i % 3]:
                stat_card(metric, fmt(result["scores"][metric]))
        if result["auc_note"]:
            st.info("AUC shows N/A. " + result["auc_note"])
        st.caption("Precision, Recall and F1 are macro averages (each class counts equally). "
                   "AUC is one-vs-rest, macro averaged.")

    with tab_matrix:
        left, right = st.columns([3, 2])
        with left:
            st.pyplot(draw_matrix(result["matrix"], class_names, f"{model_name} – confusion matrix"))
        with right:
            st.write("Each row is the **true** bean type, each column is what the model "
                     "**predicted**. Numbers on the diagonal are correct predictions; "
                     "anything off the diagonal is a mix-up.")

    with tab_report:
        report = pd.DataFrame(result["report"]).T
        report.index = [class_names[int(i)] if i.isdigit() else i for i in report.index]
        report["support"] = report["support"].astype(int)
        st.dataframe(report.round(4), width="stretch")

    with tab_compare:
        rows = {name: evaluate(name, features, y_true, models[name], n_classes)["scores"]
                for name in MODEL_FILES}
        table = pd.DataFrame(rows).T[METRIC_NAMES]
        styled = (table.style.format(na_rep="N/A", precision=4)
                  .highlight_max(axis=0, color="#F2C879"))
        st.dataframe(styled, width="stretch")
        st.caption("Gold cell = best model for that metric.")
        st.pyplot(draw_comparison_bars(table))

    with tab_data:
        st.write(f"Showing the first 10 of {len(data):,} rows · {data.shape[1]} columns.")
        st.dataframe(data.head(10), width="stretch")
        st.write("Class counts in the uploaded file")
        st.bar_chart(data[TARGET].value_counts())


main()
