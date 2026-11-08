# Machine Learning Assignment 2 – Dry Bean Classification

Five classification models on the UCI Dry Bean dataset, compared with six metrics, plus a Streamlit app ("Bean Sorter Lab") that loads the saved models and evaluates them on an uploaded test CSV.

## a. Problem Statement

Predict which of seven dry bean varieties a grain belongs to, using 16 shape measurements taken from a photograph of the grain. This is a **multiclass classification** problem. The goal is to train five different classifiers on the same data, compare them on the same held-out test set with Accuracy, AUC, Precision, Recall, F1 and MCC, and offer an interactive app to inspect them.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository – *Dry Bean Dataset* (Koklu & Ozkan, 2020) – <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
- **Size:** 13,611 rows and 16 features + 1 target in the original file. 68 fully duplicated rows were found and removed **before** splitting, leaving **13,543 rows**.
- **Missing values:** 0 in every column.
- **Target:** `Class`, 7 classes.
- **Split:** stratified 80/20 → 10,834 training rows, 2,709 test rows (`test_data.csv`).

| Feature (name as in the file) | Meaning |
|---|---|
| Area | Number of pixels inside the bean's boundary |
| Perimeter | Length of the bean's border |
| MajorAxisLength | Length of the longest line that can be drawn across the bean |
| MinorAxisLength | Longest line perpendicular to the major axis |
| AspectRation | Ratio between major and minor axis length (the spelling is from the original dataset) |
| Eccentricity | Eccentricity of the ellipse that has the same moments as the bean region |
| ConvexArea | Pixels in the smallest convex polygon that contains the bean |
| EquivDiameter | Diameter of a circle with the same area as the bean |
| Extent | Ratio of bounding-box pixels to bean area |
| Solidity | Ratio of convex-shell pixels to bean pixels (convexity) |
| roundness | 4·π·Area / Perimeter² |
| Compactness | EquivDiameter / MajorAxisLength |
| ShapeFactor1 – ShapeFactor4 | Four further shape descriptors (the dataset documentation does not define them in more detail) |

**Class counts after removing duplicates (13,543 rows):**

| Class | Rows |
|---|---|
| DERMASON | 3,546 |
| SIRA | 2,636 |
| SEKER | 2,027 |
| HOROZ | 1,860 |
| CALI | 1,630 |
| BARBUNYA | 1,322 |
| BOMBAY | 522 |

The classes are imbalanced (BOMBAY is the smallest), which is why a stratified split and macro averaging are used.

## c. Links

- **GitHub repository:** `<ADD GITHUB LINK HERE>`
- **Live Streamlit app:** `<ADD STREAMLIT LINK HERE>`

## d. Models Used

All models are scikit-learn `Pipeline` objects. Where a model needs scaled inputs, a `StandardScaler` is the first pipeline step, so it is fitted on training rows only (no data leakage). `RANDOM_SEED = 7` is used everywhere.

- **Logistic Regression** – a linear model that gives each class a score from a weighted sum of the features and converts the scores to probabilities (`max_iter=2000`, scaled inputs).
- **Decision Tree** – learns a series of yes/no questions on feature thresholds. `max_depth=8` limits how deep the tree grows so it does not memorise the training data. No scaling needed.
- **K-Nearest Neighbor (KNN)** – classifies a bean by majority vote of its 7 closest training beans in scaled feature space (`n_neighbors=7`: odd to reduce ties, and neither too noisy nor too smooth).
- **Naive Bayes (Gaussian)** – uses Bayes' rule and assumes each feature is normally distributed and independent within a class. Very fast, but many bean measurements are strongly correlated, which breaks the independence assumption.
- **Support Vector Machine (SVM)** – an RBF-kernel SVC that finds a curved boundary with the widest margin between classes (scaled inputs). `probability=True` is switched on so AUC can be calculated; this makes training slower.

### Comparison table (test set, 2,709 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9232 | 0.9945 | 0.9360 | 0.9356 | 0.9357 | 0.9072 |
| Decision Tree | 0.9155 | 0.9801 | 0.9292 | 0.9244 | 0.9267 | 0.8977 |
| KNN | 0.9236 | 0.9873 | 0.9379 | 0.9354 | 0.9365 | 0.9076 |
| Naive Bayes | 0.7619 | 0.9665 | 0.7660 | 0.7632 | 0.7622 | 0.7125 |
| Support Vector Machine | 0.9288 | 0.9956 | 0.9424 | 0.9399 | 0.9411 | 0.9138 |

(Copied from `results/metrics_comparison.md`, produced by `train_models.py`.)

**Note on averaging.** The target has 7 classes, so Precision, Recall and F1 use **macro** averaging: the metric is computed for each class separately and then averaged with equal weight, so the small BOMBAY class counts as much as the big DERMASON class. **AUC** is computed **one-vs-rest** (each class against all the others) from the predicted probabilities, then macro averaged.

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TODO – write in my own words |
| Decision Tree | TODO – write in my own words |
| KNN | TODO – write in my own words |
| Naive Bayes | TODO – write in my own words |
| Support Vector Machine | TODO – write in my own words |
| Overall, Winner for your dataset? | TODO – write in my own words |

## Project Structure

```
├── app.py                  # Streamlit app (loads saved models, never trains)
├── train_models.py         # trains, evaluates and saves all 5 models
├── explore_data.py         # dataset exploration + creates data/dry_bean.csv
├── requirements.txt        # pinned dependencies
├── README.md
├── test_data.csv           # held-out 20% test split (features + Class)
├── .gitignore
├── .streamlit/config.toml  # custom colour theme
├── data/
│   ├── raw/Dry_Bean_Dataset.xlsx
│   └── dry_bean.csv        # cleaned copy (duplicates removed)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── label_encoder.joblib    # maps class names <-> numbers
│   └── feature_columns.json    # feature names and order the models expect
└── results/
    ├── metrics_comparison.csv
    └── metrics_comparison.md
```

## How to Run Locally

Needs **Python 3.12 or newer** (the pinned numpy 2.5.4 requires it).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python explore_data.py     # optional: prints dataset facts, rebuilds data/dry_bean.csv
```

```bash
python train_models.py     # trains and saves the 5 models, prints the comparison table
```

```bash
streamlit run app.py       # opens the app in the browser
```

## How to Use the App

1. In the sidebar, click **Download sample test data** (or prepare your own CSV with the 16 feature columns and a `Class` column).
2. Upload the CSV with the sidebar uploader. Only test data is uploaded; the models are never retrained.
3. Pick a model from the dropdown.
4. Use the tabs: **Metrics** (6 metric cards), **Confusion matrix** (heatmap), **Classification report**, **Compare all 5 models** (table with best values highlighted + bar chart) and **Data preview**.
5. If the file is missing columns, has unknown labels, or does not contain all 7 classes, the app shows a clear message instead of crashing (AUC is shown as N/A when not all 7 classes are present).
