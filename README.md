# Machine Learning Assignment 2 – Dry Bean Classification

In this project I train 5 classification models on the UCI Dry Bean dataset and compare them using 6 metrics. I also built a small Streamlit app. It loads the saved models and checks them on a test csv that the user uploads.

## a. Problem Statement

A bean can be one of 7 types. We have 16 measurements of its shape (size, roundness and so on) taken from a photo. The task is to predict the bean type from these measurements. This is a multiclass classification problem.

I train 5 different models on the same data, test them on the same test set, and compare Accuracy, AUC, Precision, Recall, F1 and MCC.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository, Dry Bean Dataset (Koklu and Ozkan, 2020) – <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
- **Rows:** 13,611 in the original file. 68 rows were exact duplicates, so I removed them before splitting. That leaves **13,543 rows**.
- **Features:** 16 (all numbers). **Target:** `Class` with 7 classes.
- **Missing values:** none, 0 in every column.
- **Train/test split:** 80/20, stratified. 10,834 rows for training and 2,709 rows for testing (`test_data.csv`).

| Feature (name in the file) | What it means |
|---|---|
| Area | Number of pixels inside the bean |
| Perimeter | Length of the bean's border |
| MajorAxisLength | Longest line that can be drawn across the bean |
| MinorAxisLength | Longest line across the bean at right angles to the major axis |
| AspectRation | Major axis length divided by minor axis length (the spelling is from the original file) |
| Eccentricity | How stretched the bean is, like an ellipse with the same moments |
| ConvexArea | Pixels in the smallest convex shape that fits around the bean |
| EquivDiameter | Diameter of a circle that has the same area as the bean |
| Extent | Bean area compared with its bounding box |
| Solidity | Convex shape pixels compared with bean pixels |
| roundness | 4 × pi × Area / Perimeter squared |
| Compactness | EquivDiameter / MajorAxisLength |
| ShapeFactor1 to ShapeFactor4 | Four more shape numbers (the dataset notes do not explain them further) |

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

The classes are not equal in size. BOMBAY is the smallest. This is why I used a stratified split and macro averaging.

## c. Links

- **GitHub repository:** https://github.com/Kavitabhalerao97/ml-assignment-2-dry-bean
- **Live Streamlit app:** https://ml-assignment-2-dry-bean-3sdwzb2sa3yqt4dqhpuhha.streamlit.app/

## d. Models Used

Every model is a scikit-learn `Pipeline`. For the models that need scaling, a `StandardScaler` is the first step. The pipeline fits the scaler only on the training rows, so the test data never leaks into training. I used `RANDOM_SEED = 7` everywhere.

- **Logistic Regression** – a simple linear model. It adds up the features with weights and turns the result into class probabilities. Uses scaled data, `max_iter=2000`.
- **Decision Tree** – asks a chain of yes/no questions about the features. I set `max_depth=8` so it does not just memorise the training data. Scaling is not needed.
- **K-Nearest Neighbor (KNN)** – looks at the 7 closest training beans and takes a vote. I chose `n_neighbors=7` because it is an odd number (fewer ties) and not too small or too big. Uses scaled data.
- **Naive Bayes (Gaussian)** – uses Bayes' rule and assumes every feature is normally distributed and independent of the others. It is very fast, but many bean features are closely related, so the independence idea does not fit well.
- **Support Vector Machine (SVM)** – RBF kernel. It finds a curved boundary with the biggest gap between classes. Uses scaled data. I turned on `probability=True` so that AUC can be calculated, which makes it slower.

### Comparison table (test set, 2,709 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9232 | 0.9945 | 0.9360 | 0.9356 | 0.9357 | 0.9072 |
| Decision Tree | 0.9155 | 0.9801 | 0.9292 | 0.9244 | 0.9267 | 0.8977 |
| KNN | 0.9236 | 0.9873 | 0.9379 | 0.9354 | 0.9365 | 0.9076 |
| Naive Bayes | 0.7619 | 0.9665 | 0.7660 | 0.7632 | 0.7622 | 0.7125 |
| Support Vector Machine | 0.9288 | 0.9956 | 0.9424 | 0.9399 | 0.9411 | 0.9138 |

(These numbers come from `results/metrics_comparison.md`, which `train_models.py` creates.)

**About the averaging:** there are 7 classes, so Precision, Recall and F1 are **macro** averages. The score is worked out for each class and then averaged, so each class counts the same, even the small BOMBAY class. **AUC** is **one-vs-rest**: each class is compared against all the other classes using the predicted probabilities, and then the 7 values are averaged.

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed well, with 92.32% accuracy. Its results are close to the other top models. This shows that even a simple model can give good results for this dataset. |
| Decision Tree | Decision Tree achieved 91.55% accuracy. Its performance is good, but slightly lower than Logistic Regression, KNN and SVM. It was able to classify most of the data correctly. |
| KNN | KNN achieved 92.36% accuracy, which is almost the same as Logistic Regression. Scaling the data helped KNN compare the features properly and make better predictions. |
| Naive Bayes | Naive Bayes achieved 76.19% accuracy, which is lower than all the other models. It may be because this model assumes that the features are independent, but some features in this dataset may be related to each other. |
| Support Vector Machine | SVM gave the best results, with 92.88% accuracy. It also achieved the highest scores in all the other evaluation metrics. It seems to separate the different classes better, even when some beans have similar features. |
| Overall, Winner for your dataset? | I would choose Support Vector Machine as the winner because it performed best in all six evaluation metrics. Logistic Regression and KNN also gave very close results, but SVM performed slightly better overall. |

## Project Structure

```
├── app.py                  # Streamlit app (only loads saved models)
├── train_models.py         # trains, tests and saves the 5 models
├── explore_data.py         # looks at the data and makes data/dry_bean.csv
├── requirements.txt        # packages with fixed versions
├── README.md
├── test_data.csv           # the 20% test split (features + Class)
├── .gitignore
├── .streamlit/config.toml  # app colours
├── data/
│   ├── raw/Dry_Bean_Dataset.xlsx
│   └── dry_bean.csv        # clean copy (duplicates removed)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── label_encoder.joblib    # class names <-> numbers
│   └── feature_columns.json    # feature names and their order
└── results/
    ├── metrics_comparison.csv
    └── metrics_comparison.md
```

## How to Run Locally

You need Python 3.12 or newer (the fixed numpy version needs it).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python explore_data.py     # optional: prints facts about the data and makes data/dry_bean.csv
```

```bash
python train_models.py     # trains and saves the 5 models and prints the comparison table
```

```bash
streamlit run app.py       # opens the app in the browser
```

## How to Use the App

1. In the sidebar, click **Download sample test data**. You can also use your own csv, but it needs the 16 feature columns and a `Class` column.
2. Upload the csv in the sidebar. Only test data is uploaded. The models are never trained again.
3. Pick a model from the dropdown.
4. Look at the tabs: **Metrics** (6 cards), **Confusion matrix**, **Classification report**, **Compare all 5 models** (table plus bar chart, best value highlighted) and **Data preview**.
5. If the file has missing columns, unknown labels, or does not contain all 7 classes, the app shows a message and does not crash. AUC shows N/A when not all 7 classes are in the file.
