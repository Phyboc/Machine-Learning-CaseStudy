# Fraud Detection Bank Dataset — ML Case Study: EDA & Preprocessing Architecture

## Overview
This repository implements a rigorous, leak-free **Exploratory Data Analysis (EDA) and Data Preprocessing Pipeline** for a Fraud Detection Bank Dataset. 

The preprocessing architecture is modularly designed to support downstream model training across three distinct machine learning paradigms:
1. **Classification** (Predicting binary fraud flag `targets`).
2. **Regression** (Predicting continuous financial/transaction target `col_67`).
3. **Clustering** (Unsupervised customer/transaction segmentation matrix `X`).

---

## Technical Architecture & Pipeline Order

```
ML-Case-Study/
│
├── data/
│   ├── raw/
│   │   └── fraud_detection_bank_dataset_raw.csv   # Untouched original dataset (20,468 rows x 114 cols)
│   │
│   └── processed/
│       ├── classification/
│       │   ├── X_train.csv     # Scaled training features (16,234 rows x 96 cols)
│       │   ├── X_test.csv      # Scaled testing features  (4,059 rows x 96 cols)
│       │   ├── y_train.csv     # Classification target     (16,234 rows x 1 col)
│       │   └── y_test.csv      # Classification target     (4,059 rows x 1 col)
│       │
│       ├── regression/
│       │   ├── X_train.csv     # Scaled training features (16,231 rows x 95 cols)
│       │   ├── X_test.csv      # Scaled testing features  (4,058 rows x 95 cols)
│       │   ├── y_train.csv     # Regression target        (16,231 rows x 1 col)
│       │   └── y_test.csv      # Regression target        (4,058 rows x 1 col)
│       │
│       └── clustering/
│           └── X.csv           # Scaled feature matrix    (20,289 rows x 96 cols)
│
├── src/
│   └── eda_preprocessing/
│       └── eda_preprocessing.ipynb   # Main notebook following 13-stage order
│
├── results/
│   ├── classification/         # Downstream classification model artifacts
│   ├── regression/             # Downstream regression model artifacts
│   └── clustering/             # Downstream clustering model artifacts
│
├── README.md                   # Project documentation
└── requirements.txt            # Environment dependencies
```

---

## Key Data Quality & Cleaning Actions

1. **Raw Data Immutability**:
   - `data/raw/fraud_detection_bank_dataset_raw.csv` remains strictly untouched. Verified via SHA-256 hash assertions before and after pipeline execution.

2. **Common Dataset Cleaning**:
   - **Row ID Column (`Unnamed: 0`)**: Dropped as an arbitrary index artifact.
   - **15 Constant Columns**: Dropped (`col_8`, `col_9`, `col_10`, `col_11`, `col_12`, `col_18`, `col_19`, `col_20`, `col_21`, `col_35`, `col_51`, `col_52`, `col_53`, `col_70`, `col_71`) due to 100% zero variance.
   - **1 Duplicate Feature Column**: Dropped (`col_7`, which is an exact duplicate of `col_0`).
   - **175 Duplicate Rows**: Dropped during common cleaning to prevent **train-test row overlap data leakage**. Clean dataset size: **20,293 unique rows**.
   - **4 Label-Only Duplicate Rows (Regression / Clustering)**: Regression and clustering drop `targets` from the feature space before deduplicating, so the 4 records that differ **only** in the classification label collapse into duplicate feature rows and are removed. This is why regression and clustering have **20,289 feature-space rows** while classification has **20,293** — a deliberate, data-verified difference, not an inconsistency.

3. **Strict Zero-Leakage Pipeline**:
   - **Train/Test Splitting**: Performed before fitting any dataset-dependent parameters (80/20 train/test split, `random_state=42`).
   - **Stratified Split for Classification**: `stratify=y_cls` preserves the ~26.57% fraud class ratio across both `y_train` and `y_test`.
   - **Feature Transformation Selection**: `log1p` is applied only to **non-binary, non-negative, right-skewed** features (`skewness > 1.0`). Binary flag features (≤ 2 unique values, e.g. 0/1) are never log-transformed. Eligibility is evaluated **strictly on `X_train`** for classification and regression, and on the full feature matrix for clustering. Executed result: 53 features transformed for classification, 52 for regression, 53 for clustering.
   - **Scaling**: `StandardScaler` parameters ($\mu, \sigma$) are fitted on `X_train` only and applied to `X_test`.
   - **Unsupervised Clustering**: Evaluated on the full clean feature matrix $X_{clu}$ without train/test splitting, retaining continuous feature `col_67` while dropping the ground truth label `targets`.

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Preprocessing Notebook**:
   ```bash
   python -m jupyter nbconvert --to notebook --execute --inplace src/eda_preprocessing/eda_preprocessing.ipynb
   ```
   Or run all cells in `src/eda_preprocessing/eda_preprocessing.ipynb` using Jupyter Notebook / VS Code.

---

## Verification & Integrity Checks

The notebook executes automated assertions at Stage 13 verifying:
- All 9 processed CSV files are created without NaNs, infs, or row index artifacts.
- Exact feature width matching: Classification (96 cols), Regression (95 cols), Clustering (96 cols).
- Zero row intersection between `X_train` and `X_test` (`len(pd.merge(X_test, X_train, how='inner')) == 0`).
- Unaltered raw dataset SHA-256 hash.
