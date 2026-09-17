# Machine Learning Case Study — Classification & Regression Tracks

## Overview

This repository implements a full end-to-end **Machine Learning Case Study** across two supervised learning paradigms, with a shared emphasis on **leak-free preprocessing, reproducibility (`random_state=42` throughout), and rigorous model benchmarking**:

1. **Classification Track** — Binary fraud detection on the *Fraud Detection Bank Dataset* (10 algorithms: 5 classical + 5 ensemble/MLP, with hyperparameter tuning and a champion model selection).
2. **Regression Track** — Critical temperature prediction on the *UCI Superconductivity Dataset* (10 regressors: linear/regularized baselines through tree ensembles, with cross-validated tuning).

All trained artifacts and evaluation reports are exported to `results/`.

---

## Repository Structure

```
ML-Case-Study/
│
├── data/
│   ├── raw/
│   │   ├── fraud_detection_bank_dataset_raw.csv   # Fraud dataset (20,468 rows x 114 cols)
│   │   └── train.csv                              # UCI Superconductivity dataset (21,263 rows x 82 cols)
│   │
│   └── processed/
│       └── classification/
│           ├── X_train.csv     # Training features      (16,234 rows x 96 cols)
│           ├── X_test.csv      # Testing features       ( 4,059 rows x 96 cols)
│           ├── y_train.csv     # Classification target  (16,234 rows x 1 col)
│           └── y_test.csv      # Classification target  ( 4,059 rows x 1 col)
│
├── src/
│   ├── classification/
│   │   └── classification.ipynb   # Full 10-algorithm classification pipeline
│   └── regression/
│       ├── regression1.ipynb      # Regression track (final): feature engineering + tuned models
│       └── regression.ipynb       # Regression track (earlier iteration)
│
├── results/
│   └── classification/            # Model reports, benchmark CSVs, and charts
│
├── README.md                      # Project documentation
└── requirements.txt               # Environment dependencies
```

---

## Datasets

| Dataset | Task | Raw Shape | Target | Class/Value Profile |
|---|---|---|---|---|
| **Fraud Detection Bank Dataset** | Classification | 20,468 × 114 | `targets` (binary fraud flag) | ~26.5% fraud (imbalanced) |
| **UCI Superconductivity Dataset** | Regression | 21,263 × 82 | `critical_temp` (Kelvin) | Right-skewed continuous |

---

## Track 1 — Classification (Fraud Detection)

**Notebook:** `src/classification/classification.ipynb`

### Pipeline

1. **Data Loading** — Prefers the preprocessed, leak-free splits in `data/processed/classification/`; if absent, falls back to the raw CSV with on-the-fly cleaning, stratified splitting, and scaling.
2. **EDA** — Target distribution, correlation heatmap, and feature behavior analysis.
3. **Part A / Review 1 (Algorithms 1–5)** — Logistic Regression (odds-ratio interpretation), KNN (tuned over `k ∈ [3, 5, 7, 9, 11]`), Gaussian Naive Bayes, Decision Tree (tuned `max_depth`, structure visualized), SVM (RBF kernel).
4. **Part B / Review 2 (Algorithms 6–10)** — Random Forest, AdaBoost, Gradient Boosting, Bagging, and MLP Neural Network.
5. **Champion Selection** — Top-2 models from the consolidated benchmark undergo stratified cross-validation + `GridSearchCV`; the winner is reported with a full confusion matrix and classification report.

### Consolidated Benchmark (Test Set)

| Algorithm | Accuracy | Precision | Recall | F1 (Weighted) | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** ⭐ | **0.9350*** | **0.8896*** | **0.8615*** | **0.9346*** | **0.9779*** |
| Support Vector Machine | 0.9256 | 0.8886 | 0.8225 | 0.9247 | 0.9687 |
| Bagging (Decision Tree) | 0.9214 | 0.8527 | 0.8504 | 0.9214 | 0.9694 |
| Random Forest | 0.9219 | 0.9103 | 0.7825 | 0.9199 | 0.9675 |
| Logistic Regression | 0.9177 | 0.8786 | 0.8002 | 0.9164 | 0.9580 |
| Decision Tree | 0.9165 | 0.8447 | 0.8392 | 0.9164 | 0.9505 |
| K-Nearest Neighbors | 0.9103 | 0.8511 | 0.8020 | 0.9095 | 0.9508 |
| MLP Neural Network | 0.9071 | 0.8276 | 0.8206 | 0.9070 | 0.9465 |
| AdaBoost | 0.8810 | 0.8603 | 0.6580 | 0.8755 | 0.9188 |
| Naive Bayes (Gaussian) | 0.5627 | 0.3713 | 0.9368 | 0.5744 | 0.8655 |

\* Champion model after hyperparameter tuning (`learning_rate=0.1`, `max_depth=5`, `n_estimators=150`); all other rows are pre-tuning benchmark scores.

### Key Observations

- **Gradient Boosting** wins on every metric after tuning (Accuracy 0.9350, ROC-AUC 0.9779).
- **Naive Bayes** illustrates the conditional-independence violation on correlated banking features: high recall (0.9368) but very poor precision (0.3713).
- **Random Forest** has the best precision (0.9103) but under-recalls fraud, reflecting the class imbalance (~26.5% fraud rate).
- Scaling is essential for distance-based models (KNN/SVM): features span vastly different raw magnitudes.

---

## Track 2 — Regression (Superconductor Critical Temperature)

**Notebook:** `src/regression/regression1.ipynb` (final; `regression.ipynb` is an earlier iteration)

### Pipeline

1. **EDA** — Target distribution, feature boxplots, feature-vs-target scatter plots.
2. **Cleaning & Multicollinearity Filtering** — Duplicate row removal; highly correlated feature pairs dropped at a **|r| > 0.95** threshold.
3. **Feature Engineering** — Derived physico-chemical features to strengthen signal for the linear baselines.
4. **Split & Scaling** — 80/20 train/test split, `random_state=42`; `StandardScaler` fitted on training data only.
5. **Standardized Evaluation** — Every model reports Test R²/RMSE/MAE plus **5-fold cross-validated R²** (KFold, shuffled), with per-model hyperparameter tuning.

### Model Comparison (Test Set)

| # | Model | Test R² | RMSE | MAE | 5-Fold CV R² |
|---|---|---|---|---|---|
| 1 | Linear Regression (baseline) | 0.7153 | 18.307 | 13.548 | 0.7085 |
| 2 | Ridge (α = 10) | 0.7154 | 18.304 | 13.549 | 0.7085 |
| 3 | Lasso (α = 0.01) | 0.7153 | 18.308 | 13.554 | 0.7083 |
| 4 | ElasticNet (α = 0.001, l1_ratio = 0.5) | 0.7154 | 18.304 | 13.549 | 0.7085 |
| 5 | Polynomial Regression (degree 2) | 0.8411 | 13.676 | 9.017 | 0.8096 |
| 6 | Decision Tree (max_depth = 20) | 0.8810 | 11.834 | 6.091 | 0.8656 |
| 7 | **Random Forest (n_estimators = 100)** ⭐ | **0.9196** | **9.730** | 5.655 | **0.9163** |
| 8 | Gradient Boosting (learning_rate = 0.2) | 0.9186 | 9.787 | 5.740 | 0.9126 |
| 9 | SVR (C = 100, RBF kernel) | 0.8495 | 13.311 | 7.670 | 0.8437 |
| 10 | KNN Regressor (k = 3) | 0.9075 | 10.432 | 5.630 | 0.8985 |

### Key Observations

- **Random Forest** is the champion (R² 0.9196, RMSE ≈ 9.73 K), with **Gradient Boosting** a close second — non-linear tree ensembles dominate this dataset.
- Regularized linear models (Ridge/Lasso/ElasticNet) add no lift over plain Linear Regression; the multicollinearity filter already removed most redundant features.
- A dedicated ablation shows KNN loses **0.0244 R²** without feature scaling — confirming the necessity of `StandardScaler` for distance-based learners.

---

## Methodology Highlights

- **Reproducibility** — `random_state=42` enforced across splitting, model initialization, CV folds, and tuning searches.
- **Leakage Prevention** — All dataset-dependent statistics (scaler parameters, transformation eligibility) are fitted on training data only; stratified splitting preserves the fraud class ratio.
- **Class-Imbalance Awareness** — Classification reports macro/weighted metrics alongside accuracy; per-class precision/recall for the minority fraud class is analyzed.
- **Cross-Validation Everywhere** — No single split is trusted: 5-fold CV backs every regression result, and stratified CV gates champion selection in classification.

---

## Results Artifacts (`results/classification/`)

| Artifact | Description |
|---|---|
| `champion_model_report.txt` | Champion (tuned Gradient Boosting) metrics + classification report |
| `champion_confusion_matrix.png` | Confusion matrix of the tuned champion |
| `consolidated_benchmark_summary.csv` | 10-algorithm benchmark table |
| `consolidated_roc_curves.png` | ROC curves for all Part B + top Part A models |
| `part_a_preliminary_summary.csv` | Review 1 (Part A) preliminary results |
| `part_a_confusion_matrices.png` | Confusion matrices for the 5 Part A classifiers |
| `correlation_heatmap.png` | Feature correlation heatmap (EDA) |
| `target_distribution.png` | Fraud vs. genuine class distribution (EDA) |
| `logistic_regression_odds_ratios.png` | Odds-ratio interpretation chart |
| `decision_tree_structure.png` | Tuned decision tree structure visualization |
| `rf_feature_importance.png` | Random Forest feature importances |

---

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Classification track** (expects `data/processed/classification/` splits; auto-falls back to raw-data preprocessing if missing — run from the repository root):

   ```bash
   python -m jupyter nbconvert --to notebook --execute --inplace src/classification/classification.ipynb
   ```

3. **Regression track** (notebooks use relative paths — run from `src/regression/`):

   ```bash
   cd src/regression
   python -m jupyter nbconvert --to notebook --execute --inplace regression1.ipynb
   ```

   Or open the notebooks in Jupyter Notebook / VS Code and run all cells.

---

## Dependencies

- Python ≥ 3.10
- pandas ≥ 2.0, numpy ≥ 1.24
- scikit-learn ≥ 1.2, scipy ≥ 1.10
- matplotlib ≥ 3.7, seaborn ≥ 0.12
- jupyter ≥ 1.0
