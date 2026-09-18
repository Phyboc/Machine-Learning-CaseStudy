# Machine Learning Case Study — Classification, Regression & Clustering Tracks

## Overview

This repository implements a full end-to-end **Machine Learning Case Study** across three learning paradigms, with a
shared emphasis on **leak-free preprocessing, reproducibility (`random_state=42` throughout), and rigorous model
benchmarking**:

1. **Classification Track** — Binary fraud detection on the *Fraud Detection Bank Dataset* (10 algorithms: 5 classical + 5 ensemble/MLP, with hyperparameter tuning and a champion model selection).
2. **Regression Track** — Critical temperature prediction on the *UCI Superconductivity Dataset* (10 regressors: linear/regularized baselines through tree ensembles, with cross-validated tuning).
3. **Clustering Track** — Customer segmentation on the *Airplane Customer Satisfaction Dataset* (K-Means with a four-criterion model-selection sweep, cluster profiling, post-hoc label validation, and PCA / t-SNE projections).

All trained artifacts and evaluation reports are exported to `results/`.

---

## Repository Structure

```
ML-Case-Study/
│
├── data/
│   ├── raw/
│   │   ├── fraud_detection_bank_dataset_raw.csv   # Fraud dataset (20,468 rows x 114 cols)
│   │   ├── train.csv                              # UCI Superconductivity dataset (21,263 rows x 82 cols)
│   │   └── airplane customer satisfaction.csv     # Airline satisfaction survey (103,904 rows x 25 cols)
│   │
│   └── processed/
│       ├── classification/
│       │   ├── X_train.csv     # Training features      (16,234 rows x 96 cols)
│       │   ├── X_test.csv      # Testing features       ( 4,059 rows x 96 cols)
│       │   ├── y_train.csv     # Classification target  (16,234 rows x 1 col)
│       │   └── y_test.csv      # Classification target  ( 4,059 rows x 1 col)
│       │
│       └── clustering/
│           ├── X.csv                      # Standardised clustering matrix (103,904 rows x 23 cols)
│           ├── satisfaction.csv           # Post-hoc validation label  (103,904 rows x 1 col)
│           └── cluster_assignments.csv    # Deployed K-Means segment per customer (103,904 rows x 1 col)
│
├── src/
│   ├── classification/
│   │   └── classification.ipynb   # Full 10-algorithm classification pipeline
│   ├── regression/
│   │   ├── regression1.ipynb      # Regression track (final): feature engineering + tuned models
│   │   └── regression.ipynb       # Regression track (earlier iteration)
│   └── clustering/
│       └── clustering.ipynb       # Merged clustering track: preprocessing + EDA + K-Means + PCA/t-SNE
│
├── results/
│   ├── classification/            # Model reports, benchmark CSVs, and charts
│   └── clustering/                # Cluster benchmark, evaluation report, and 16 diagnostic figures
│
├── .gitignore                     # Python / Jupyter artifacts
├── README.md                      # Project documentation
└── requirements.txt               # Environment dependencies
```

---

## Datasets

| Dataset | Task | Raw Shape | Target | Class/Value Profile |
|---|---|---|---|---|
| **Fraud Detection Bank Dataset** | Classification | 20,468 × 114 | `targets` (binary fraud flag) | ~26.5% fraud (imbalanced) |
| **UCI Superconductivity Dataset** | Regression | 21,263 × 82 | `critical_temp` (Kelvin) | Right-skewed continuous |
| **Airplane Customer Satisfaction** | Clustering | 103,904 × 25 | *(unsupervised — `satisfaction` held out post-hoc)* | 56.7% neutral/dissatisfied, 43.3% satisfied |

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

## Track 3 — Clustering (Airplane Customer Satisfaction Segmentation)

**Notebook:** `src/clustering/clustering.ipynb` — a single self-contained pipeline that supersedes the two previously
separate notebooks (`clustering_preprocessing.ipynb` + `clustering.ipynb`), structured like the Classification and
Regression tracks.

### Pipeline

1. **Raw Audit** — 103,904 × 25 audit: shape, dtypes, memory footprint, descriptive statistics.
2. **Data Quality Audit** — 310 missing cells (0.298%, one column), 0 duplicate rows, 0 constant columns, cardinality profile.
3. **Feature Identification** — 4 nominal categorical / 14 ordinal 0–5 survey ratings / 4 continuous numerical; the `satisfaction` target is **excluded from the feature space** and retained as a post-hoc label.
4. **EDA (six figures, each with its own inference, insight and commentary)** — target balance and nominal-category profiles; continuous feature distributions raw vs `log1p`; IQR outlier screening; the 14 service-rating distributions split by satisfaction; service-rating satisfaction gaps; and the feature correlation structure.
5. **Data Cleaning** — de-duplication verification (no-op, 0 duplicates found) and **median imputation** (0.0 min) of the 310 missing `Arrival Delay in Minutes` values, with all 103,904 customers retained.
6. **Skewness Treatment** — `log1p` compression of the three features breaching the |skew| > 1.0 threshold (`Flight Distance` 1.11 → −0.20, `Departure Delay` 6.73 → 0.92, `Arrival Delay` 6.61 → 0.88). No outlier rows are deleted.
7. **Encoding** — one-hot with `drop_first=True`, taking the matrix from 22 to 23 features.
8. **Scaling** — `StandardScaler` fitted on the full population (correct for unsupervised learning: there is no held-out target to leak).
9. **Validation** — eight structural gates (numeric, finite, non-constant, no identifier leakage, no target leakage, row-count alignment).
10. **Model Selection** — K-Means `k = 2…10` scored on **Silhouette**, **Davies-Bouldin**, **Calinski-Harabasz** and inertia, plus a marginal-inertia elbow analysis.
11. **Deployed Model** — K-Means `k = 3`, with cluster profiling, post-hoc validation, PCA and t-SNE projections, and CSV/report export.

### Model-Selection Sweep (all three mandatory indices, every `k`)

| k | Inertia | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| 2 | 2,092,596 | **0.1232** | 2.6087 | **14,756.5** |
| **3** | **1,965,481** | 0.1108 | 2.7640 | 11,215.2 |
| 4 | 1,861,225 | 0.0909 | 2.6483 | 9,835.5 |
| 5 | 1,770,992 | 0.0951 | 2.4512 | 9,075.8 |
| 6 | 1,685,451 | 0.1029 | 2.2626 | 8,683.7 |
| 7 | 1,616,661 | 0.1096 | **2.1351** | 8,281.1 |
| 8 | 1,573,018 | 0.1091 | 2.1420 | 7,706.7 |
| 9 | 1,533,010 | 0.1110 | 2.1425 | 7,258.2 |
| 10 | 1,501,589 | 0.1063 | 2.2133 | 6,828.3 |

| Metric | Deployed model (K-Means, k = 3) |
|---|---|
| **Silhouette Score** | **0.1108** (reproducible 10,000-customer sample) |
| **Davies-Bouldin Index** | **2.7640** (full 103,904-customer matrix) |
| **Calinski-Harabasz Index** | **11,215.2** (full matrix) |
| Inertia | 1,965,481 |

### The Three Customer Segments

| Segment | Size | Characterisation | Satisfied |
|---|---|---|---|
| **Cluster 1** | 43,653 (42.0%) | *Satisfied advocates* — above average on all 14 service dimensions, Business class, business travel, longest flights | 74.4% |
| **Cluster 0** | 31,345 (30.2%) | *Inconsistent service experience* — food, seating and cleanliness fine; crew service, leg room, baggage and inflight service well below average | 25.4% |
| **Cluster 2** | 28,906 (27.8%) | *Dissatisfied at-risk* — below average on nearly every dimension, Economy class, personal travel, disloyal, youngest | 15.9% |

### Key Observations

- **No single `k` is optimal across all three indices.** Silhouette and Calinski-Harabasz both peak at `k = 2`, Davies-Bouldin at `k = 7`, and the elbow (the largest marginal inertia reduction, −127,115 = −6.07%, with strictly decaying gains thereafter) sits at `k = 3`. **`k = 3` is selected as the elbow with a near-optimal silhouette (gap 0.0124) and second-best CH**, explicitly accepting the worst Davies-Bouldin in the sweep — the trade-off is documented in the notebook, the report and the benchmark CSV rather than presented as unanimous.
- **Absolute separation is weak, as pre-registered.** A silhouette of 0.1108 would be poor on well-separated data; on bounded 0–5 ordinal survey ratings it is expected, and EDA 3 predicts this *before* any metric is computed. The PCA projection makes the cause visible: the first two components retain only 27.91% of variance.
- **The post-hoc validation is strong.** Although `satisfaction` was excluded from the feature matrix, the fitted clusters separate it by **58.4 percentage points** (74.4% / 25.4% / 15.9% vs a 43.3% baseline; χ² = 30,060.1, p ≪ 0.001, Cramér's V = 0.538).
- **The segmentation is about service experience, not demographics.** The strongest discriminators are `Inflight entertainment` (1.71 SD spread), `Seat comfort` (1.66), `Cleanliness` (1.59), `Inflight service` (1.52) and `Baggage handling` (1.49), while `Gate location` (0.06), the delay features (0.12 and 0.16) and `Gender_Male` (0.20) contribute almost nothing — the direct payoff of the `log1p` + scaling treatment, without which delays and age would have dominated the distance metric.
- **Cluster 0 is the most actionable segment:** its members rate food, seating and cleanliness at or above average yet rate crew and ground service far below it — a service-inconsistency pattern that neither a 2-cluster solution nor any single-variable analysis surfaces.

---

## Methodology Highlights

- **Reproducibility** — `random_state=42` enforced across splitting, model initialization, CV folds, tuning searches, K-Means initialisation and every sampling step. The clustering pipeline was verified by deleting all 21 output artifacts and re-executing the notebook from scratch: all 21 were regenerated **byte-identical**.
- **Leakage Prevention** — All dataset-dependent statistics (scaler parameters, transformation eligibility) are fitted on training data only in the supervised tracks; in the clustering track the scaler and imputer are fitted on the full population (there is no held-out target), while the `satisfaction` label is excluded from the feature matrix, asserted absent, and used only for post-hoc validation.
- **Class-Imbalance Awareness** — Classification reports macro/weighted metrics alongside accuracy; per-class precision/recall for the minority fraud class is analyzed.
- **Cross-Validation Everywhere** — No single split is trusted: 5-fold CV backs every regression result, and stratified CV gates champion selection in classification.
- **Evidence-Based Model Selection** — The clustering track reports all three internal indices for every candidate `k` and makes the `k` choice with an explicit, written justification of the trade-off accepted.
- **Sampling Disclosed Where Necessary** — Silhouette (O(N²)) is computed on a fixed 10,000-customer sample assigned by the full-data model; t-SNE is computed on a fixed 5,000-customer sample. Both are seeded and documented in-notebook, alongside the variance retained.

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

## Results Artifacts (`results/clustering/`)

| Artifact | Description |
|---|---|
| `benchmark_summary.csv` | One row per candidate `k` (2–10) with all three indices, inertia and a `Selected Model` flag |
| `clustering_report.txt` | Written evaluation report: pipeline, selection rationale, deployed metrics, cluster distribution, post-hoc validation, top discriminators |
| `kmeans_selection_curves.png` | Four-panel selection curves — elbow, silhouette, Davies-Bouldin, Calinski-Harabasz |
| `kmeans_marginal_inertia.png` | Marginal inertia reduction per added cluster (the elbow evidence) |
| `kmeans_cluster_distribution.png` | Cluster sizes and per-cluster cohesion |
| `cluster_profile.png` | Centroid heatmap + top discriminating features |
| `cluster_satisfaction_posthoc.png` | Post-hoc satisfaction by cluster + alternative-`k` comparison |
| `pca_cluster_scatter.png` | **PCA-reduced 2D scatter with cluster colour-coding** and projected centroids |
| `pca_component_analysis.png` | PCA explained variance (10 components) and component loadings |
| `tsne_cluster_projection.png` | **t-SNE embedding** of clusters, side-by-side with the held-out satisfaction label |
| `data_quality_audit.png` | Missing-value and cardinality audit (EDA) |
| `target_and_categorical_profile.png` | Target balance and nominal-category satisfaction profiles (EDA) |
| `numerical_feature_distributions.png` | Continuous features, raw vs `log1p` (EDA) |
| `delay_outlier_boxplots.png` | IQR outlier screening (EDA) |
| `service_rating_distributions.png` | The 14 ordinal service-rating distributions by satisfaction (EDA) |
| `feature_correlation_heatmap.png` | Service-rating gaps and feature correlation structure (EDA) |
| `skewness_transformation.png` | Skewness before vs after `log1p` compression |
| `feature_scaling_comparison.png` | Feature ranges before vs after standardisation |

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

4. **Clustering track** (run from `src/clustering/`; the notebook also resolves the repository root automatically, so it works from the repository root too):

   ```bash
   cd src/clustering
   python -m jupyter nbconvert --to notebook --execute --inplace clustering.ipynb
   ```

   End-to-end runtime is roughly three minutes on a laptop CPU. This regenerates `data/processed/clustering/X.csv`, `satisfaction.csv` and `cluster_assignments.csv`, plus all artifacts in `results/clustering/`.

Or open the notebooks in Jupyter Notebook / VS Code and run all cells.

---

## Dependencies

- Python ≥ 3.10
- pandas ≥ 2.0, numpy ≥ 1.24
- scikit-learn ≥ 1.2, scipy ≥ 1.10
- matplotlib ≥ 3.7, seaborn ≥ 0.12
- jupyter ≥ 1.0
