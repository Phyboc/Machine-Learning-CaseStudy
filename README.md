<div align="center">

# Machine Learning Case Study

### Fraud Detection · Regression · Customer Segmentation

<p>
  <strong>An end-to-end machine learning portfolio project built around reproducible experimentation, leakage-safe preprocessing, model comparison, and an interactive Streamlit application.</strong>
</p>

<p>
[![Live Streamlit demo](https://shields.io)](https://streamlit.app)
</p>

<p>
  <a href="#-project-at-a-glance">At a glance</a> •
  <a href="#-repository-structure">Structure</a> •
  <a href="#-results">Results</a> •
  <a href="#-quick-start">Quick start</a> •
  <a href="#-interactive-application">Application</a>
</p>

</div>

---

## Project at a glance

This repository presents three complementary machine learning tracks using three real-world datasets:

| Track | Problem | Dataset | Approach |
|---|---|---|---|
| **Classification** | Detect fraudulent banking transactions | Fraud Detection Bank Dataset | 10 classifiers, stratified evaluation, tuning, champion selection |
| **Regression** | Predict superconducting critical temperature | UCI Superconductivity Dataset | 10 regressors, feature engineering, cross-validation, tuning |
| **Clustering** | Segment airline customers by service experience | Airplane Customer Satisfaction Dataset | K-Means model selection, profiling, PCA/t-SNE, post-hoc validation |

The project is designed as a complete workflow rather than a collection of isolated notebooks:

> **Raw data → quality checks → preprocessing → modelling → evaluation → persisted artifacts → interactive inference**

Key engineering principles include `random_state=42` throughout, explicit leakage prevention, reproducible exports, standardized metrics, and automated tests for the deployment layer.

> **[Try the live Streamlit application →](https://machine-learning-casestudy-c2hrudjn2vlgwjzeqbs4b5.streamlit.app/)**

---

## Headline results

### Fraud detection

The tuned **Gradient Boosting** classifier is the classification champion:

| Metric | Score |
|---|---:|
| Accuracy | **0.9350** |
| Precision | **0.8896** |
| Recall | **0.8615** |
| Weighted F1 | **0.9346** |
| ROC-AUC | **0.9779** |

The dataset is imbalanced, with fraud representing approximately 26.5% of observations. Evaluation therefore includes class-aware metrics in addition to accuracy.

### Superconductor regression

The **Random Forest** regressor provides the strongest reported performance:

| Metric | Score |
|---|---:|
| Test R² | **0.9196** |
| RMSE | **9.730 K** |
| MAE | **5.655 K** |
| 5-fold CV R² | **0.9163** |

Tree ensembles outperform the linear baselines, while feature engineering and scaling support the comparison across model families.

### Customer segmentation

The deployed K-Means solution uses **k = 3**. The three segments are interpretable and meaningfully separated by their post-hoc satisfaction rates, even though satisfaction is excluded from model fitting:

| Segment | Size | Profile | Satisfied |
|---|---:|---|---:|
| **Cluster 1** | 43,653 · 42.0% | Satisfied advocates; strong service ratings, business-oriented travellers | **74.4%** |
| **Cluster 0** | 31,345 · 30.2% | Inconsistent service experience; service and comfort gaps | **25.4%** |
| **Cluster 2** | 28,906 · 27.8% | Dissatisfied, at-risk customers; weaker ratings across most dimensions | **15.9%** |

The clustering analysis deliberately reports weak absolute separation rather than overstating it: the deployed model's silhouette score is **0.1108** on the fixed 10,000-customer evaluation sample.

---

## Repository structure

```text
Machine-Learning-CaseStudy/
├── app/                         # Streamlit deployment and inference UI
│   ├── main.py                  # Application entry point
│   ├── ui.py                    # Shared UI components
│   ├── core/                    # Paths, schemas, loaders, inference, figures
│   └── pages/                   # Track pages, comparison, and visualization gallery
├── data/
│   ├── raw/                     # Source datasets
│   └── processed/               # Reusable, processed inputs and assignments
├── models/                      # Persisted fitted pipelines and metadata
│   ├── classification/
│   └── regression/
├── results/                     # Reports, benchmark tables, and visualizations
│   ├── classification/
│   ├── regression/
│   └── clustering/
├── scripts/                     # Artifact and figure export utilities
├── src/
│   ├── classification/          # classification.ipynb
│   ├── regression/              # regression1.ipynb and earlier iteration
│   └── clustering/              # clustering.ipynb
├── tests/                       # Artifact, schema, inference, and safety tests
├── requirements.txt
└── README.md
```

### Main notebooks

- [`src/classification/classification.ipynb`](src/classification/classification.ipynb) — fraud detection benchmark and champion tuning.
- [`src/regression/regression1.ipynb`](src/regression/regression1.ipynb) — final superconductivity regression workflow.
- [`src/clustering/clustering.ipynb`](src/clustering/clustering.ipynb) — customer segmentation, K-Means selection, and agglomerative extension.

---

## Datasets

| Dataset | Task | Shape | Target / usage |
|---|---|---:|---|
| Fraud Detection Bank Dataset | Classification | 20,468 × 114 | `targets`, a binary fraud indicator |
| UCI Superconductivity Dataset | Regression | 21,263 × 82 | `critical_temp`, measured in Kelvin |
| Airplane Customer Satisfaction | Clustering | 103,904 × 25 | `satisfaction` held out for post-hoc validation |

The clustering workflow excludes the satisfaction label from feature construction, model selection, and fitting. This preserves the unsupervised nature of the segmentation while allowing an interpretable validation analysis afterward.

---

## Methodology

### Classification

1. Load preprocessed, leak-free splits or preprocess the raw data as a fallback.
2. Explore target balance, correlations, and feature behaviour.
3. Benchmark five classical models and five ensemble/neural models.
4. Compare accuracy, precision, recall, weighted F1, and ROC-AUC.
5. Tune the strongest candidates with stratified cross-validation.
6. Persist complete inference pipelines and metadata.

### Regression

1. Audit distributions, outliers, duplicates, and multicollinearity.
2. Remove highly correlated features using a `|r| > 0.95` threshold.
3. Add domain-inspired feature engineering, including `thermal_to_mass_ratio`.
4. Fit transformations on training data only.
5. Compare ten models using test-set R²/RMSE/MAE and 5-fold CV R².
6. Persist pipelines that clip physically impossible predictions below `0 K`.

### Clustering

1. Audit 103,904 records, including missing values, duplicates, cardinality, and constants.
2. Impute missing arrival-delay values and transform strongly skewed delay/distance features.
3. One-hot encode nominal variables and standardize the resulting feature matrix.
4. Evaluate K-Means for `k = 2…10` using silhouette, Davies–Bouldin, Calinski–Harabasz, and inertia.
5. Profile the selected clusters and validate them against the held-out satisfaction label.
6. Extend the analysis with PCA, t-SNE, and agglomerative clustering.

### Reproducibility and safety

- `random_state=42` is used for splitting, sampling, initialization, cross-validation, and tuning.
- Supervised preprocessing is fitted on training data only.
- Persisted `.joblib` files contain complete fitted pipelines, preventing training/inference preprocessing drift.
- `metadata.json` records feature order, preprocessing details, split information, and hold-out metrics.
- The Streamlit app performs inference only; it does not train, tune, or cross-validate models.

---

## Interactive application

The Streamlit app is a deployment and exploration layer over the persisted project artifacts.

### Pages

| Page | Functionality |
|---|---|
| **Overview** | Datasets, tracks, headline results, and methodology |
| **Classification** | Enter one sample or upload a CSV and compare all classifiers |
| **Regression** | Predict `critical_temp` with all persisted regressors |
| **Clustering** | Explore the saved segmentation analysis without fake prediction forms |
| **Model Comparison** | Compare persisted test-set metrics with ranking charts |
| **Visualizations** | Browse figures and reports generated by the notebooks |

The classification page distinguishes genuine probabilities from raw decision margins: the SVM is trained with `probability=False`, so its decision score is not labelled as a probability.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Phyboc/Machine-Learning-CaseStudy.git
cd Machine-Learning-CaseStudy

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch the application

```bash
streamlit run app/main.py
```

Run the command from the repository root. Application paths are resolved from the project files, so the UI does not depend on the working directory.

### 3. Re-run the notebooks

```bash
# Classification — run from the repository root
python -m jupyter nbconvert --to notebook --execute --inplace \\
  src/classification/classification.ipynb

# Regression — run from its notebook directory
cd src/regression
python -m jupyter nbconvert --to notebook --execute --inplace regression1.ipynb
cd ../..

# Clustering — run from its notebook directory
cd src/clustering
python -m jupyter nbconvert --to notebook --execute --inplace clustering.ipynb
cd ../..
```

You can also open the notebooks in Jupyter Notebook or VS Code and run all cells interactively.

### 4. Regenerate persisted artifacts

```bash
python scripts/export_classification_models.py
python scripts/export_regression_models.py
python scripts/export_regression_figures.py
```

### 5. Run the test suite

```bash
python -m pytest tests/ -q
```

The tests cover artifact loading, metadata validity, schema and feature-order checks, target-column rejection, CSV validation, inference behaviour, metric reproduction, and the no-training guarantee for `app/`.

---

## Results and artifacts

Generated outputs are organized by track under [`results/`](results/):

- **Classification:** benchmark tables, ROC curves, confusion matrices, target distribution, correlation heatmap, odds ratios, tree visualization, and feature importance.
- **Regression:** leaderboard tables, predicted-vs-actual plots, residual analysis, feature importance, and model comparisons.
- **Clustering:** model-selection curves, cluster profiles, PCA/t-SNE projections, data-quality audits, agglomerative comparisons, and written evaluation reports.

Persisted models live under [`models/`](models/). Each model file is a complete fitted pipeline suitable for inference:

```python
import joblib

pipeline = joblib.load("models/regression/random_forest.joblib")
predictions = pipeline.predict(raw_feature_frame)
```

---

## Technology

- Python 3.10+
- pandas, NumPy, SciPy
- scikit-learn and joblib
- Matplotlib and Seaborn
- Jupyter / nbconvert
- Streamlit
- pytest

---

## Interpretation notes

- Reported metrics are hold-out or cross-validation results from the persisted experiments; they are not guarantees for new data.
- Classification inputs contain 96 features, so arbitrary manual edits may create combinations not represented in the training distribution.
- The SVM exposes a decision score rather than a calibrated probability.
- Clustering separation is weak in absolute terms, although the resulting segments are useful for exploratory profiling and post-hoc comparison.

---

<div align="center">

**Built as an end-to-end machine learning case study — from dataset audit to deployed inference.**

[Open the live demo](https://machine-learning-casestudy-c2hrudjn2vlgwjzeqbs4b5.streamlit.app/) · [Explore the notebooks](src/) · [View the results](results/)

</div>
