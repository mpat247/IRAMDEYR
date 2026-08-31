# Predicting Past-Year Major Depressive Episode (IRAMDEYR)

Screening-grade prediction of past-year Major Depressive Episode (MDE) in adults using the **NSDUH 2023** survey, with a focus on:

- **Compact, interpretable feature sets**
- **High recall** and **high NPV** for triage
- Work that can be understood by both technical and clinical stakeholders

---

## 1. Research Question

> **Can we build a compact, interpretable, survey-weighted model that predicts past-year MDE (IRAMDEYR) in adults, using NSDUH variables, while maximizing recall and negative predictive value (NPV) for triage?**

The long-term goal is a brief feature set that could be embedded into:
- Population surveys
- Short screening questionnaires
- Triage workflows where clinician time and data availability are limited

---

## 2. Why This Matters

- NSDUH is widely used for **prevalence** and descriptive statistics, but not as often for **individual-level prediction**.
- Most ML work in mental health uses large black-box models that are hard to deploy and explain.
- Here, we explicitly aim for:
  - **High recall / NPV** (few missed MDE cases)
  - **Small, interpretable feature sets**
  - A design that could plausibly be used in **triage** and **EHR**-like contexts.

---

## 3. Dataset

We use the **2023 National Survey on Drug Use and Health (NSDUH)**:

- ~45,000 adult respondents (after filtering to AGE3 ≥ 4 / adults)
- ~2,300+ variables initially
- Target variable: **IRAMDEYR**  
  - Indicator for past-year Major Depressive Episode (MDE)
- Survey weight: **ANALWT2_C** (used in later modeling phase, not in all feature-selection notebooks)

> **Note:** NSDUH data cannot be redistributed in this repo.  
> You must obtain it directly from the official source and place it in the `data/` directory.

---

## 4. High-Level Methodology

The project is structured into three main phases:

### Phase 1 – Data Cleaning & Leakage Control

- Filter to **adult** respondents.
- Convert NSDUH special codes (e.g., 94, 97, 98, 99, 9999, etc.) to `NaN`.
- Drop:
  - Obvious identifiers (e.g., `QUESTID2`, CASEID-like variables)
  - Sampling design variables (e.g., `VESTR_C`, `VEREP`)
  - File metadata (e.g., `FILEDATE`, `YEAR`, `QUARTER`)
  - Variables with >99% missingness
  - Variables used to impute/recode the target to prevent target leakage (About 50 relevant variables were removed)
- Keep survey weight **ANALWT2_C** in the cleaned dataset for later phases.
- Save a single cleaned dataset as:  
  `../data/cleaned/NSDUH_2023_clean.pkl` , This cleaned file serves as the input to a subsequent **feature preparation step**, which performs final column alignment and prepares model-ready inputs.
- **Feature Preparation**: A subsequent feature preparation step performs class abstraction and replaces raw NSDUH variables with recoded features, saving the updated dataset back to the same pickle file for downstream use.
- Updated cleaned dataset saved as:
  `../data/cleaned/NSDUH_2023_clean.pkl`

A single **train/test split** (with `random_state=42`) is used consistently in the feature-selection notebooks to avoid leaking information from the test set into the feature selection stage.

---

### Phase 2 – Data-Driven Feature Selection (4 Methods × AUC/Recall)

We apply four feature-selection approaches, each optimized twice:

1. **LASSO** (L1-penalized logistic regression)
2. **Elastic Net** (combined L1/L2-penalized logistic regression)
3. **RFE** (Recursive Feature Elimination with logistic regression)
4. **RFECV** (RFE with cross-validation, logistic regression base estimator)

Each method is run in two modes:

- **AUC-oriented version**  
- **Recall-oriented version**

Total: **8 ranked feature lists** (each typically exporting the **top 100** features to CSV).

These notebooks:

- Read the **same cleaned pickle** (`NSDUH_2023_clean.pkl`).
- Drop `IRAMDEYR` from `X` and keep it as `y`.
- Drop `ANALWT2_C` from `X` for feature selection (weights are not used to rank features).
- Use:
  - Train/test split (`train_test_split`, `random_state=42`, `stratify=y`)
  - Imputation and scaling (usually via `SimpleImputer` + `StandardScaler` or pre-processing steps)
  - Logistic regression for model-based feature importance.

Outputs:

- `../results/lasso_auc_top100.csv`  
- `../results/lasso_recall_top100.csv`  
- `../results/elasticnet_auc_top100.csv`  
- `../results/elasticnet_recall_top100.csv`  
- `../results/rfe_auc_top100.csv`  
- `../results/rfe_recall_top100.csv`  
- `../results/rfecv_auc_top100.csv`  
- `../results/rfecv_recall_top100.csv`

These files are used to build the agreement table.

---

### Phase 3 – Agreement Table & Intuition-Based Tiers

1. **Agreement Table (Data-Driven)**  
   A central notebook loads all 8 top-100 CSVs and builds a unified **Feature Agreement Table**:

   - Each row = one feature (NSDUH variable name)
   - Columns = 8 methods (LASSO_AUC, LASSO_REC, EN_AUC, EN_REC, RFE_AUC, RFE_REC, RFECV_AUC, RFECV_REC)
   - Cells = 1 if the feature is in that method’s top-100, else 0
   - `SUM` column = how many methods selected this feature

   Export:  
   `../results/feature_agreement_table.csv`

2. **Intuition-Based Tiering (Clinical / Triage Perspective)**  
   Using:
   - The **agreement table**,  
   - The NSDUH **codebook**, and  
   - A reference **tiering paper** (basic/clinical/personalized approach),  

   we assign features into **three practical tiers**:

   - **Tier 1 – Basic/Demographic**  
     - Low-burden information typically available in triage/intake (age, sex, basic sociodemographics, very simple self-report items).
   - **Tier 2 – Clinical/EHR-Accessible**  
     - Includes Tier 1 + variables plausibly available in an EHR or rapid intake (comorbid conditions, some medication/treatment indicators, simple functioning items).
   - **Tier 3 – Personalized/Extended**  
     - Includes Tier 2 + more detailed symptom scales, functioning metrics, and additional nuance that may not always be available at first contact but improve performance.
  
The finalized tier definitions are formalized using a dedicated preparation script (`tier_preparation.py`), which exports each tier as a reusable feature list. These tier-specific feature sets are then passed directly to the predictive modeling notebooks for evaluation and comparison.

The tier assignment is stored in a combined CSV (e.g., `combined_feature_translation.csv`), which includes:

- Feature_Code  
- Description  
- Category (e.g., demographics, functioning, substance use, treatment, etc.)  
- Tier (1, 2, or 3)  
- Agreement_Score (how many methods selected it)

This CSV is used later when we build and evaluate final predictive models.

---

## 5. Repository Structure:

```text
project-root/
├─ data/
│  ├─ raw/
│  │  └─ NSDUH_2023_Tab.txt (or similar NSDUH raw file)  [not in repo]
│  ├─ cleaned/
│  │  └─ NSDUH_2023_clean.pkl
|
├── models
│   ├── lightgbm_tier_2_clinical_recall.joblib
│   └── lightgbm_tier_3_personalized_recall.joblib
|
├── notebooks
│   ├── Baseline_Characteristics_General_EDA.ipynb
│   ├── create_metric_tables.ipynb
│   ├── create_model_comparison_table.ipynb
│   ├── data_cleaning.ipynb
│   ├── feature_preparation.ipynb
│   ├── feature_selection_agreement.ipynb
│   ├── feature_selection_EDA.ipynb
│   ├── feature_selection_elasticnet.ipynb
│   ├── feature_selection_lasso.ipynb
│   ├── feature_selection_RFECV.ipynb
│   ├── feature_selection_RFE.ipynb
│   ├── predictive_model_lightGBM.ipynb
│   ├── predictive_model_logistics_regression.ipynb
│   ├── predictive_model_random_forest.ipynb
│   ├── predictive_model_support_vector_machine.ipynb
│   ├── predictive_model_xgboost.ipynb
│   ├── shap_lightgbm_interpretability.ipynb
│   └── tiering_preparation.py
|
├── README.md
├── requirements.txt
|
├── results
│   ├── all_models_by_fold.csv
│   ├── all_models_comparison.csv
│   ├── elasticnet_auc_top100.csv
│   ├── elasticnet_recall_top100.csv
│   ├── feature_agreement_table.csv
│   ├── figures
│   │   ├── rf_grid_pr_curve.png
│   │   ├── rf_grid_roc_curve.png
│   │   ├── rf_random_pr_curve.png
│   │   ├── rf_random_roc_curve.png
│   │   ├── xgb_grid_pr_curve.png
│   │   ├── xgb_grid_roc_curve.png
│   │   ├── xgb_random_pr_curve.png
│   │   └── xgb_random_roc_curve.png
│   ├── lasso_auc_top100.csv
│   ├── lasso_recall_top100.csv
│   ├── lightgbm_5fold_threshold0.50_by_tier.csv
│   ├── lightgbm_all_tiers_results.csv
│   ├── lightgbm_testset_metrics_by_tier.csv
│   ├── lightgbm_two_rows_per_tier.csv
│   ├── logreg_5fold_threshold0.50_by_tier.csv
│   ├── model_comparison_table_means.csv
│   ├── rf_5fold_threshold0.50_by_tier.csv
│   ├── rfe_auc_top100.csv
│   ├── rfecv_auc_top100.csv
│   ├── rfecv_recall_top100.csv
│   ├── rfe_recall_top100.csv
│   ├── rf_grid_tier_performance.csv
│   ├── rf_random_tier_performance.csv
│   ├── rf_two_rows_per_tier.csv
│   ├── table1_Baseline_Characteristics.csv
│   ├── xgb_5fold_threshold0.50_by_tier.csv
│   ├── xgb_comprehensive_results.csv
│   ├── xgb_grid_tier_performance.csv
│   ├── xgb_random_tier_performance.csv
│   ├── xgb_shap_importance_by_tier.csv
│   ├── xgb_shap_values_Tier_1_Basic.csv
│   ├── xgb_shap_values_Tier_2_Clinical.csv
│   ├── xgb_shap_values_Tier_3_Personalized.csv
│   └── xgb_two_rows_per_tier.csv
|
└── tiers
    ├── tier_1_basic_features.csv
    ├── tier_2_clinical_features.csv
    └── tier_3_personalized_features.csv

```
**pipeline_publication5.py** — publication-ready end-to-end pipeline for final modeling, threshold selection, weighted evaluation, and SHAP analysis used in the manuscript.

## Instructions & Reproducibility Guide

This repository is organized as a **research-oriented data science project**, with a strong emphasis on **reproducibility, transparency, and auditability**.  
Please follow the steps below **in order**.

---

## 1. Obtain the NSDUH Dataset

This project uses the **2023 National Survey on Drug Use and Health (NSDUH)** public-use data.

### Official Source
You must download the data directly from the official SAMHSA site:

👉 https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health

Use the **public-use data file** (tab-delimited format).

> **Note:**  
> NSDUH data cannot be redistributed. The raw dataset is **not** included in this repository.

---

## 2. Place the Raw Dataset in the Correct Folder

After downloading the NSDUH file, place it here:
```text
data/raw/
└── NSDUH_2023_Tab.txt
```

> ⚠️ Do not rename the file unless you also update the corresponding path in the data cleaning notebook.

---

## 3. Set Up the Python Environment (Required)

To ensure identical results across machines, **all users must install the exact dependencies listed in `requirements.txt`**.

### Recommended Steps

Create and activate a virtual environment:

```bash
python -m venv venv
```
Activate it:

- **Windows (PowerShell)**

  venv\Scripts\activate

- **macOS / Linux**

  source venv/bin/activate


Install dependencies:
```
pip install -r requirements.txt 
```

## 4. Run the Data Cleaning & Feature Preparation Notebooks (Required, Order Matters)

This step is mandatory and must be completed before running any feature selection or modeling notebooks.
The notebooks must be executed in the order shown below.

### 4.1. Step 1: Data Cleaning
```
notebooks/data_cleaning.ipynb
```
This notebook performs the core data cleaning and leakage control steps:
- Loads the raw NSDUH 2023 public-use dataset
- Filters to adult respondents
- Converts NSDUH special missing-value codes (e.g., 94, 97, 98, 99) to NaN
- Removes identifiers, survey design variables, metadata, and leakage-prone variables
- Retains the target variable (IRAMDEYR) and survey weight (ANALWT2_C)

### 4.2. Step 2: Feature Preparation (Must Run After Cleaning)
```
notebooks/feature_preparation.ipynb
```
This notebook performs additional feature preparation, including:
- Class abstraction and recoding of raw NSDUH variables
- Transformation of selected raw fields into analysis-ready representations
- Alignment of feature formats used consistently across downstream notebooks

The prepared dataset is saved to the pickle file:
```
data/cleaned/NSDUH_2023_clean.pkl
```

🔑 This pickle file is the single source of truth
All feature selection and modeling notebooks depend on it.


## 5. Feature Selection, Agreement Table, and Tier Construction

After data cleaning and feature preparation, the cleaned pickle file is used for **data-driven feature selection** and tier construction.
```
feature_selection_lasso.ipynb
feature_selection_elasticnet.ipynb
feature_selection_RFE.ipynb
feature_selection_RFECV.ipynb
```
All feature selection notebooks follow the same high-level pattern:

- Load the prepared dataset:
  ```
  df = pd.read_pickle("../data/cleaned/NSDUH_2023_clean.pkl")
  ```
- Perform a **single train/test split**:
  - test_size = 0.2
  - stratify = y
  - random_state = 42
- Apply preprocessing **only after the split**:
  - Imputation
  - Scaling / standardization (where required)
- Ensure that feature selection and model fitting never see the test set

### 5.1. Feature Selection Methods

Four data-driven feature selection methods are applied, each in **two optimization modes**:

- **LASSO** (AUC-optimized, Recall-optimized)
- **Elastic Net** (AUC-optimized, Recall-optimized)
- **RFE** (AUC-optimized, Recall-optimized)
- **RFECV** (AUC-optimized, Recall-optimized)

For each method and objective:
- Features are ranked based on model-derived importance
- The **top 100 features** are retained
- Results are exported as CSV files to the ```results/``` directory

### 5.2. Agreement Table Construction

The ranked feature lists from all methods are combined to create a unified Feature Agreement Table:

- Unified feature set created as the union of all top-100 lists
- Binary indicator for each (feature × method) pair:
  - ```1``` if selected in that method’s top-100
  - ```0``` otherwise
- SUM column representing how many methods selected each feature (range: 1-8)
- Features are sorted by SUM in descending order

The resulting agreement table is saved as:
```
results/feature_agreement_table.csv
```

### 5.3. Tier Finalization and Downstream Modeling

The agreement table is then used, together with:
- NSDUH codebook descriptions
- Clinical and triage-oriented intuition

The finalized tier definitions are formalized in ```tiering_preparation.py```, which exports tier-specific feature lists. These tier feature lists are subsequently used by all predictive modeling notebooks to evaluate model performance across tiers.

## 6. Model Performance and Evalution

Using the finalized tier-specific feature sets, multiple predictive models are trained and evaluated to assess screening performance across tiers.

### 6.1. Models Evaluated

The following algorithms are implemented, each serving a distinct purpose:

- **Logistic Regression** - ```predictive_model_logistics_regression.ipynb```
- **Random Forest** - ```predictive_model_random_forest.ipynb```
- **XGBoost (Extreme Gradient Boosting)** - ```predictive_model_xgboost.ipynb```
- **LightGBM (Light Gradient Boosting Machine)** - ```predictive_model_lightGBM.ipynb```

These models provide a balance between **interpretability, non-linearity**, and **predictive performance**.

### 6.2. Training and Validation Strategy

A consistent data-splitting and validation strategy is used across **all models and tiers**:
- **Train/Test Split**
  - 80% training set
  - 20% held-out test set
  - Stratified on the target variable to preserve class imbalance
  - Fixed random seed (random_state = 42) for reproducibility
  - Survey weights (**ANALWT2_C**) are split alongside features and labels

The test set is **completely held out** and is **never used** during feature selection or hyperparameter tuning.

### 6.3. Hyperparameter Tuning and Cross-Validation

Model hyperparameters are optimized using **stratified 5-fold cross-validation** on the training data:
- Stratification preserves the original class distribution in each fold
- Shuffling is enabled to prevent bias due to sample ordering
- For each hyperparameter configuration:
  - The model is trained on 5 folds and validated on the remaining fold
  - Performance is evaluated on the validation fold

Results are aggregated as **mean ± standard deviation** across the 5 folds

Hyperparameter tuning is primarily **optimized for recall**, reflecting the screening objective of minimizing false negatives.

### 6.4. Performance Metrics
Model performance is evaluated using both threshold-independent and threshold-based metrics, with emphasis on screening utility.

- **AUC**
  - Used to assess overall discriminative ability between MDE and non-MDE cases, independent of any classification threshold.

- **Sensitivity (Recall)**
  - Proportion of true MDE cases correctly identified.

- **Specificity**
  - Proportion of non-MDE cases correctly identified.

- **PPV (Precision)**
  - Probability that a predicted MDE case truly has MDE.

- **NPV**
  - Probability that a predicted non-MDE case truly does not have MDE.

- **Accuracy**
  - Overall correctness across both classes.

- **F1 and F2 Scores**
  - Summary metrics balancing precision and recall, with F2 placing greater emphasis on recall to align with screening objectives.

Unless otherwise stated, threshold-based metrics are computed at a **fixed decision threshold of 0.50** on the held-out test set.

### 6.5. Outputs
Model evaluation outputs include:
- Fold-level performance metrics (cross-validation)
- Tier-wise performance summaries
- Final test-set metrics for each model and tier
- ROC and Precision-Recall curves
- SHAP-based interpretability analyses for selected models

All results are written to the ```results/``` directory.

## 7. Explainability and Interpretability (SHAP Analysis)

To ensure clinical transparency and interpretability, **SHAP** was used to explain predictions from the **best-performing LightGBM model**.

```
shap_lightgbm_interpretability.ipynb
```

SHAP analysis was conducted on:
- **Tier 2 (Clinical)** features
- **Tier 3 (Personalized)** features

SHAP values were computed separately for the **training and test sets** to assess stability and generalization of feature contributions.

### 7.1. SHAP Visualizations
For both Tier 2 and Tier 3 models, the following were generated:
- **SHAP Summary (Beeswarm) Plots**
  - Show global feature importance and directionality.
  - Positive SHAP values (right) indicate increased predicted MDE risk.
  - Negative SHAP values (left) indicate decreased predicted MDE risk.

- **SHAP Bar Plots**
  - Display mean absolute SHAP values per feature.
  - Rank features by their average impact on predictions.

- **SHAP Dependence Plots (Top Features)**
  - Illustrate how feature values relate to their SHAP contributions.
  - Reveal non-linear effects and interactions.

### 7.2. Key Findings

**Tier 2 (Clinical Features)**:
- **KSSLR6MAX_ABS (psychological distress)** was the strongest contributor: lower distress reduced risk, while higher distress substantially increased predicted MDE risk.
- **RCVYMHPRB (recovering from a mental health problem)** was associated with higher predicted risk.
- **IRSEX** showed higher predicted risk for females compared to males.
- **IRPINC3_ABS** indicated increased risk at lower income levels.
- **IRSUICTHNK (suicidal ideation)** strongly increased predicted risk.
- **CATG3 (age group)** showed structured age-related risk patterns.

Train and test SHAP patterns were highly consistent for Tier 2, indicating stable model behavior.

**Tier 3 (Personalized Features)**:
- **KSSLR6MAX_ABS** remained the most influential feature.
- Additional high-impact contributors included:
  - **WHODASDASC (disability/functioning score)**
  - **RCVYMHPRB**
  - **IRIMPGOUT_ABS**
  - **IRMARIT_ABS**
  - **COCLALCUSE**

Overall, SHAP analysis confirmed that the model relies on **clinically intuitive, interpretable signals**, with consistent feature effects across training and test sets.






















