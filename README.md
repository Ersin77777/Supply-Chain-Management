# Supply Chain Management – Delivery & Quantity Risk Modeling

This project analyzes procurement and supplier performance using real-world
purchase order data from two anonymized companies (Company A and Company B).

The goal is to identify operational risks in supply chains and to build
data-driven models that support supplier evaluation, delivery reliability
assessment, and inventory-related decision-making.

---

## 🎯 Key Objectives

- Identify abnormal orders:
  - Invalid orders (e.g. ordered quantity = 0)
  - Undelivered/open orders
  - Extreme or erroneous planned dates
- Analyze delivery delays and quantity deviations
- Build predictive models for:
  - **Delivery status**: Early / On-time / Late
  - **Quantity accuracy**: Less / Correct / More
- Evaluate supplier risk profiles based on historical performance

---

## 🧭 Methodology Overview

1. **Data Cleaning & Anomaly Labeling**
   - Missing arrival dates
   - Invalid quantities
   - Weekend and public holiday effects
   - Extreme date anomalies

2. **Feature Engineering**
   - Supplier-level historical performance
   - Product / material identifiers
   - Time-based features (month, weekday, quarter)
   - Aggregated delay and deviation statistics

3. **Exploratory Analysis & Visualization**
   - Missing and zero-value analysis
   - Delivery delay distributions
   - Special order distributions
   - Supplier performance comparison

4. **Machine Learning Models**
   - Logistic Regression (baseline)
   - Random Forest
   - LightGBM
   - XGBoost
   - CatBoost

5. **Model Evaluation**
   - Macro F1-score (classification)
   - Robustness checks via cross-validation

6. **Model Interpretability**
   - SHAP-based feature importance analysis
   - Supplier, product, and time effect interpretation

---

## 🧱 Project Structure

```text
supply-chain-management/
├─ notebooks/
│  └─ Supply_Chain_Management.ipynb   # End-to-end analysis & experiments
├─ src/
│  ├─ config.py        # Path and project configuration
│  ├─ io.py            # Data loading and saving
│  ├─ cleaning.py      # Data cleaning & anomaly flags
│  ├─ labeling.py      # Delivery & quantity labels
│  ├─ features.py      # Feature engineering
│  ├─ visualization.py# Plots & exploratory analysis
│  ├─ models.py        # Training, evaluation, persistence
│  ├─ explain.py       # SHAP-based interpretability
│  └─ predict.py       # Single-order prediction utilities
├─ data/
│  └─ README.md        # Data schema & assumptions (no data included)
├─ outputs/
│  ├─ preprocessed/    # Generated intermediate data
│  ├─ models/          # Trained models (ignored by git)
│  └─ figures/         # Generated plots
├─ requirements.txt
└─ .gitignore
```


## Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn
- LightGBM / XGBoost / CatBoost
- SHAP
- Matplotlib / Seaborn
- Google Colab

## Data
The original datasets are confidential and not included in this repository.

## Author
Created by Ersin77777