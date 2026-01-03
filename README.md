# Supply Chain Management – Delivery & Quantity Risk Modeling

This project analyzes procurement and supplier performance using real-world
purchase order data from two anonymized companies (Company A and Company B).

## Key Objectives
- Identify abnormal orders (invalid, undelivered, extreme dates)
- Analyze delivery delay and quantity deviation patterns
- Build predictive models for:
  - Delivery status (Early / On-time / Late)
  - Quantity accuracy (Less / Correct / More)
- Evaluate supplier risk profiles

## Methodology Overview
1. Data Cleaning & Anomaly Labeling
2. Feature Engineering (Supplier, Product, Time)
3. Visualization & Exploratory Analysis
4. Machine Learning Models:
   - Logistic Regression (Baseline)
   - Random Forest
   - LightGBM
   - XGBoost
   - CatBoost
5. Model Evaluation (Macro F1-score)
6. Model Interpretability (SHAP)

## Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn
- LightGBM / XGBoost / CatBoost
- SHAP
- Google Colab

## Data
The original datasets are anonymized and not included in this repository.

## Author
Created by Ersin77777

