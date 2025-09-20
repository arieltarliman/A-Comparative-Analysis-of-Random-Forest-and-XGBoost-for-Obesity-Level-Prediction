# A Comparative Analysis of Random Forest and XGBoost for Obesity Level Prediction

## Overview
This project experiments with **machine learning models** for early health screening.  
Using dataset **1B**, the goal is to predict the **obesity level of screened individuals**.  

Two models were explored: **Random Forest** and **XGBoost**. Both were fine-tuned and evaluated to understand their predictive performance and feature importance.

---

## Project Tasks

### A. Exploratory Data Analysis (EDA)
- Checked variable distributions, correlations, and missing values.
- Identified anomalies such as outliers and inconsistent values, then handled them through filtering and preprocessing.
- Key findings:
  - Some features were highly correlated with obesity levels (e.g., caloric intake, activity levels).
  - Outliers in lifestyle-related variables required treatment for robust modeling.

### B. Model Training
- Trained **Random Forest** and **XGBoost** classifiers.
- Fine-tuned **at least 3 hyperparameters** for each model with **≥3 search spaces**:
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`.
  - XGBoost: `n_estimators`, `max_depth`, `learning_rate`.
- Models were trained and validated, then evaluated on an **independent test set**.

### C. Evaluation & Results
Models were compared using at least **three metrics**:
- **Accuracy**  
- **Precision / Recall**  
- **F1-Score**

**Findings:**
- Both models performed well, but XGBoost showed slightly higher performance across most metrics.
- Random Forest had more stability but underperformed on minority classes compared to XGBoost.

### D. Feature Importance
From the best model (**XGBoost**):
- Top predictive features included dietary habits, physical activity, and demographic variables.
- Feature importance analysis highlights which lifestyle factors are most associated with obesity levels.

---

## Requirements
- **Python 3.10**
- **Jupyter Notebook**
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

---
