# A Comparative Analysis of Random Forest and XGBoost for Obesity Level Prediction

## 🚀 Introduction

This project presents a comparative analysis of two powerful machine learning models—**Random Forest** and **XGBoost**—for the prediction of obesity levels based on a comprehensive set of demographic, lifestyle, and health-related features. The primary objective is to develop a reliable classification model that can be used for early screening and to identify the most significant factors contributing to an individual's weight status.

This repository contains the complete analysis in a Jupyter Notebook, detailing every step from data exploration and preprocessing to model training, hyperparameter tuning, and in-depth evaluation.

***

## 🎯 Problem Statement

The core task of this project is to build and evaluate machine learning models capable of accurately predicting an individual's obesity level, which is categorized into seven distinct classes. The project aims to:

1.  Perform a thorough Exploratory Data Analysis (EDA) to understand the data and handle any anomalies.
2.  Train and fine-tune both a **Random Forest** and an **XGBoost** classifier.
3.  Conduct a detailed comparative analysis of the models' performance using multiple evaluation metrics.
4.  Identify the most influential features for predicting obesity levels from the best-performing model.

***

## 📊 Dataset

The analysis is based on the "Dataset 1B," which contains various attributes of screened individuals. The key features include:

* **Demographics**: Gender and Age (derived from Birth Date).
* **Physical Metrics**: Height, Weight, and the calculated Body Mass Index (BMI).
* **Health & Lifestyle**: Family history with overweight, frequency of high-calorie meals and vegetable consumption, daily meal patterns, snacking habits, smoking and alcohol consumption, daily water intake, and weekly physical activity.
* **Environmental Factors**: Mode of transportation.
* **Target Variable**: Obesity level, categorized into seven classes (Insufficient Weight, Normal Weight, Overweight Level I & II, Obesity Type I, II, & III).

***

## 🛠️ Methodology

A systematic data science workflow was followed to ensure the robustness and reliability of the results.

### 1. Exploratory Data Analysis (EDA) & Preprocessing

The EDA phase focused on data cleaning, feature engineering, and handling anomalies to prepare the dataset for modeling. Key steps included:

* **Data Cleaning**:
    * Missing values in the `Snack Frequency` column were minimal (10 out of 2111 rows) and were handled by dropping the affected rows.
    * No duplicated values were found in the dataset.
    * Anomalous text entries in the `Weight` ('delapan puluh') and `Smoking` ('hehe') columns were identified and corrected. The `Weight` anomaly was converted to its numeric equivalent (80.0), and the few anomalous `Smoking` entries were removed.
* **Feature Engineering**:
    * The `Birth Date` column was converted into a more practical `Age` column.
    * A **Body Mass Index (BMI)** column was engineered from the `Height` and `Weight` features, as it is a critical indicator of obesity.
    * Continuous variables like `Veggies in Meals freq`, `Daily Main Meals`, `Daily Water Consumption`, and `Weekly Physical Activity` were binned into categorical features to simplify the model's learning process.
* **Data Transformation**: All categorical features were label-encoded to be compatible with the machine learning models. The original continuous and high-cardinality columns were subsequently dropped.

### 2. Model Training & Hyperparameter Tuning

Two state-of-the-art ensemble models were trained and fine-tuned:

* **Random Forest**: A robust, tree-based model known for its ability to handle complex interactions and outliers.
* **XGBoost**: A powerful gradient-boosting algorithm recognized for its high performance and speed.

Both models were fine-tuned using `GridSearchCV` with a 5-fold cross-validation strategy. The hyperparameter search space was designed to optimize for the **F1-score**, which is a suitable metric for multi-class classification, especially with imbalanced classes.

### 3. Evaluation

The models were evaluated on an independent test set using a comprehensive set of metrics:

* **ROC AUC Score**: To assess the models' ability to discriminate between different obesity classes.
* **Confusion Matrix**: To visualize the classification accuracy and identify common misclassifications.
* **Classification Report**: To analyze precision, recall, and F1-score for each class, providing a detailed view of the models' strengths and weaknesses.

***

## 📈 Results & Analysis

Both the Random Forest and XGBoost models demonstrated exceptionally high performance, achieving near-perfect classification on the test set.

* **Random Forest**: Achieved a remarkable **ROC AUC score of 0.9994**, indicating almost perfect class separation. The F1-scores for all classes were consistently high, ranging from 96% to 98%.
* **XGBoost**: Also performed excellently, with a **ROC AUC score of 0.9957**. While slightly lower than Random Forest, this score still signifies outstanding predictive power. The F1-scores were also impressive, with a notable 99% for "Obesity Type II."

Overall, while both models are highly effective, **Random Forest showed a slight edge** in terms of overall performance and consistency across all evaluation metrics.

***

## 💡 Feature Importance

Based on the best-performing model, **Random Forest**, an embedded feature importance analysis was conducted. The key findings were:

1.  **BMI**: Unsurprisingly, **Body Mass Index** was the most influential feature by a significant margin, confirming its central role in determining obesity levels.
2.  **Age**: The second most important feature, highlighting the impact of age-related metabolic changes and lifestyle patterns.
3.  **Gender**: Also a significant predictor, suggesting that gender-specific physiological differences play a role in weight status.

Interestingly, lifestyle factors such as diet, exercise, and smoking had a much lower impact on the model's predictions compared to these three core features.

***

## ⚙️ How to Reproduce

To reproduce this analysis, you can follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/arieltarliman/A-Comparative-Analysis-of-Random-Forest-and-XGBoost-for-Obesity-Level-Prediction.git](https://github.com/arieltarliman/A-Comparative-Analysis-of-Random-Forest-and-XGBoost-for-Obesity-Level-Prediction.git)
    ```
2.  **Install dependencies**: Make sure you have Python and the necessary libraries (e.g., pandas, scikit-learn, xgboost, seaborn) installed.
3.  **Run the Jupyter Notebook**: Open and run the `UAS NO 1.ipynb` notebook to see the full analysis.
