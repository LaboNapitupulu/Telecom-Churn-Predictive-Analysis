# Telco Customer Churn Prediction: Multi-Model Analysis & Interpretability

**Domain:** Business Intelligence & Telecommunication | **Tech Stack:** Python, Scikit-Learn, Machine Learning

## Project Overview

This project aims to build a predictive model capable of identifying customers who are at risk of terminating their subscription (churn) at a telecommunications company. Utilizing a dataset encompassing 7,043 customers, this project simulates an end-to-end Data Science project lifecycle, ranging from raw data cleaning to the formulation of customer retention strategies based on the model's findings.

## Tech Stack & Libraries

*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Tree, KNN)
*   **Interpretability:** SHAP & Decision Rules Visualization

## Project Workflow

### 1. Data Understanding & Cleaning
*   **Anomaly Handling:** Identified and corrected the `TotalCharges` column, which was read as a string object due to empty space characters for new customers (tenure 0).
*   **Imputation:** Converted data types to numeric and filled missing values with 0 based on business logic.

### 2. Exploratory Data Analysis (EDA)
*   **Class Imbalance:** Discovered that the target variable is imbalanced, with a ratio of 73.5% (Retained) to 26.5% (Churn).
*   **Key Insights:** Customers with month-to-month contracts and Fiber Optic service users exhibit a significantly higher churn probability compared to other segments.

### 3. Data Preprocessing
*   **Encoding:** Applied One-Hot Encoding with `drop_first=True` to transform 20 categorical features into 31 numerical features while avoiding the Dummy Variable Trap.
*   **Scaling:** Implemented `StandardScaler` (Z-Score Normalization) to standardize the scale of numerical features, thereby optimizing distance-based (KNN) and gradient-based (Logistic Regression) algorithms.
*   **Stratified Splitting:** Partitioned the data (80:20) while maintaining the original target class proportions.

### 4. Modeling & Optimization
Evaluated three different algorithms with hyperparameter optimization:
*   **K-Nearest Neighbors (KNN):** Utilized `GridSearchCV` to determine the optimal $K$ value ($K=21$).
*   **Decision Tree:** Restricted `max_depth` to mitigate overfitting risks.
*   **Logistic Regression:** Served as a robust linear baseline model.

## Model Evaluation Results

| Model | Test Accuracy | CV Mean Accuracy | AUC-ROC Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **80.70%** | **80.25%** | **0.8418** |
| Decision Tree | 79.42% | 78.97% | 0.8267 |
| KNN (Tuned) | 77.08% | 78.83% | 0.8105 |

**Conclusion:** Logistic Regression yields the best performance in distinguishing potential churn customers, achieving the highest AUC score (0.8418).

## Interpretability & Business Recommendations

Based on Feature Importance and Decision Rules analysis:
1.  **Risk Factors:** High monthly charges and the use of Fiber Optic services are the primary drivers for customer churn.
2.  **Retention Factors:** Extended subscription duration (tenure) and long-term contracts (1-2 years) are highly effective in maintaining customer loyalty.
3.  **Strategy:** It is recommended that the company provides loyalty promotions during the initial months of the contract to increase the tenure of new customers.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/LaboNapitupulu/Telecom-Churn-Predictive-Analysis.git
   cd Telecom-Churn-Predictive-Analysis
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Telecom_Churn_Prediction.ipynb
   ```
4. Run all cells to view the output, exploratory analysis, and model evaluations.
