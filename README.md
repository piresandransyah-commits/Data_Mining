# Personal Loan Acceptance Prediction using KNN and Support Vector Machine (SVM)

## Project Overview
This project focuses on predicting customer acceptance of personal loan offers using historical banking data (`Loan.csv`). By analyzing demographic and financial attributes, the project implements classification models—specifically **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**—combined with feature scaling and hyperparameter tuning to optimize loan conversion prediction accuracy.

---

## Key Execution Steps

1. **Exploratory Data Analysis (EDA) & Data Cleaning**
   * **Dataset Verification:** Checked dataset completeness (5,000 entries, 14 features) and confirmed no missing or duplicate values.
   * **Outlier Handling:** Identified non-standard distributions in financial features (`Income`, `CCAvg`, `Mortgage`) using Interquartile Range (IQR) boxplots and replaced extreme outliers with column mean values.
   * **Correlation Analysis:** Generated a correlation heatmap to analyze feature interactions and target dependencies.

2. **Data Preprocessing & Splitting**
   * **Feature Scaling:** Applied `StandardScaler` across numeric predictors to normalize feature magnitude for distance-based models (KNN) and margin-based models (SVM).
   * **Dataset Partitioning:** Split data into an **80% training set** and **20% testing set** (`random_state=1`).

3. **Classification Modeling & Hyperparameter Tuning**
   * **Baseline Models:** Trained initial KNN and SVM models on both raw and preprocessed datasets.
   * **Hyperparameter Optimization:** Conducted `GridSearchCV` cross-validation (5-fold) to identify optimal parameter configurations:
     * **KNN Tuning:** Explored neighbor counts ($K$) ranging from 3 to 101.
     * **SVM Tuning:** Evaluated kernel variations (`rbf`, `linear`) alongside $C$ and $\gamma$ parameters.

4. **Model Evaluation**
   * Assessed performance using **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, and **F1-Score** metrics to ensure model reliability across imbalanced loan acceptance classes.
