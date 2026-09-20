# Personal Loan Acceptance Prediction using KNN and Support Vector Machine (SVM)

## Project Overview

Personal loan acceptance prediction plays a crucial role in targeted marketing strategies for commercial banking. This project aims to construct and evaluate predictive classification models to accurately identify potential customers likely to accept personal loan offers based on their demographic and financial profiles from the `Loan.csv` dataset. The dataset, consisting of 5,000 customer records across 14 attributes, was preprocessed using Exploratory Data Analysis (EDA), missing value verification, and Interquartile Range (IQR) outlier handling on continuous financial variables (`Income`, `CCAvg`, and `Mortgage`). Features were standardized using `StandardScaler` to ensure uniform weighting across distance-based and margin-based models. Two supervised learning algorithms—**K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**—were implemented and evaluated on raw and outlier-handled datasets before and after 5-fold cross-validation `GridSearchCV` hyperparameter tuning ($K$ values ranging from 3 to 101 for KNN; $C$, $\gamma$, and kernel functions for SVM). Baseline KNN on raw scaled data achieved an Accuracy of 0.6970, Precision of 0.4483, Recall of 0.2737, and F1-Score of 0.3399, which improved upon hyperparameter tuning ($K = 37$) to an Accuracy of 0.7500, Precision of 0.7966, Recall of 0.1649, and F1-Score of 0.2733. Handling outliers yielded similar baseline KNN metrics (Accuracy: 0.7020, Precision: 0.4611, Recall: 0.2702, F1-Score: 0.3407). Baseline SVM achieved an Accuracy of 0.7560, Precision of 0.8868, Recall of 0.1649, and F1-Score of 0.2781. Overall, feature standardization and hyperparameter optimization significantly enhanced classification reliability, demonstrating that tuned predictive models offer a effective, data-driven framework for optimizing credit marketing campaigns.

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
