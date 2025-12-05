# Diabetes Hospital Readmission Prediction with Patient Risk Segmentation

## Project Overview
This is a Advanced Data Science project that demonstrates advanced machine learning techniques on the UCI Diabetes 130-US hospitals dataset. The goal is to predict whether a diabetic patient will be readmitted within 30 days and segment patients into meaningful risk groups.

Dataset: diabetic_data.csv (101,766 rows × 55 columns)  
Target: Binary classification → 1 if readmitted within 30 days, else 0 (highly imbalanced ~11% positive class)

Project Structure
```
├── diabetic_data.csv          # Main dataset
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── main.py                    # Main execution script
├── utils.py                   # Utility functions for visualization and reporting
├── data_preprocess.py         # EDA, preprocessing, and feature engineering
├── models.py                  # Supervised learning models training and evaluation
├── clustering.py              # Unsupervised learning and patient segmentation
└── outputs/                   # Generated plots and results (auto-created)
```

 1. Exploratory Data Analysis & Statistics
- Missing value analysis and intelligent imputation
- Correlation + covariance matrix with heatmaps
- Chi-square tests for categorical features
- ANOVA for numerical features vs target

 2. Advanced Preprocessing & Feature Engineering
- Proper encoding of categorical variables (One-Hot + Label Encoding)
- New features: medication_changes, total_procedures, medication_complexity
- Outlier detection and treatment using IQR and Z-score methods

 3. Handling Severe Class Imbalance
- Class distribution visualization
- SMOTE (Synthetic Minority Over-sampling Technique)
- Comparison with class_weight='balanced' approach

 4. Feature Selection + Dimensionality Reduction
- Lasso regression with alpha tuning for feature selection
- PCA retaining ≥90% variance
- Scree plot visualization

 5. Advanced Supervised Learning
Six models trained and compared:
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Models evaluated using StratifiedKFold with metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

 6. Hyperparameter Tuning
- RandomizedSearchCV on best performing model
- Before/after comparison tables
- Detailed improvement analysis

 7. Advanced Unsupervised Learning
- Gaussian Mixture Models (GMM)
- DBSCAN/HDBSCAN clustering
- 2D visualization using t-SNE
- Cluster interpretation and business insights

 8. Results & Business Insights
- Comprehensive model comparison
- Feature importance analysis
- Top factors causing readmission
- Actionable hospital recommendations






