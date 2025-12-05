# Diabetes Hospital Readmission Prediction with Patient Risk Segmentation

## Project Overview
This is a comprehensive end-semester Data Science project that demonstrates advanced machine learning techniques on the UCI Diabetes 130-US hospitals dataset. The goal is to predict whether a diabetic patient will be readmitted within 30 days and segment patients into meaningful risk groups.

**Dataset:** diabetic_data.csv (101,766 rows × 55 columns)  
**Target:** Binary classification → 1 if readmitted within 30 days, else 0 (highly imbalanced ~11% positive class)

## Project Structure
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

## Topics Covered (Syllabus Requirements)

### 1. Exploratory Data Analysis & Statistics
- Missing value analysis and intelligent imputation
- Correlation + covariance matrix with heatmaps
- Chi-square tests for categorical features
- ANOVA for numerical features vs target

### 2. Advanced Preprocessing & Feature Engineering
- Proper encoding of categorical variables (One-Hot + Label Encoding)
- New features: medication_changes, total_procedures, medication_complexity
- Outlier detection and treatment using IQR and Z-score methods

### 3. Handling Severe Class Imbalance
- Class distribution visualization
- SMOTE (Synthetic Minority Over-sampling Technique)
- Comparison with class_weight='balanced' approach

### 4. Feature Selection + Dimensionality Reduction
- Lasso regression with alpha tuning for feature selection
- PCA retaining ≥90% variance
- Scree plot visualization

### 5. Advanced Supervised Learning
Six models trained and compared:
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

All models evaluated using StratifiedKFold with metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 6. Hyperparameter Tuning
- RandomizedSearchCV on best performing model
- Before/after comparison tables
- Detailed improvement analysis

### 7. Advanced Unsupervised Learning
- Gaussian Mixture Models (GMM)
- DBSCAN/HDBSCAN clustering
- 2D visualization using t-SNE
- Cluster interpretation and business insights

### 8. Results & Business Insights
- Comprehensive model comparison
- Feature importance analysis
- Top factors causing readmission
- Actionable hospital recommendations

## Installation & Setup

### Option 1: Local Setup
```bash
# Clone or download the project
cd "vac cp"

# Install dependencies
pip install -r requirements.txt

# Run the complete project
python main.py
```

### Option 2: Google Colab
```python
# Upload all .py files and diabetic_data.csv to Colab
!pip install -r requirements.txt
!python main.py
```

## Execution
Simply run:
```bash
python main.py
```

The script will:
1. Load and preprocess the data
2. Perform comprehensive EDA with statistical tests
3. Engineer features and handle class imbalance
4. Apply feature selection and dimensionality reduction
5. Train and evaluate 6+ machine learning models
6. Perform hyperparameter tuning on the best model
7. Apply unsupervised learning for patient segmentation
8. Generate all visualizations in the `outputs/` folder
9. Display comprehensive results and insights

## Expected Runtime
- Local laptop: 10-20 minutes
- Google Colab: 5-10 minutes

## Outputs
All plots and visualizations are saved in the `outputs/` folder:
- Missing value heatmap
- Correlation matrix
- Class imbalance visualization
- PCA scree plot
- ROC curves for all models
- Confusion matrices
- Feature importance plots
- Cluster visualizations
- Model comparison tables

## Key Results Preview
- **Best Model:** XGBoost/LightGBM (ROC-AUC ~0.65-0.70)
- **Top Readmission Factors:** 
  1. Number of inpatient visits
  2. Number of diagnoses
  3. Time in hospital
  4. Discharge disposition
  5. Age group
- **Patient Segments:** 3-4 risk groups identified with distinct characteristics

## Author
Data Science Course Project - End Semester Submission

## License
Educational Use Only

## References
- UCI Machine Learning Repository: Diabetes 130-US hospitals dataset
- Scikit-learn documentation
- XGBoost, LightGBM, CatBoost documentation

