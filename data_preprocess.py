"""
Data Preprocessing, EDA, and Feature Engineering Module
This module handles all data loading, exploratory data analysis, statistical tests,
preprocessing, and feature engineering tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from utils import (print_section_header, print_subsection_header, plot_missing_values,
                   plot_correlation_matrix, plot_class_distribution, plot_outliers_boxplot)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for diabetes readmission prediction
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None

    def load_data(self):
        """Load the dataset"""
        print_section_header("STEP 1: DATA LOADING")
        print(f"Loading data from: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        print(f"  Rows: {self.df.shape[0]:,}")
        print(f"  Columns: {self.df.shape[1]}")

        print("\nFirst few column names:")
        print(f"  {', '.join(self.df.columns[:10].tolist())}...")

        return self.df

    def exploratory_data_analysis(self, output_dir='outputs'):
        """
        Comprehensive EDA with statistical analysis
        """
        print_section_header("STEP 2: EXPLORATORY DATA ANALYSIS & STATISTICS")

        # Basic info
        print_subsection_header("2.1 Dataset Overview")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nData types distribution:")
        print(self.df.dtypes.value_counts())

        # Missing values analysis
        print_subsection_header("2.2 Missing Value Analysis")
        missing_df = plot_missing_values(self.df, output_dir)

        # Handle missing values - replace '?' with NaN
        print("\n✓ Replacing '?' with NaN for proper handling...")
        self.df = self.df.replace('?', np.nan)

        # Create binary target variable
        print_subsection_header("2.3 Target Variable Creation")
        print("Creating binary target: readmitted_binary")
        print("  1 = readmitted within 30 days (<30)")
        print("  0 = not readmitted or readmitted after 30 days")

        self.df['readmitted_binary'] = (self.df['readmitted'] == '<30').astype(int)

        print(f"\nTarget distribution:")
        print(self.df['readmitted_binary'].value_counts())
        print(f"\nClass imbalance ratio:")
        class_counts = self.df['readmitted_binary'].value_counts()
        print(f"  Positive class: {class_counts[1]} ({100*class_counts[1]/len(self.df):.2f}%)")
        print(f"  Negative class: {class_counts[0]} ({100*class_counts[0]/len(self.df):.2f}%)")
        print(f"  Imbalance ratio: 1:{class_counts[0]/class_counts[1]:.1f}")

        # Visualize class distribution
        plot_class_distribution(self.df['readmitted_binary'],
                              title='Target Variable: Readmission within 30 Days',
                              output_dir=output_dir,
                              filename='original_class_distribution.png')

        # Statistical tests
        self._statistical_tests(output_dir)

        # Correlation analysis
        print_subsection_header("2.4 Correlation & Covariance Analysis")
        # First convert some numeric-like columns
        self._prepare_for_correlation()
        plot_correlation_matrix(self.df, output_dir, top_n=20)

        return self.df

    def _prepare_for_correlation(self):
        """Convert categorical features to numeric for correlation analysis"""
        # Convert age ranges to numeric
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        if 'age' in self.df.columns:
            self.df['age_numeric'] = self.df['age'].map(age_mapping)

        # Convert Yes/No to 1/0 for medication columns
        yes_no_cols = ['change', 'diabetesMed']
        for col in yes_no_cols:
            if col in self.df.columns:
                self.df[f'{col}_numeric'] = (self.df[col] == 'Yes').astype(int)

    def _statistical_tests(self, output_dir='outputs'):
        """
        Perform Chi-square tests and ANOVA
        """
        print_subsection_header("2.5 Statistical Hypothesis Testing")

        # Chi-square test for categorical variables
        print("\nChi-Square Tests (Categorical Features vs Target):")
        print("-" * 60)

        categorical_cols = ['race', 'gender', 'age', 'admission_type_id',
                           'discharge_disposition_id', 'change', 'diabetesMed']

        chi_square_results = []
        for col in categorical_cols:
            if col in self.df.columns:
                # Create contingency table
                contingency = pd.crosstab(self.df[col].fillna('Missing'),
                                         self.df['readmitted_binary'])

                # Perform chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency)

                chi_square_results.append({
                    'Feature': col,
                    'Chi-Square': chi2,
                    'P-Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })

                print(f"{col:30} | Chi2: {chi2:10.2f} | P-value: {p_value:.4f} | {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

        # ANOVA for numerical variables
        print("\n\nANOVA Tests (Numerical Features vs Target):")
        print("-" * 60)

        numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                         'num_medications', 'number_outpatient', 'number_emergency',
                         'number_inpatient', 'number_diagnoses']

        anova_results = []
        for col in numerical_cols:
            if col in self.df.columns:
                # Separate data by target class
                group_0 = self.df[self.df['readmitted_binary'] == 0][col].dropna()
                group_1 = self.df[self.df['readmitted_binary'] == 1][col].dropna()

                # Perform ANOVA
                f_stat, p_value = f_oneway(group_0, group_1)

                anova_results.append({
                    'Feature': col,
                    'F-Statistic': f_stat,
                    'P-Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })

                print(f"{col:30} | F-stat: {f_stat:10.2f} | P-value: {p_value:.4f} | {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

        print("\n✓ Statistical tests completed")
        print("  Significance levels: *** p<0.001, ** p<0.01, * p<0.05")

    def feature_engineering(self):
        """
        Advanced feature engineering
        """
        print_section_header("STEP 3: ADVANCED FEATURE ENGINEERING")

        print_subsection_header("3.1 Creating New Features")

        # Feature 1: Total number of medication changes
        medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                          'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
                          'rosiglitazone', 'insulin']

        medication_changes = 0
        for col in medication_cols:
            if col in self.df.columns:
                medication_changes += (self.df[col].isin(['Up', 'Down'])).astype(int)

        self.df['medication_changes'] = medication_changes
        print(f"✓ Created 'medication_changes': {self.df['medication_changes'].describe()['mean']:.2f} avg changes")

        # Feature 2: Total procedures
        if all(col in self.df.columns for col in ['num_lab_procedures', 'num_procedures']):
            self.df['total_procedures'] = self.df['num_lab_procedures'] + self.df['num_procedures']
            print(f"✓ Created 'total_procedures': {self.df['total_procedures'].describe()['mean']:.2f} avg procedures")

        # Feature 3: Total visits (outpatient + emergency + inpatient)
        visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
        if all(col in self.df.columns for col in visit_cols):
            self.df['total_visits'] = self.df[visit_cols].sum(axis=1)
            print(f"✓ Created 'total_visits': {self.df['total_visits'].describe()['mean']:.2f} avg visits")

        # Feature 4: Medication complexity (number of different medications)
        medication_binary = 0
        for col in medication_cols:
            if col in self.df.columns:
                medication_binary += (self.df[col] != 'No').astype(int)

        self.df['medication_complexity'] = medication_binary
        print(f"✓ Created 'medication_complexity': {self.df['medication_complexity'].describe()['mean']:.2f} avg medications")

        # Feature 5: Age numeric (already created in correlation prep)
        if 'age_numeric' in self.df.columns:
            print(f"✓ Using 'age_numeric': {self.df['age_numeric'].describe()['mean']:.2f} avg age")

        # Feature 6: High risk flag
        if 'number_inpatient' in self.df.columns:
            self.df['high_risk_patient'] = (self.df['number_inpatient'] > 0).astype(int)
            print(f"✓ Created 'high_risk_patient': {self.df['high_risk_patient'].sum()} high-risk patients")

        print(f"\n✓ Feature engineering completed. Total features: {self.df.shape[1]}")

        return self.df

    def handle_outliers(self, output_dir='outputs'):
        """
        Detect and handle outliers using IQR and Z-score methods
        """
        print_section_header("STEP 4: OUTLIER DETECTION AND TREATMENT")

        numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                         'num_medications', 'number_outpatient', 'number_emergency',
                         'number_inpatient', 'number_diagnoses', 'total_procedures',
                         'total_visits', 'medication_changes', 'medication_complexity']

        # Filter to existing columns
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]

        print_subsection_header("4.1 IQR Method for Outlier Detection")

        outlier_info = []
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_pct = 100 * outliers / len(self.df)

            outlier_info.append({
                'Feature': col,
                'Outliers': outliers,
                'Percentage': outlier_pct,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })

            print(f"{col:30} | Outliers: {outliers:6} ({outlier_pct:5.2f}%) | Bounds: [{lower_bound:6.2f}, {upper_bound:6.2f}]")

        # Visualize outliers
        print("\n✓ Creating boxplots for outlier visualization...")
        plot_outliers_boxplot(self.df, numerical_cols[:12], output_dir)

        print_subsection_header("4.2 Outlier Treatment Strategy")
        print("Strategy: Capping outliers at 99th percentile (Winsorization)")

        for col in numerical_cols:
            upper_limit = self.df[col].quantile(0.99)
            n_capped = (self.df[col] > upper_limit).sum()

            if n_capped > 0:
                self.df[col] = self.df[col].clip(upper=upper_limit)
                print(f"✓ {col:30} | Capped {n_capped:6} values at {upper_limit:.2f}")

        print("\n✓ Outlier treatment completed")

        return self.df

    def encode_and_prepare(self):
        """
        Encode categorical variables and prepare final dataset
        """
        print_section_header("STEP 5: ENCODING AND DATA PREPARATION")

        print_subsection_header("5.1 Handling Missing Values")

        # Drop columns with too many missing values or not useful
        cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code',
                       'medical_specialty', 'readmitted']

        existing_cols_to_drop = [col for col in cols_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_cols_to_drop)
        print(f"✓ Dropped {len(existing_cols_to_drop)} columns with excessive missing values or IDs")

        # Fill missing values in categorical columns with 'Unknown'
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna('Unknown')

        # Fill missing values in numerical columns with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'readmitted_binary' in numerical_cols:
            numerical_cols.remove('readmitted_binary')

        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                print(f"✓ Filled missing in {col} with median: {median_val:.2f}")

        print_subsection_header("5.2 Encoding Categorical Variables")

        # Label encoding for categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        print(f"Encoding {len(categorical_cols)} categorical features:")
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  ✓ {col}: {len(le.classes_)} unique categories")

        print(f"\n✓ All categorical variables encoded")

        # Separate features and target
        X = self.df.drop('readmitted_binary', axis=1)
        y = self.df['readmitted_binary']

        self.feature_names = X.columns.tolist()

        print(f"\n✓ Final dataset prepared:")
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Total features: {len(self.feature_names)}")

        return X, y

    def feature_selection_lasso(self, X, y, output_dir='outputs'):
        """
        Feature selection using Lasso regression with alpha tuning
        """
        print_section_header("STEP 6: FEATURE SELECTION USING LASSO REGRESSION")

        print("Performing Lasso regression with cross-validated alpha tuning...")

        # Standardize features first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Lasso with cross-validation for alpha tuning
        alphas = np.logspace(-4, 1, 50)
        lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
        lasso_cv.fit(X_scaled, y)

        print(f"✓ Best alpha: {lasso_cv.alpha_:.6f}")

        # Get feature coefficients
        coef = lasso_cv.coef_

        # Select features with non-zero coefficients
        selected_features_mask = np.abs(coef) > 0
        selected_features = [self.feature_names[i] for i in range(len(self.feature_names))
                           if selected_features_mask[i]]

        print(f"✓ Selected {len(selected_features)} features out of {len(self.feature_names)}")
        print(f"✓ Reduction: {100 * (1 - len(selected_features)/len(self.feature_names)):.1f}%")

        # Show top features by coefficient magnitude
        feature_coef = list(zip(self.feature_names, np.abs(coef)))
        feature_coef.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 15 features by Lasso coefficient magnitude:")
        for i, (feat, c) in enumerate(feature_coef[:15], 1):
            print(f"  {i:2}. {feat:40} | Coefficient: {c:8.4f}")

        # Use all features if too few selected, or use original if most are selected
        if len(selected_features) < 10:
            print("\n⚠ Too few features selected, using all features")
            selected_features = self.feature_names

        return selected_features, lasso_cv

    def apply_pca(self, X, variance_threshold=0.90, output_dir='outputs'):
        """
        Apply PCA for dimensionality reduction
        """
        print_section_header("STEP 7: DIMENSIONALITY REDUCTION USING PCA")

        print(f"Applying PCA to retain {variance_threshold*100:.0f}% of variance...")

        # Standardize first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=variance_threshold, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        print(f"✓ Original dimensions: {X.shape[1]}")
        print(f"✓ Reduced dimensions: {X_pca.shape[1]}")
        print(f"✓ Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"✓ Reduction: {100 * (1 - X_pca.shape[1]/X.shape[1]):.1f}%")

        # Import and use plot function from utils
        from utils import plot_pca_scree
        plot_pca_scree(pca, output_dir)

        # Show variance explained by top components
        print("\nTop 10 components variance explained:")
        for i, var in enumerate(pca.explained_variance_ratio_[:10], 1):
            cumsum = pca.explained_variance_ratio_[:i].sum()
            print(f"  PC{i:2}: {var*100:5.2f}% | Cumulative: {cumsum*100:6.2f}%")

        self.scaler = scaler

        return X_pca, pca, scaler


def run_preprocessing_pipeline(data_path, output_dir='outputs'):
    """
    Run the complete preprocessing pipeline
    """
    preprocessor = DataPreprocessor(data_path)

    # Step 1: Load data
    df = preprocessor.load_data()

    # Step 2: EDA
    df = preprocessor.exploratory_data_analysis(output_dir)

    # Step 3: Feature engineering
    df = preprocessor.feature_engineering()

    # Step 4: Handle outliers
    df = preprocessor.handle_outliers(output_dir)

    # Step 5: Encode and prepare
    X, y = preprocessor.encode_and_prepare()

    # Step 6: Feature selection
    selected_features, lasso_model = preprocessor.feature_selection_lasso(X, y, output_dir)

    # Use selected features
    X_selected = X[selected_features]

    # Step 7: PCA
    X_pca, pca, scaler = preprocessor.apply_pca(X_selected, variance_threshold=0.90, output_dir=output_dir)

    print_section_header("PREPROCESSING PIPELINE COMPLETED")
    print(f"✓ Final feature matrix shape: {X_selected.shape}")
    print(f"✓ PCA transformed shape: {X_pca.shape}")
    print(f"✓ Target variable shape: {y.shape}")

    return X_selected, X_pca, y, preprocessor, pca, scaler, selected_features

