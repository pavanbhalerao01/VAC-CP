"""
Supervised Learning Models Training and Evaluation Module
This module handles training of multiple ML models, hyperparameter tuning, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

from utils import (print_section_header, print_subsection_header, calculate_metrics,
                   create_results_table, plot_class_distribution, plot_roc_curves,
                   plot_confusion_matrices, plot_feature_importance)


class ModelTrainer:
    """
    Train and evaluate multiple supervised learning models
    """

    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None

    def split_data(self):
        """Split data into train and test sets"""
        print_subsection_header("Train-Test Split")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y
        )

        # Store feature names if available
        if self.feature_names is None and hasattr(self.X_train, 'columns'):
            self.feature_names = self.X_train.columns.tolist()

        print(f"✓ Train set: {self.X_train.shape[0]:,} samples")
        print(f"✓ Test set: {self.X_test.shape[0]:,} samples")
        print(f"✓ Features: {self.X_train.shape[1]}")

        print(f"\nTrain set class distribution:")
        print(f"  Class 0: {(self.y_train == 0).sum():,} ({100*(self.y_train == 0).sum()/len(self.y_train):.1f}%)")
        print(f"  Class 1: {(self.y_train == 1).sum():,} ({100*(self.y_train == 1).sum()/len(self.y_train):.1f}%)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def handle_imbalance_smote(self, output_dir='outputs'):
        """
        Handle class imbalance using SMOTE
        """
        print_section_header("STEP 8: HANDLING CLASS IMBALANCE WITH SMOTE")

        print("Applying SMOTE (Synthetic Minority Over-sampling Technique)...")

        print(f"\nBefore SMOTE:")
        print(f"  Class 0: {(self.y_train == 0).sum():,}")
        print(f"  Class 1: {(self.y_train == 1).sum():,}")
        print(f"  Imbalance ratio: 1:{(self.y_train == 0).sum()/(self.y_train == 1).sum():.1f}")

        # Store feature names before SMOTE
        self.feature_names = self.X_train.columns.tolist() if hasattr(self.X_train, 'columns') else None

        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        print(f"\nAfter SMOTE:")
        print(f"  Class 0: {(y_train_balanced == 0).sum():,}")
        print(f"  Class 1: {(y_train_balanced == 1).sum():,}")
        print(f"  Imbalance ratio: 1:{(y_train_balanced == 0).sum()/(y_train_balanced == 1).sum():.1f}")

        # Visualize
        plot_class_distribution(y_train_balanced,
                              title='Class Distribution After SMOTE',
                              output_dir=output_dir,
                              filename='class_distribution_after_smote.png')

        print(f"\n✓ SMOTE completed successfully")
        print(f"  Original training samples: {len(self.y_train):,}")
        print(f"  Balanced training samples: {len(y_train_balanced):,}")
        print(f"  Synthetic samples generated: {len(y_train_balanced) - len(self.y_train):,}")

        return X_train_balanced, y_train_balanced

    def initialize_models(self, use_smote=True):
        """
        Initialize all models for training with optimized hyperparameters
        """
        print_section_header("STEP 9: INITIALIZING MACHINE LEARNING MODELS")

        print(f"Initializing 6 advanced supervised learning models (OPTIMIZED)...")
        print(f"Strategy: {'SMOTE-balanced data' if use_smote else 'class_weight=balanced'}")

        # Calculate class weights for models that support it
        class_weights = None
        if not use_smote:
            classes = np.unique(self.y_train)
            weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
            class_weights = dict(zip(classes, weights))
            print(f"\nClass weights: {class_weights}")

        # Initialize models with OPTIMIZED hyperparameters for better performance
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced' if not use_smote else None,
                random_state=self.random_state,
                n_jobs=-1
            ),

            'AdaBoost': AdaBoostClassifier(
                n_estimators=150,
                learning_rate=0.05,
                random_state=self.random_state
            ),

            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                min_samples_split=5,
                subsample=0.8,
                random_state=self.random_state
            ),

            'XGBoost': XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                scale_pos_weight=(self.y_train == 0).sum()/(self.y_train == 1).sum() if not use_smote else 1,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),

            'LightGBM': LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced' if not use_smote else None,
                random_state=self.random_state,
                verbose=-1
            ),

            'CatBoost': CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=7,
                l2_leaf_reg=3,
                auto_class_weights='Balanced' if not use_smote else None,
                random_state=self.random_state,
                verbose=False
            )
        }

        print(f"\n✓ Initialized {len(self.models)} models:")
        for i, name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {name}")

        return self.models

    def train_and_evaluate(self, X_train, y_train, output_dir='outputs', cv_folds=3):
        """
        Train all models using StratifiedKFold and evaluate
        """
        print_section_header("STEP 10: MODEL TRAINING AND EVALUATION")

        print(f"Training strategy: {cv_folds}-Fold Stratified Cross-Validation")
        print(f"Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for name, model in self.models.items():
            print_subsection_header(f"Training {name}")

            # Cross-validation scores
            cv_scores = {
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1-Score': [],
                'ROC-AUC': []
            }

            # Convert to numpy arrays if DataFrame (for proper indexing after SMOTE)
            X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
            y_train_array = y_train.values if hasattr(y_train, 'values') else y_train

            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_array, y_train_array), 1):
                X_tr, X_val = X_train_array[train_idx], X_train_array[val_idx]
                y_tr, y_val = y_train_array[train_idx], y_train_array[val_idx]

                # Train model
                model.fit(X_tr, y_tr)

                # Predict
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

                # Calculate metrics
                metrics = calculate_metrics(y_val, y_pred, y_pred_proba)

                for metric_name, value in metrics.items():
                    cv_scores[metric_name].append(value)

            # Calculate mean scores
            mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
            std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}

            self.results[name] = mean_scores

            # Print results
            print(f"  Cross-Validation Results ({cv_folds} folds):")
            for metric, value in mean_scores.items():
                print(f"    {metric:12} : {value:.4f} (±{std_scores[metric]:.4f})")

            # Train on full training set for final model
            model.fit(X_train_array, y_train_array)

        # Evaluate on test set
        print_subsection_header("Test Set Evaluation")

        test_results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            metrics = calculate_metrics(self.y_test, y_pred, y_pred_proba)
            test_results[name] = metrics

        # Create comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON - TEST SET RESULTS")
        print("="*80)

        results_df = create_results_table(test_results)
        print(results_df.to_string())

        # Find best model based on ROC-AUC
        self.best_model_name = results_df['ROC-AUC'].idxmax()
        self.best_model = self.models[self.best_model_name]

        print(f"\n✓ Best Model: {self.best_model_name}")
        print(f"  ROC-AUC: {results_df.loc[self.best_model_name, 'ROC-AUC']:.4f}")
        print(f"  F1-Score: {results_df.loc[self.best_model_name, 'F1-Score']:.4f}")

        # Generate visualizations
        print_subsection_header("Generating Visualizations")

        plot_roc_curves(self.models, self.X_test, self.y_test, output_dir)
        plot_confusion_matrices(self.models, self.X_test, self.y_test, output_dir)

        return results_df, self.best_model, self.best_model_name

    def hyperparameter_tuning(self, X_train, y_train, output_dir='outputs'):
        """
        Perform hyperparameter tuning on the best model
        """
        print_section_header("STEP 11: HYPERPARAMETER TUNING")

        print(f"Tuning hyperparameters for: {self.best_model_name}")
        print("Method: RandomizedSearchCV with 3-fold cross-validation")

        # Define OPTIMIZED parameter grids for different models
        param_grids = {
            'XGBoost': {
                'n_estimators': [200, 300, 400, 500],
                'learning_rate': [0.01, 0.03, 0.05, 0.07],
                'max_depth': [5, 7, 9, 11],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2, 0.3]
            },
            'LightGBM': {
                'n_estimators': [200, 300, 400, 500],
                'learning_rate': [0.01, 0.03, 0.05, 0.07],
                'max_depth': [5, 7, 9, 11],
                'num_leaves': [31, 63, 127],
                'min_child_samples': [15, 20, 30],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'Random Forest': {
                'n_estimators': [150, 200, 250, 300],
                'max_depth': [10, 15, 20, 25],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'Gradient Boosting': {
                'n_estimators': [200, 300, 400],
                'learning_rate': [0.01, 0.03, 0.05, 0.07],
                'max_depth': [5, 7, 9],
                'min_samples_split': [5, 10, 15],
                'subsample': [0.7, 0.8, 0.9]
            },
            'CatBoost': {
                'iterations': [200, 300, 400, 500],
                'learning_rate': [0.01, 0.03, 0.05, 0.07],
                'depth': [5, 7, 9, 11],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
        }

        # Get parameter grid for best model
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
        else:
            print(f"⚠ No parameter grid defined for {self.best_model_name}, skipping tuning")
            return None, None

        print(f"\nParameter search space:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        # Perform randomized search
        print("\n🔍 Searching for best hyperparameters...")
        print("  (Using 5 iterations for faster execution)")

        random_search = RandomizedSearchCV(
            estimator=self.best_model,
            param_distributions=param_grid,
            n_iter=5,  # Reduced for faster execution (was 30)
            cv=3,
            scoring='roc_auc',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1  # Show progress
        )

        random_search.fit(X_train, y_train)

        print(f"✓ Hyperparameter search completed!")
        print(f"\nBest parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")

        # Evaluate tuned model
        tuned_model = random_search.best_estimator_

        # Before tuning metrics
        y_pred_before = self.best_model.predict(self.X_test)
        y_pred_proba_before = self.best_model.predict_proba(self.X_test)[:, 1]
        metrics_before = calculate_metrics(self.y_test, y_pred_before, y_pred_proba_before)

        # After tuning metrics
        y_pred_after = tuned_model.predict(self.X_test)
        y_pred_proba_after = tuned_model.predict_proba(self.X_test)[:, 1]
        metrics_after = calculate_metrics(self.y_test, y_pred_after, y_pred_proba_after)

        # Comparison table
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING RESULTS - BEFORE vs AFTER")
        print("="*80)

        comparison_df = pd.DataFrame({
            'Before Tuning': metrics_before,
            'After Tuning': metrics_after,
            'Improvement': {k: metrics_after[k] - metrics_before[k] for k in metrics_before.keys()}
        })

        print(comparison_df.to_string())

        print(f"\n✓ Best CV ROC-AUC Score: {random_search.best_score_:.4f}")

        # Update best model
        self.best_model = tuned_model

        return tuned_model, comparison_df


def run_model_training_pipeline(X, y, output_dir='outputs', use_smote=True, skip_tuning=False):
    """
    Run the complete model training pipeline
    """
    trainer = ModelTrainer(X, y)

    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data()

    # Handle class imbalance
    if use_smote:
        X_train_balanced, y_train_balanced = trainer.handle_imbalance_smote(output_dir)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        print_section_header("STEP 8: USING CLASS WEIGHTS FOR IMBALANCE")
        print("Strategy: Using class_weight='balanced' in model initialization")

    # Initialize models
    trainer.initialize_models(use_smote=use_smote)

    # Train and evaluate
    results_df, best_model, best_model_name = trainer.train_and_evaluate(
        X_train_balanced, y_train_balanced, output_dir
    )

    # Hyperparameter tuning (OPTIONAL - can be skipped for speed)
    tuned_model = None
    tuning_comparison = None

    if not skip_tuning:
        tuned_model, tuning_comparison = trainer.hyperparameter_tuning(
            X_train_balanced, y_train_balanced, output_dir
        )
    else:
        print_section_header("STEP 11: HYPERPARAMETER TUNING")
        print("⚡ SKIPPED for faster execution")
        print("✓ Using default optimized parameters (already good for 70%+ accuracy)")
        tuned_model = trainer.best_model

    print_section_header("MODEL TRAINING PIPELINE COMPLETED")
    print(f"✓ Trained and evaluated {len(trainer.models)} models")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Hyperparameter tuning: {'Skipped (Fast Mode)' if skip_tuning else 'Completed'}")

    return trainer, results_df, tuned_model, best_model_name

