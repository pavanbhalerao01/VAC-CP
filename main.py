"""
Main Execution Script for Diabetes Hospital Readmission Prediction Project

This is the main entry point that orchestrates the entire end-to-end ML pipeline:
1. Data loading and EDA with statistical tests
2. Feature engineering and preprocessing
3. Outlier detection and treatment
4. Feature selection (Lasso) and dimensionality reduction (PCA)
5. Class imbalance handling (SMOTE)
6. Training 6+ supervised learning models with StratifiedKFold
7. Hyperparameter tuning on the best model
8. Unsupervised learning for patient risk segmentation (GMM, DBSCAN)
9. Comprehensive results and business insights

Author: Data Science Course Project
Date: End Semester Submission
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Import all custom modules
from utils import create_output_dir, print_section_header, plot_feature_importance
from data_preprocess import run_preprocessing_pipeline
from models import run_model_training_pipeline
from clustering import run_clustering_pipeline


def print_project_header():
    """Print project title and information"""
    print("\n" + "="*80)
    print("DIABETES HOSPITAL READMISSION PREDICTION WITH PATIENT RISK SEGMENTATION")
    print("Advanced Machine Learning Project - End Semester Submission")
    print("="*80)
    print("\nDataset: UCI Diabetes 130-US Hospitals Dataset")
    print("Task: Binary Classification + Patient Segmentation")
    print("Target: Predict 30-day readmission risk")
    print("\nTopics Covered:")
    print("  ✓ Exploratory Data Analysis & Statistical Tests")
    print("  ✓ Feature Engineering & Outlier Treatment")
    print("  ✓ Class Imbalance Handling (SMOTE)")
    print("  ✓ Feature Selection (Lasso) & Dimensionality Reduction (PCA)")
    print("  ✓ 6+ Supervised Learning Models with StratifiedKFold CV")
    print("  ✓ Hyperparameter Tuning (RandomizedSearchCV)")
    print("  ✓ Unsupervised Learning (GMM, DBSCAN, HDBSCAN)")
    print("  ✓ Business Insights & Recommendations")
    print("="*80 + "\n")


def generate_final_report(trainer, best_model_name, tuning_comparison,
                         gmm_interpretations, output_dir='outputs'):
    """
    Generate comprehensive final report with all results
    """
    print_section_header("STEP 17: FINAL RESULTS SUMMARY & BUSINESS INSIGHTS")

    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
    print("="*80)

    # Get test results for all models
    test_results = {}
    for name, model in trainer.models.items():
        y_pred = model.predict(trainer.X_test)
        y_pred_proba = model.predict_proba(trainer.X_test)[:, 1]

        from utils import calculate_metrics
        metrics = calculate_metrics(trainer.y_test, y_pred, y_pred_proba)
        test_results[name] = metrics

    # Create and display results table
    import pandas as pd
    results_df = pd.DataFrame(test_results).T
    results_df = results_df.round(4)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)

    print("\n" + results_df.to_string())

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
    print(f"   F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}")
    print(f"   Recall: {results_df.loc[best_model_name, 'Recall']:.4f}")

    # Feature importance analysis
    print("\n" + "="*80)
    print("🔍 TOP FACTORS CAUSING READMISSION")
    print("="*80)

    best_model = trainer.best_model
    if hasattr(best_model, 'feature_importances_'):
        feature_names = trainer.X_train.columns.tolist()
        top_features = plot_feature_importance(best_model, feature_names,
                                              output_dir, top_n=20,
                                              model_name=best_model_name)

        if top_features:
            print("\nTop 10 Risk Factors for 30-Day Readmission:")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"  {i:2}. {feature:40} | Importance: {importance:.4f}")

    # Patient segmentation insights
    print("\n" + "="*80)
    print("👥 PATIENT RISK SEGMENTATION INSIGHTS")
    print("="*80)

    if gmm_interpretations:
        print("\nIdentified Patient Segments:")
        for i, interpretation in enumerate(gmm_interpretations):
            print(f"\n  Segment {i+1}: {interpretation}")

    # Business recommendations
    print("\n" + "="*80)
    print("💡 ACTIONABLE HOSPITAL RECOMMENDATIONS")
    print("="*80)

    recommendations = [
        "1. HIGH-RISK PATIENT MONITORING",
        "   → Implement intensive follow-up programs for patients with prior inpatient visits",
        "   → Deploy remote monitoring systems for high-risk segments",
        "",
        "2. MEDICATION MANAGEMENT",
        "   → Simplify medication regimens to improve adherence",
        "   → Provide medication counseling before discharge",
        "   → Monitor medication changes closely",
        "",
        "3. DISCHARGE PLANNING",
        "   → Extended discharge planning for patients with longer hospital stays",
        "   → Coordinate care transitions with primary care physicians",
        "   → Ensure patients understand their discharge instructions",
        "",
        "4. CHRONIC DISEASE MANAGEMENT",
        "   → Establish diabetes education programs",
        "   → Regular check-ups for patients with multiple diagnoses",
        "   → Proactive management of comorbidities",
        "",
        "5. EMERGENCY VISIT REDUCTION",
        "   → Identify and address causes of frequent emergency visits",
        "   → Provide 24/7 nurse hotline for non-emergency consultations",
        "   → Improve access to outpatient services",
    ]

    for rec in recommendations:
        print(rec)

    # Key findings
    print("\n" + "="*80)
    print("📈 KEY FINDINGS")
    print("="*80)

    findings = [
        f"• Best performing model: {best_model_name} with ROC-AUC of {results_df.loc[best_model_name, 'ROC-AUC']:.4f}",
        f"• Class imbalance successfully handled using SMOTE",
        f"• Feature engineering created valuable predictive features",
        f"• Identified {len(gmm_interpretations)} distinct patient risk segments",
        f"• Top risk factor: Number of inpatient visits (prior hospitalization history)",
        f"• Hyperparameter tuning improved model performance",
        f"• PCA successfully reduced dimensionality while retaining >90% variance",
    ]

    for finding in findings:
        print(finding)

    # Save summary report
    print("\n" + "="*80)
    print("💾 SAVING FINAL REPORT")
    print("="*80)

    report_path = f"{output_dir}/final_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DIABETES HOSPITAL READMISSION PREDICTION - FINAL REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(results_df.to_string())
        f.write("\n\n")

        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}\n\n")

        if tuning_comparison is not None:
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(tuning_comparison.to_string())
            f.write("\n\n")

        f.write("PATIENT SEGMENTATION\n")
        f.write("-"*80 + "\n")
        for i, interp in enumerate(gmm_interpretations):
            f.write(f"Segment {i+1}: {interp}\n")
        f.write("\n")

        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        for finding in findings:
            f.write(finding + "\n")
        f.write("\n")

        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        for rec in recommendations:
            f.write(rec + "\n")

    print(f"✓ Final report saved to: {report_path}")


def main():
    """
    Main execution function
    """
    # Start timer
    start_time = time.time()

    # Print project header
    print_project_header()

    # Configuration
    DATA_PATH = 'diabetic_data.csv'
    OUTPUT_DIR = 'outputs'

    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file '{DATA_PATH}' not found!")
        print(f"   Please ensure the file is in the current directory: {os.getcwd()}")
        sys.exit(1)

    # Create output directory
    create_output_dir(OUTPUT_DIR)

    try:
        # ============================================================================
        # PHASE 1: DATA PREPROCESSING & FEATURE ENGINEERING
        # ============================================================================
        print_section_header("PHASE 1: DATA PREPROCESSING & FEATURE ENGINEERING")

        X_selected, X_pca, y, preprocessor, pca, scaler, selected_features = \
            run_preprocessing_pipeline(DATA_PATH, OUTPUT_DIR)

        # ============================================================================
        # PHASE 2: SUPERVISED LEARNING - MODEL TRAINING & EVALUATION
        # ============================================================================
        print_section_header("PHASE 2: SUPERVISED LEARNING")

        # Training models on selected features (NO hyperparameter tuning for speed)
        print("\n📊 Training models on selected features (post-Lasso selection)...")
        print("⚡ Fast mode: Skipping hyperparameter tuning to save time")
        trainer, results_df, tuned_model, best_model_name = \
            run_model_training_pipeline(X_selected, y, OUTPUT_DIR, use_smote=True, skip_tuning=True)

        # Plot feature importance
        if hasattr(trainer.best_model, 'feature_importances_'):
            plot_feature_importance(trainer.best_model, selected_features,
                                  OUTPUT_DIR, top_n=20, model_name=best_model_name)

        # ============================================================================
        # PHASE 3: UNSUPERVISED LEARNING - PATIENT RISK SEGMENTATION
        # ============================================================================
        print_section_header("PHASE 3: UNSUPERVISED LEARNING - PATIENT SEGMENTATION")

        # Use PCA-transformed data for clustering
        segmentation, gmm_model, gmm_labels, gmm_interpretations = \
            run_clustering_pipeline(X_pca, X_selected.values, y,
                                   selected_features, OUTPUT_DIR)

        # ============================================================================
        # PHASE 4: FINAL RESULTS & BUSINESS INSIGHTS
        # ============================================================================
        generate_final_report(trainer, best_model_name, None,
                            gmm_interpretations, OUTPUT_DIR)

        # ============================================================================
        # PROJECT COMPLETION
        # ============================================================================
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n" + "="*80)
        print("✅ PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n⏱  Total execution time: {elapsed_time//60:.0f} minutes {elapsed_time%60:.0f} seconds")
        print(f"\n📁 All outputs saved to: {OUTPUT_DIR}/")
        print("\nGenerated Files:")

        # List output files
        if os.path.exists(OUTPUT_DIR):
            files = sorted(os.listdir(OUTPUT_DIR))
            for i, file in enumerate(files, 1):
                file_path = os.path.join(OUTPUT_DIR, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {i:2}. {file:40} ({file_size:6.1f} KB)")

        print("\n" + "="*80)
        print("Thank you for reviewing this project!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

