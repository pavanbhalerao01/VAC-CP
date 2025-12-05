"""
Utility Functions for Visualization and Reporting
This module contains helper functions for plotting, metrics calculation, and result reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                            auc, accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score)
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_output_dir(output_dir='outputs'):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    return output_dir

def plot_missing_values(df, output_dir='outputs'):
    """
    Visualize missing values in the dataset
    """
    print("\n" + "="*80)
    print("MISSING VALUE ANALYSIS")
    print("="*80)

    missing_counts = df.isnull().sum()
    missing_percent = 100 * missing_counts / len(df)
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percent': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    if len(missing_df) > 0:
        print(f"\nColumns with missing values:\n{missing_df.to_string(index=False)}")

        plt.figure(figsize=(12, 6))
        plt.barh(missing_df['Column'][:20], missing_df['Missing_Percent'][:20], color='coral')
        plt.xlabel('Missing Percentage (%)', fontsize=12)
        plt.title('Top 20 Columns with Missing Values', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/missing_values.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/missing_values.png")
        plt.close()
    else:
        print("\n✓ No missing values found in the dataset!")

    return missing_df

def plot_correlation_matrix(df, output_dir='outputs', top_n=20):
    """
    Plot correlation and covariance matrices
    """
    print("\n" + "="*80)
    print("CORRELATION & COVARIANCE ANALYSIS")
    print("="*80)

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > top_n:
        # Select top N most correlated with target
        if 'readmitted_binary' in numeric_cols:
            target_corr = df[numeric_cols].corr()['readmitted_binary'].abs().sort_values(ascending=False)
            top_cols = target_corr.head(top_n).index.tolist()
        else:
            top_cols = numeric_cols[:top_n]
    else:
        top_cols = numeric_cols

    # Correlation matrix
    corr_matrix = df[top_cols].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Matrix (Top {len(top_cols)} Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/correlation_matrix.png")
    plt.close()

    # Covariance matrix
    cov_matrix = df[top_cols].cov()

    plt.figure(figsize=(14, 12))
    sns.heatmap(cov_matrix, annot=False, cmap='viridis', square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Covariance Matrix (Top {len(top_cols)} Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/covariance_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/covariance_matrix.png")
    plt.close()

    print(f"\n✓ Analyzed {len(top_cols)} numeric features")
    return corr_matrix

def plot_class_distribution(y, title='Class Distribution', output_dir='outputs', filename='class_distribution.png'):
    """
    Visualize class distribution
    """
    plt.figure(figsize=(10, 6))

    if isinstance(y, pd.Series):
        class_counts = y.value_counts()
    else:
        unique, counts = np.unique(y, return_counts=True)
        class_counts = pd.Series(counts, index=unique)

    bars = plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'salmon'])
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add percentage labels
    total = class_counts.sum()
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({100*count/total:.1f}%)',
                ha='center', va='bottom', fontsize=11)

    plt.xticks(class_counts.index)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/{filename}")
    plt.close()

def plot_pca_scree(pca, output_dir='outputs'):
    """
    Plot PCA scree plot showing variance explained
    """
    plt.figure(figsize=(12, 5))

    # Subplot 1: Individual variance explained
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Variance Explained Ratio', fontsize=12)
    plt.title('Scree Plot - Individual Variance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Cumulative variance explained
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% Variance')
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Variance Explained', fontsize=12)
    plt.title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_scree_plot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/pca_scree_plot.png")
    plt.close()

    print(f"\n✓ PCA retained {len(pca.explained_variance_ratio_)} components")
    print(f"✓ Total variance explained: {cumsum[-1]*100:.2f}%")

def plot_roc_curves(models_dict, X_test, y_test, output_dir='outputs'):
    """
    Plot ROC curves for all models on the same plot
    """
    plt.figure(figsize=(12, 8))

    for name, model in models_dict.items():
        # Get predicted probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/roc_curves_comparison.png")
    plt.close()

def plot_confusion_matrices(models_dict, X_test, y_test, output_dir='outputs'):
    """
    Plot confusion matrices for all models
    """
    n_models = len(models_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)

    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/confusion_matrices.png")
    plt.close()

def plot_feature_importance(model, feature_names, output_dir='outputs', top_n=20, model_name='Best Model'):
    """
    Plot feature importance from tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/feature_importance.png")
        plt.close()

        # Return top features
        top_features = [(feature_names[i], importances[i]) for i in indices[:10]]
        return top_features
    else:
        print("⚠ Model does not have feature_importances_ attribute")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate all evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_pred_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)

    return metrics

def create_results_table(results_dict):
    """
    Create a formatted results table from metrics dictionary
    """
    df_results = pd.DataFrame(results_dict).T
    df_results = df_results.round(4)
    return df_results

def print_section_header(title):
    """
    Print a formatted section header
    """
    print("\n" + "="*80)
    print(f"{title.center(80)}")
    print("="*80)

def print_subsection_header(title):
    """
    Print a formatted subsection header
    """
    print("\n" + "-"*80)
    print(f"  {title}")
    print("-"*80)

def plot_outliers_boxplot(df, columns, output_dir='outputs'):
    """
    Create boxplots to visualize outliers
    """
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if len(columns) > 1 else [axes]

    for idx, col in enumerate(columns[:len(axes)]):
        if col in df.columns:
            axes[idx].boxplot(df[col].dropna(), vert=True)
            axes[idx].set_title(col, fontsize=10)
            axes[idx].grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/outliers_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/outliers_boxplot.png")
    plt.close()

def plot_clusters_2d(X_2d, labels, title='Cluster Visualization', output_dir='outputs', filename='clusters_2d.png'):
    """
    Plot 2D visualization of clusters
    """
    plt.figure(figsize=(12, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points (for DBSCAN)
            color = 'gray'
            marker = 'x'
            label_name = 'Noise'
        else:
            marker = 'o'
            label_name = f'Cluster {label}'

        mask = labels == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color],
                   label=label_name, marker=marker, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)

    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/{filename}")
    plt.close()

def print_cluster_analysis(df, labels, cluster_name='Cluster'):
    """
    Print detailed cluster analysis
    """
    print(f"\n{cluster_name} Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Cluster {label}: {count} patients ({100*count/len(labels):.1f}%)")

    # Add cluster labels to dataframe for analysis
    df_clustered = df.copy()
    df_clustered['cluster'] = labels

    print(f"\n{cluster_name} Characteristics:")
    numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')

    # Show mean values for each cluster (top 5 numeric features)
    if len(numeric_cols) > 0:
        for label in unique:
            if label != -1:  # Skip noise cluster
                cluster_data = df_clustered[df_clustered['cluster'] == label]
                print(f"\n  Cluster {label} (n={len(cluster_data)}):")

                # Show means of some key features
                for col in numeric_cols[:5]:
                    mean_val = cluster_data[col].mean()
                    print(f"    - {col}: {mean_val:.2f}")

