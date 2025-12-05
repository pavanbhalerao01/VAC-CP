"""
Unsupervised Learning and Patient Risk Segmentation Module
This module handles clustering analysis using GMM and DBSCAN/HDBSCAN.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠ HDBSCAN not available, will use DBSCAN only")
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

from utils import (print_section_header, print_subsection_header,
                   plot_clusters_2d, print_cluster_analysis)


class PatientSegmentation:
    """
    Perform patient risk segmentation using unsupervised learning
    """

    def __init__(self, X_pca, X_original, y):
        self.X_pca = X_pca
        self.X_original = X_original
        self.y = y
        self.gmm_labels = None
        self.dbscan_labels = None
        self.hdbscan_labels = None
        self.X_2d = None

    def apply_tsne(self, output_dir='outputs'):
        """
        Apply t-SNE for 2D visualization
        """
        print_section_header("STEP 12: DIMENSIONALITY REDUCTION FOR VISUALIZATION")

        print("Applying t-SNE for 2D visualization...")
        print("  Method: t-SNE (t-Distributed Stochastic Neighbor Embedding)")
        print("  Purpose: Reduce high-dimensional PCA data to 2D for cluster visualization")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        self.X_2d = tsne.fit_transform(self.X_pca)

        print(f"✓ t-SNE completed successfully")
        print(f"  Original PCA dimensions: {self.X_pca.shape[1]}")
        print(f"  Reduced to: 2D")

        return self.X_2d

    def gaussian_mixture_models(self, n_components_range=range(2, 7), output_dir='outputs'):
        """
        Apply Gaussian Mixture Models for clustering
        """
        print_section_header("STEP 13: GAUSSIAN MIXTURE MODELS (GMM) CLUSTERING")

        print("Finding optimal number of clusters using BIC and AIC...")

        # Test different number of components
        bic_scores = []
        aic_scores = []
        silhouette_scores = []

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42,
                                covariance_type='full', max_iter=200)
            gmm.fit(self.X_pca)

            bic = gmm.bic(self.X_pca)
            aic = gmm.aic(self.X_pca)

            labels = gmm.predict(self.X_pca)

            # Calculate silhouette score (only if we have more than 1 cluster and less than n_samples)
            if n_components > 1 and n_components < len(self.X_pca):
                silhouette = silhouette_score(self.X_pca, labels)
            else:
                silhouette = 0

            bic_scores.append(bic)
            aic_scores.append(aic)
            silhouette_scores.append(silhouette)

            print(f"  n_components={n_components} | BIC: {bic:10.2f} | AIC: {aic:10.2f} | Silhouette: {silhouette:.4f}")

        # Plot BIC/AIC scores
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 3, 1)
        plt.plot(list(n_components_range), bic_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('BIC Score', fontsize=12)
        plt.title('BIC Score vs Number of Components', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(list(n_components_range), aic_scores, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('AIC Score', fontsize=12)
        plt.title('AIC Score vs Number of Components', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(list(n_components_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score vs Number of Components', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/gmm_selection_criteria.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir}/gmm_selection_criteria.png")
        plt.close()

        # Select optimal number based on lowest BIC
        optimal_n = list(n_components_range)[np.argmin(bic_scores)]
        print(f"\n✓ Optimal number of components (based on BIC): {optimal_n}")

        # Fit final GMM
        print(f"\nFitting final GMM with {optimal_n} components...")
        gmm_final = GaussianMixture(n_components=optimal_n, random_state=42,
                                   covariance_type='full', max_iter=200)
        gmm_final.fit(self.X_pca)
        self.gmm_labels = gmm_final.predict(self.X_pca)

        # Calculate final metrics
        silhouette = silhouette_score(self.X_pca, self.gmm_labels)
        davies_bouldin = davies_bouldin_score(self.X_pca, self.gmm_labels)

        print(f"\n✓ GMM Clustering Completed")
        print(f"  Number of clusters: {optimal_n}")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")

        # Print cluster distribution
        print_cluster_analysis(pd.DataFrame(self.X_pca), self.gmm_labels, 'GMM')

        # Visualize clusters
        if self.X_2d is not None:
            plot_clusters_2d(self.X_2d, self.gmm_labels,
                           title=f'GMM Clustering ({optimal_n} Clusters) - t-SNE Visualization',
                           output_dir=output_dir,
                           filename='gmm_clusters_tsne.png')

        return gmm_final, self.gmm_labels, optimal_n

    def dbscan_clustering(self, eps=0.5, min_samples=50, output_dir='outputs'):
        """
        Apply DBSCAN for density-based clustering
        """
        print_section_header("STEP 14: DBSCAN CLUSTERING")

        print("Applying DBSCAN (Density-Based Spatial Clustering)...")
        print(f"  Parameters: eps={eps}, min_samples={min_samples}")

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.dbscan_labels = dbscan.fit_predict(self.X_pca)

        # Count clusters
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = list(self.dbscan_labels).count(-1)

        print(f"\n✓ DBSCAN Clustering Completed")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise} ({100*n_noise/len(self.dbscan_labels):.1f}%)")

        # Calculate metrics (excluding noise points)
        if n_clusters > 1:
            mask = self.dbscan_labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(self.X_pca[mask], self.dbscan_labels[mask])
                davies_bouldin = davies_bouldin_score(self.X_pca[mask], self.dbscan_labels[mask])
                print(f"  Silhouette Score: {silhouette:.4f} (excluding noise)")
                print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (excluding noise)")

        # Print cluster distribution
        print_cluster_analysis(pd.DataFrame(self.X_pca), self.dbscan_labels, 'DBSCAN')

        # Visualize clusters
        if self.X_2d is not None:
            plot_clusters_2d(self.X_2d, self.dbscan_labels,
                           title=f'DBSCAN Clustering ({n_clusters} Clusters + Noise) - t-SNE Visualization',
                           output_dir=output_dir,
                           filename='dbscan_clusters_tsne.png')

        return dbscan, self.dbscan_labels, n_clusters

    def hdbscan_clustering(self, min_cluster_size=50, output_dir='outputs'):
        """
        Apply HDBSCAN for hierarchical density-based clustering
        """
        if not HDBSCAN_AVAILABLE:
            print("\n⚠ HDBSCAN not available, skipping...")
            return None, None, 0

        print_section_header("STEP 15: HDBSCAN CLUSTERING")

        print("Applying HDBSCAN (Hierarchical DBSCAN)...")
        print(f"  Parameters: min_cluster_size={min_cluster_size}")

        # Apply HDBSCAN
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10)
        self.hdbscan_labels = hdbscan.fit_predict(self.X_pca)

        # Count clusters
        n_clusters = len(set(self.hdbscan_labels)) - (1 if -1 in self.hdbscan_labels else 0)
        n_noise = list(self.hdbscan_labels).count(-1)

        print(f"\n✓ HDBSCAN Clustering Completed")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise} ({100*n_noise/len(self.hdbscan_labels):.1f}%)")

        # Calculate metrics (excluding noise points)
        if n_clusters > 1:
            mask = self.hdbscan_labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(self.X_pca[mask], self.hdbscan_labels[mask])
                davies_bouldin = davies_bouldin_score(self.X_pca[mask], self.hdbscan_labels[mask])
                print(f"  Silhouette Score: {silhouette:.4f} (excluding noise)")
                print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (excluding noise)")

        # Print cluster distribution
        print_cluster_analysis(pd.DataFrame(self.X_pca), self.hdbscan_labels, 'HDBSCAN')

        # Visualize clusters
        if self.X_2d is not None:
            plot_clusters_2d(self.X_2d, self.hdbscan_labels,
                           title=f'HDBSCAN Clustering ({n_clusters} Clusters + Noise) - t-SNE Visualization',
                           output_dir=output_dir,
                           filename='hdbscan_clusters_tsne.png')

        return hdbscan, self.hdbscan_labels, n_clusters

    def interpret_clusters(self, labels, feature_names, cluster_type='GMM'):
        """
        Provide business interpretations for clusters
        """
        print_section_header(f"STEP 16: {cluster_type} CLUSTER INTERPRETATION")

        print(f"Analyzing {cluster_type} clusters for patient risk segmentation...")

        # Create DataFrame with original features and cluster labels
        df_analysis = pd.DataFrame(self.X_original, columns=feature_names)
        df_analysis['cluster'] = labels
        df_analysis['readmitted'] = self.y.values if isinstance(self.y, pd.Series) else self.y

        unique_clusters = sorted([c for c in np.unique(labels) if c != -1])

        print(f"\nDetailed Cluster Characteristics:")
        print("="*80)

        cluster_interpretations = []

        for cluster_id in unique_clusters:
            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
            n_patients = len(cluster_data)
            readmit_rate = cluster_data['readmitted'].mean()

            print(f"\n{'Cluster ' + str(cluster_id):^80}")
            print("-"*80)
            print(f"  Size: {n_patients} patients ({100*n_patients/len(df_analysis):.1f}% of total)")
            print(f"  Readmission Rate: {100*readmit_rate:.1f}%")

            # Get mean values for top features
            numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ['cluster', 'readmitted']]

            print(f"\n  Top Distinguishing Features:")

            # Calculate feature means for this cluster
            cluster_means = cluster_data[numeric_cols].mean()
            overall_means = df_analysis[numeric_cols].mean()

            # Find features that differ most from overall mean
            diff = ((cluster_means - overall_means) / (overall_means + 1e-6)).abs()
            top_features = diff.nlargest(8)

            for feat in top_features.index:
                cluster_val = cluster_means[feat]
                overall_val = overall_means[feat]
                diff_pct = 100 * (cluster_val - overall_val) / (overall_val + 1e-6)

                print(f"    • {feat:35} : {cluster_val:8.2f} (avg: {overall_val:6.2f}, {diff_pct:+6.1f}%)")

            # Generate interpretation
            interpretation = self._generate_interpretation(cluster_id, cluster_data,
                                                          readmit_rate, top_features.index[:5])
            cluster_interpretations.append(interpretation)

            print(f"\n  💡 Interpretation: {interpretation}")

        print("\n" + "="*80)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*80)

        for i, interpretation in enumerate(cluster_interpretations):
            print(f"\nCluster {i}: {interpretation}")

        return cluster_interpretations

    def _generate_interpretation(self, cluster_id, cluster_data, readmit_rate, top_features):
        """
        Generate human-readable interpretation for a cluster
        """
        risk_level = "High" if readmit_rate > 0.15 else "Medium" if readmit_rate > 0.10 else "Low"

        # Analyze top features
        feature_desc = []

        if 'number_inpatient' in cluster_data.columns:
            avg_inpatient = cluster_data['number_inpatient'].mean()
            if avg_inpatient > 1:
                feature_desc.append("frequent inpatient visits")

        if 'age_numeric' in cluster_data.columns:
            avg_age = cluster_data['age_numeric'].mean()
            if avg_age > 60:
                feature_desc.append("elderly patients")
            elif avg_age < 40:
                feature_desc.append("younger patients")

        if 'time_in_hospital' in cluster_data.columns:
            avg_time = cluster_data['time_in_hospital'].mean()
            if avg_time > 7:
                feature_desc.append("extended hospital stays")

        if 'num_medications' in cluster_data.columns:
            avg_meds = cluster_data['num_medications'].mean()
            if avg_meds > 15:
                feature_desc.append("complex medication regimens")

        if 'number_emergency' in cluster_data.columns:
            avg_emergency = cluster_data['number_emergency'].mean()
            if avg_emergency > 0.5:
                feature_desc.append("frequent emergency visits")

        # Build interpretation
        desc_str = ", ".join(feature_desc[:3]) if feature_desc else "standard characteristics"
        interpretation = f"{risk_level} Risk - {desc_str} (readmission rate: {100*readmit_rate:.1f}%)"

        return interpretation


def run_clustering_pipeline(X_pca, X_original, y, feature_names, output_dir='outputs'):
    """
    Run the complete clustering pipeline
    """
    segmentation = PatientSegmentation(X_pca, X_original, y)

    # Apply t-SNE for visualization
    X_2d = segmentation.apply_tsne(output_dir)

    # Gaussian Mixture Models
    gmm_model, gmm_labels, n_gmm_clusters = segmentation.gaussian_mixture_models(
        n_components_range=range(2, 7), output_dir=output_dir
    )

    # DBSCAN
    dbscan_model, dbscan_labels, n_dbscan_clusters = segmentation.dbscan_clustering(
        eps=0.5, min_samples=50, output_dir=output_dir
    )

    # HDBSCAN
    hdbscan_model, hdbscan_labels, n_hdbscan_clusters = segmentation.hdbscan_clustering(
        min_cluster_size=50, output_dir=output_dir
    )

    # Interpret GMM clusters (most stable)
    gmm_interpretations = segmentation.interpret_clusters(
        gmm_labels, feature_names, cluster_type='GMM'
    )

    print_section_header("CLUSTERING PIPELINE COMPLETED")
    print(f"✓ GMM: {n_gmm_clusters} clusters identified")
    print(f"✓ DBSCAN: {n_dbscan_clusters} clusters identified")
    if HDBSCAN_AVAILABLE:
        print(f"✓ HDBSCAN: {n_hdbscan_clusters} clusters identified")
    print(f"✓ Cluster interpretations generated")

    return segmentation, gmm_model, gmm_labels, gmm_interpretations

