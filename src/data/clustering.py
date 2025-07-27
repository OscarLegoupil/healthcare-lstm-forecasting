import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientSegmentation:
    """
    Advanced client segmentation using behavioral and demographic clustering.
    Segments clients into distinct groups for specialized LSTM modeling.
    """
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_profiles = {}
        
    def prepare_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for client clustering."""
        logger.info("Preparing clustering features...")
        
        # Aggregate person-level statistics
        person_features = df.groupby('id_personne').agg({
            # Financial behavior
            'remb_mutuelle': ['mean', 'std', 'sum', 'count'],
            'frais_reels': ['mean', 'std', 'sum'],
            'nb_acte': ['mean', 'sum'],
            'cost_ratio': ['mean', 'std'],
            'avg_cost_per_act': 'mean',
            
            # Demographics (take first occurrence)
            'Age24': 'first',
            'genre_encoded': 'first',
            'type_benef_encoded': 'first',
            'type_cont': 'first',
            
            # Temporal patterns
            'month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(),
            'is_year_end': 'mean',
            'quarter': lambda x: len(x.unique())  # Activity diversity
        }).round(2)
        
        # Flatten column names
        person_features.columns = ['_'.join(col).strip() for col in person_features.columns]
        
        # Calculate additional behavioral metrics
        person_features['utilization_consistency'] = (
            person_features['remb_mutuelle_std'] / (person_features['remb_mutuelle_mean'] + 1e-6)
        )
        person_features['activity_intensity'] = (
            person_features['nb_acte_sum'] / person_features['remb_mutuelle_count']
        )
        person_features['seasonal_diversity'] = person_features['quarter_<lambda>']
        
        # Handle infinite values and NaNs
        person_features = person_features.replace([np.inf, -np.inf], np.nan)
        person_features = person_features.fillna(person_features.median())
        
        logger.info(f"Clustering features prepared for {len(person_features):,} persons")
        return person_features
    
    def find_optimal_clusters(self, features: pd.DataFrame, max_k: int = 8) -> Dict:
        """Find optimal number of clusters using multiple metrics."""
        logger.info("Finding optimal number of clusters...")
        
        # Prepare features for clustering
        clustering_cols = [
            'remb_mutuelle_mean', 'remb_mutuelle_std', 'remb_mutuelle_sum',
            'utilization_consistency', 'activity_intensity', 'Age24_first',
            'genre_encoded_first', 'type_benef_encoded_first', 'seasonal_diversity'
        ]
        
        X = features[clustering_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        metrics = {'k': [], 'silhouette': [], 'calinski_harabasz': [], 'inertia': []}
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            metrics['k'].append(k)
            metrics['silhouette'].append(silhouette_score(X_scaled, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
            metrics['inertia'].append(kmeans.inertia_)
        
        # Find optimal k (highest silhouette score)
        optimal_k = metrics['k'][np.argmax(metrics['silhouette'])]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        return metrics, optimal_k, X_scaled, clustering_cols
    
    def fit_clustering(self, features: pd.DataFrame, optimal_k: int = None) -> pd.DataFrame:
        """Fit clustering model and return cluster assignments."""
        logger.info("Fitting clustering model...")
        
        metrics, suggested_k, X_scaled, clustering_cols = self.find_optimal_clusters(features)
        
        # Use provided k or suggested optimal k
        k = optimal_k if optimal_k else suggested_k
        self.n_clusters = k
        
        # Fit final clustering model
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to features
        features_with_clusters = features.copy()
        features_with_clusters['cluster'] = cluster_labels
        
        # Create cluster profiles
        self._create_cluster_profiles(features_with_clusters, clustering_cols)
        
        logger.info(f"Clustering complete with {k} clusters")
        return features_with_clusters
    
    def _create_cluster_profiles(self, features: pd.DataFrame, clustering_cols: List) -> None:
        """Create interpretable cluster profiles."""
        logger.info("Creating cluster profiles...")
        
        self.cluster_profiles = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_data = features[features['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'avg_age': cluster_data['Age24_first'].mean(),
                'avg_reimbursement': cluster_data['remb_mutuelle_mean'].mean(),
                'total_reimbursement': cluster_data['remb_mutuelle_sum'].mean(),
                'utilization_consistency': cluster_data['utilization_consistency'].mean(),
                'activity_intensity': cluster_data['activity_intensity'].mean(),
                'gender_distribution': cluster_data['genre_encoded_first'].value_counts().to_dict(),
                'beneficiary_type': cluster_data['type_benef_encoded_first'].value_counts().to_dict()
            }
            
            # Assign interpretable names
            if profile['avg_age'] > 60 and profile['utilization_consistency'] < 1.0:
                profile['name'] = 'Senior Stable'
                profile['description'] = 'Older clients with predictable, consistent healthcare usage'
            elif profile['avg_age'] < 40 and profile['utilization_consistency'] > 1.5:
                profile['name'] = 'Young Volatile'
                profile['description'] = 'Younger clients with irregular, sporadic healthcare usage'
            else:
                profile['name'] = 'Middle Moderate'
                profile['description'] = 'Middle-aged clients with moderate, variable healthcare usage'
            
            self.cluster_profiles[cluster_id] = profile
        
        # Print cluster summary
        for cluster_id, profile in self.cluster_profiles.items():
            logger.info(f"Cluster {cluster_id} ({profile['name']}): {profile['size']:,} clients, "
                       f"avg age {profile['avg_age']:.1f}, avg reimbursement {profile['avg_reimbursement']:.2f}")
    
    def visualize_clusters(self, features: pd.DataFrame, save_path: str = "results/figures") -> None:
        """Create comprehensive cluster visualizations."""
        logger.info("Creating cluster visualizations...")
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Cluster distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age vs Reimbursement
        sns.scatterplot(data=features, x='Age24_first', y='remb_mutuelle_mean', 
                       hue='cluster', alpha=0.7, ax=axes[0,0])
        axes[0,0].set_title('Client Segments: Age vs Average Reimbursement')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Average Monthly Reimbursement')
        
        # Consistency vs Activity
        sns.scatterplot(data=features, x='utilization_consistency', y='activity_intensity',
                       hue='cluster', alpha=0.7, ax=axes[0,1])
        axes[0,1].set_title('Behavioral Patterns: Consistency vs Activity')
        axes[0,1].set_xlabel('Utilization Consistency (CV)')
        axes[0,1].set_ylabel('Activity Intensity')
        
        # Cluster sizes
        cluster_sizes = features['cluster'].value_counts().sort_index()
        cluster_names = [self.cluster_profiles[i]['name'] for i in cluster_sizes.index]
        axes[1,0].bar(cluster_names, cluster_sizes.values)
        axes[1,0].set_title('Cluster Size Distribution')
        axes[1,0].set_ylabel('Number of Clients')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Reimbursement distribution by cluster
        features.boxplot(column='remb_mutuelle_mean', by='cluster', ax=axes[1,1])
        axes[1,1].set_title('Reimbursement Distribution by Cluster')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Average Monthly Reimbursement')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/cluster_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cluster profiles heatmap
        profile_data = []
        for cluster_id in range(self.n_clusters):
            profile = self.cluster_profiles[cluster_id]
            profile_data.append([
                profile['avg_age'],
                profile['avg_reimbursement'],
                profile['utilization_consistency'],
                profile['activity_intensity'],
                profile['size']
            ])
        
        profile_df = pd.DataFrame(profile_data,
                                columns=['Age', 'Avg Reimbursement', 'Consistency', 'Activity', 'Size'],
                                index=[f"Cluster {i}\n({self.cluster_profiles[i]['name']})" 
                                      for i in range(self.n_clusters)])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(profile_df.T, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
        plt.title('Cluster Profiles Heatmap')
        plt.tight_layout()
        plt.savefig(f"{save_path}/cluster_profiles.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster visualizations saved to {save_path}")
    
    def assign_clusters_to_sequences(self, person_ids: List[str], 
                                   features_with_clusters: pd.DataFrame) -> np.ndarray:
        """Assign cluster labels to sequence data."""
        logger.info("Assigning clusters to sequences...")
        
        cluster_mapping = features_with_clusters['cluster'].to_dict()
        sequence_clusters = np.array([cluster_mapping.get(pid, 0) for pid in person_ids])
        
        logger.info(f"Cluster assignment complete for {len(person_ids):,} sequences")
        return sequence_clusters


def main():
    """Example usage of client segmentation."""
    from preprocessing import HealthcareDataProcessor
    
    # Load processed data
    processor = HealthcareDataProcessor()
    results = processor.process_pipeline()
    
    # Perform clustering
    segmentation = ClientSegmentation(n_clusters=3)
    clustering_features = segmentation.prepare_clustering_features(results['feature_data'])
    features_with_clusters = segmentation.fit_clustering(clustering_features)
    
    # Create visualizations
    segmentation.visualize_clusters(features_with_clusters)
    
    # Assign clusters to sequences
    sequence_clusters = segmentation.assign_clusters_to_sequences(
        results['person_ids'], features_with_clusters
    )
    
    # Save cluster assignments
    np.save("data/processed/sequence_clusters.npy", sequence_clusters)
    features_with_clusters.to_parquet("data/processed/features_with_clusters.parquet")
    
    print("Client segmentation completed successfully!")


if __name__ == "__main__":
    main()