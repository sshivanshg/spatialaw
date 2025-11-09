"""
K-Means Clustering for WiFi Spatial Awareness Data
Clusters WiFi signals, CSI features, or spatial patterns
"""

import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from pathlib import Path


class WiFiKMeansClustering:
    """
    K-Means clustering for WiFi data.
    Can cluster based on signal features, CSI patterns, or spatial signatures.
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42, normalize: bool = True):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            normalize: Whether to normalize features before clustering
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.normalize = normalize
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler() if normalize else None
        self.feature_names = None
        self.cluster_centers_ = None
        self.labels_ = None
        
    def load_wifi_data(self, filepath: Union[str, List[str]]) -> List[Dict]:
        """
        Load WiFi data from JSON file(s).
        
        Args:
            filepath: Path to JSON file or list of paths
            
        Returns:
            List of WiFi data dictionaries
        """
        if isinstance(filepath, str):
            filepath = [filepath]
        
        all_data = []
        for fp in filepath:
            if not os.path.exists(fp):
                print(f"⚠️  File not found: {fp}")
                continue
                
            with open(fp, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        
        print(f"✅ Loaded {len(all_data)} samples from {len(filepath)} file(s)")
        return all_data
    
    def extract_signal_features(self, wifi_data: List[Dict]) -> np.ndarray:
        """
        Extract signal-based features from WiFi data.
        
        Features:
        - RSSI
        - Signal strength
        - SNR
        - Channel
        - Number of antennas
        - Signal variance (if multiple signals)
        
        Args:
            wifi_data: List of WiFi data dictionaries
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []
        feature_names = []
        
        for sample in wifi_data:
            sample_features = []
            
            # RSSI
            rssi = sample.get('rssi', -100)
            sample_features.append(rssi)
            
            # Signal strength (0-100)
            signal_strength = sample.get('signal_strength', 0)
            sample_features.append(signal_strength)
            
            # SNR
            snr = sample.get('snr', 0)
            sample_features.append(snr)
            
            # Channel (normalized)
            channel = sample.get('channel', 1)
            sample_features.append(channel)
            
            # Number of antennas
            num_antennas = sample.get('num_antennas', 1)
            sample_features.append(num_antennas)
            
            # Signal variance (if multiple signals available)
            signals = sample.get('signals', [rssi])
            if len(signals) > 1:
                signal_variance = np.var(signals)
            else:
                signal_variance = 0
            sample_features.append(signal_variance)
            
            # Noise level
            noise = sample.get('noise', -91)
            sample_features.append(noise)
            
            features.append(sample_features)
        
        # Set feature names
        self.feature_names = [
            'rssi', 'signal_strength', 'snr', 'channel', 
            'num_antennas', 'signal_variance', 'noise'
        ]
        
        return np.array(features)
    
    def extract_csi_features(self, wifi_data: List[Dict], csi_processor=None) -> np.ndarray:
        """
        Extract CSI-based features from WiFi data.
        Uses CSI processor to generate amplitude/phase features.
        
        Args:
            wifi_data: List of WiFi data dictionaries
            csi_processor: CSIProcessor instance (optional)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if csi_processor is None:
            from src.data_collection.csi_processor import CSIProcessor
            csi_processor = CSIProcessor()
        
        features = []
        feature_names = []
        
        for sample in wifi_data:
            sample_features = []
            
            # Generate mock CSI if not available
            # In real scenario, you'd have actual CSI data
            num_antennas = sample.get('num_antennas', 3)
            num_subcarriers = 64
            
            # Use signal strength to generate realistic CSI
            signal_strength = sample.get('signal_strength', 50) / 100.0
            rssi = sample.get('rssi', -60)
            
            # Generate mock CSI matrix
            amplitude = np.random.normal(signal_strength, 0.1, (num_antennas, num_subcarriers))
            phase = np.random.uniform(-np.pi, np.pi, (num_antennas, num_subcarriers))
            
            # Extract CSI features
            csi_features = csi_processor.extract_features(
                amplitude * np.exp(1j * phase)
            )
            
            # Add key CSI features
            sample_features.extend([
                csi_features['amplitude_mean'],
                csi_features['amplitude_std'],
                csi_features['phase_mean'],
                csi_features['phase_std'],
                csi_features['power'],
                csi_features['snr'],
            ])
            
            features.append(sample_features)
        
        # Set feature names
        self.feature_names = [
            'amplitude_mean', 'amplitude_std', 'phase_mean', 
            'phase_std', 'power', 'snr'
        ]
        
        return np.array(features)
    
    def extract_combined_features(self, wifi_data: List[Dict], csi_processor=None) -> np.ndarray:
        """
        Extract combined signal + CSI features.
        
        Args:
            wifi_data: List of WiFi data dictionaries
            csi_processor: CSIProcessor instance (optional)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        signal_features = self.extract_signal_features(wifi_data)
        csi_features = self.extract_csi_features(wifi_data, csi_processor)
        
        # Combine features
        combined_features = np.hstack([signal_features, csi_features])
        
        # Update feature names
        signal_names = [
            'rssi', 'signal_strength', 'snr', 'channel', 
            'num_antennas', 'signal_variance', 'noise'
        ]
        csi_names = [
            'amplitude_mean', 'amplitude_std', 'phase_mean', 
            'phase_std', 'power', 'snr_csi'
        ]
        self.feature_names = signal_names + csi_names
        
        return combined_features
    
    def fit(self, features: np.ndarray, feature_type: str = 'signal'):
        """
        Fit K-Means model to features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            feature_type: Type of features ('signal', 'csi', 'combined')
        """
        # Normalize features
        if self.normalize:
            features = self.scaler.fit_transform(features)
        
        # Fit K-Means
        self.kmeans.fit(features)
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        print(f"✅ K-Means fitted with {self.n_clusters} clusters")
        print(f"   Samples: {len(features)}")
        print(f"   Features: {features.shape[1]}")
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return self.kmeans.predict(features)
    
    def evaluate(self, features: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        labels = self.labels_
        
        metrics = {
            'inertia': self.kmeans.inertia_,
            'silhouette_score': silhouette_score(features, labels),
            'calinski_harabasz_score': calinski_harabasz_score(features, labels),
            'davies_bouldin_score': davies_bouldin_score(features, labels),
        }
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique, counts))
        
        return metrics
    
    def visualize_clusters(self, features: np.ndarray, save_path: Optional[str] = None, 
                          feature_indices: Tuple[int, int] = (0, 1)):
        """
        Visualize clusters in 2D (using first 2 features or PCA).
        
        Args:
            features: Feature matrix (n_samples, n_features)
            save_path: Path to save visualization
            feature_indices: Indices of features to plot (default: first 2)
        """
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        labels = self.labels_
        
        # Use PCA if more than 2 features
        if features.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_scaled)
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
        else:
            features_2d = features_scaled[:, feature_indices]
            if self.feature_names:
                xlabel = self.feature_names[feature_indices[0]]
                ylabel = self.feature_names[feature_indices[1]]
            else:
                xlabel = f'Feature {feature_indices[0]}'
                ylabel = f'Feature {feature_indices[1]}'
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0], 
            features_2d[:, 1], 
            c=labels, 
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        
        # Plot cluster centers
        if features.shape[1] > 2:
            centers_2d = pca.transform(self.cluster_centers_)
        else:
            centers_2d = self.cluster_centers_[:, feature_indices]
        
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster Centers'
        )
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'K-Means Clustering (k={self.n_clusters})')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def add_cluster_labels(self, wifi_data: List[Dict], labels: np.ndarray) -> List[Dict]:
        """
        Add cluster labels to WiFi data.
        
        Args:
            wifi_data: List of WiFi data dictionaries
            labels: Cluster labels
            
        Returns:
            WiFi data with cluster labels added
        """
        labeled_data = wifi_data.copy()
        for i, sample in enumerate(labeled_data):
            sample['cluster'] = int(labels[i])
        return labeled_data
    
    def save_clustered_data(self, wifi_data: List[Dict], labels: np.ndarray, 
                           filepath: str):
        """
        Save WiFi data with cluster labels.
        
        Args:
            wifi_data: List of WiFi data dictionaries
            labels: Cluster labels
            filepath: Path to save file
        """
        labeled_data = self.add_cluster_labels(wifi_data, labels)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(labeled_data, f, indent=2)
        
        print(f"✅ Clustered data saved to {filepath}")
    
    def find_optimal_clusters(self, features: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            k_range: Range of k values to test (min, max)
            
        Returns:
            Dictionary with optimal k and evaluation metrics
        """
        if self.scaler is not None:
            features = self.scaler.fit_transform(features)
        
        k_min, k_max = k_range
        k_values = range(k_min, k_max + 1)
        
        inertias = []
        silhouette_scores = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features, labels))
        
        # Find optimal k (elbow method + highest silhouette score)
        # Simple heuristic: find k with highest silhouette score
        optimal_k_idx = np.argmax(silhouette_scores)
        optimal_k = k_values[optimal_k_idx]
        
        results = {
            'k_values': list(k_values),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k,
            'optimal_silhouette_score': silhouette_scores[optimal_k_idx]
        }
        
        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(k_values, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True, alpha=0.3)
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(k_values, silhouette_scores, 'go-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.grid(True, alpha=0.3)
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Optimal number of clusters: {optimal_k}")
        print(f"   Silhouette score: {silhouette_scores[optimal_k_idx]:.3f}")
        
        return results


if __name__ == "__main__":
    # Example usage
    clustering = WiFiKMeansClustering(n_clusters=5)
    
    # Load data
    data = clustering.load_wifi_data('data/wifi_data_20251109_150557.json')
    
    # Extract features
    features = clustering.extract_signal_features(data)
    
    # Find optimal number of clusters
    optimal_results = clustering.find_optimal_clusters(features, k_range=(2, 8))
    
    # Fit with optimal k
    clustering.n_clusters = optimal_results['optimal_k']
    clustering.kmeans = KMeans(n_clusters=clustering.n_clusters, random_state=42, n_init=10)
    clustering.fit(features)
    
    # Evaluate
    metrics = clustering.evaluate(features)
    print("\nClustering Metrics:")
    for key, value in metrics.items():
        if key != 'cluster_sizes':
            print(f"  {key}: {value:.3f}")
    print(f"  Cluster sizes: {metrics['cluster_sizes']}")
    
    # Visualize
    clustering.visualize_clusters(features, save_path='visualizations/clusters.png')
    
    # Save clustered data
    clustering.save_clustered_data(data, clustering.labels_, 'data/clustered_wifi_data.json')

