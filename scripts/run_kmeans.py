#!/usr/bin/env python3
"""
K-Means Clustering Script for WiFi Data
Clusters WiFi signals based on signal features, CSI patterns, or combined features
"""

import sys
import os
import argparse
import glob
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.kmeans_clustering import WiFiKMeansClustering


def main():
    parser = argparse.ArgumentParser(description='K-Means Clustering for WiFi Data')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to WiFi data JSON file or directory')
    parser.add_argument('--n_clusters', type=int, default=5, 
                       help='Number of clusters (default: 5)')
    parser.add_argument('--feature_type', type=str, default='signal',
                       choices=['signal', 'csi', 'combined'],
                       help='Type of features to use (default: signal)')
    parser.add_argument('--find_optimal', action='store_true',
                       help='Find optimal number of clusters')
    parser.add_argument('--k_range', type=int, nargs=2, default=[2, 10],
                       help='Range of k values for optimal cluster finding (default: 2 10)')
    parser.add_argument('--output_path', type=str, default='data/clustered_wifi_data.json',
                       help='Path to save clustered data')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate cluster visualization')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Disable feature normalization')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("K-Means Clustering for WiFi Data")
    print("=" * 70)
    print()
    
    # Initialize clustering
    clustering = WiFiKMeansClustering(
        n_clusters=args.n_clusters,
        normalize=not args.no_normalize
    )
    
    # Load data
    print("1. Loading data...")
    if os.path.isdir(args.data_path):
        # Load all JSON files in directory
        json_files = glob.glob(os.path.join(args.data_path, '*.json'))
        if not json_files:
            print(f" No JSON files found in {args.data_path}")
            return
        print(f"   Found {len(json_files)} JSON file(s)")
        wifi_data = clustering.load_wifi_data(json_files)
    else:
        wifi_data = clustering.load_wifi_data(args.data_path)
    
    if len(wifi_data) == 0:
        print(" No data loaded!")
        return
    
    print(f"   Loaded {len(wifi_data)} samples")
    print()
    
    # Extract features
    print(f"2. Extracting {args.feature_type} features...")
    if args.feature_type == 'signal':
        features = clustering.extract_signal_features(wifi_data)
    elif args.feature_type == 'csi':
        features = clustering.extract_csi_features(wifi_data)
    else:  # combined
        features = clustering.extract_combined_features(wifi_data)
    
    print(f"   Feature matrix shape: {features.shape}")
    print(f"   Features: {', '.join(clustering.feature_names)}")
    print()
    
    # Find optimal number of clusters if requested
    if args.find_optimal:
        print("3. Finding optimal number of clusters...")
        optimal_results = clustering.find_optimal_clusters(
            features, 
            k_range=tuple(args.k_range)
        )
        clustering.n_clusters = optimal_results['optimal_k']
        clustering.kmeans = WiFiKMeansClustering(
            n_clusters=clustering.n_clusters,
            normalize=clustering.normalize
        ).kmeans
        print()
    
    # Fit clustering
    print(f"4. Fitting K-Means with {clustering.n_clusters} clusters...")
    clustering.fit(features, feature_type=args.feature_type)
    print()
    
    # Evaluate
    print("5. Evaluating clustering...")
    metrics = clustering.evaluate(features)
    print("   Metrics:")
    print(f"     Inertia: {metrics['inertia']:.2f}")
    print(f"     Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"     Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    print(f"     Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
    print("   Cluster sizes:")
    for cluster_id, size in sorted(metrics['cluster_sizes'].items()):
        print(f"     Cluster {cluster_id}: {size} samples ({size/len(wifi_data)*100:.1f}%)")
    print()
    
    # Visualize
    if args.visualize:
        print("6. Generating visualization...")
        os.makedirs('visualizations', exist_ok=True)
        clustering.visualize_clusters(
            features, 
            save_path='visualizations/kmeans_clusters.png'
        )
        print()
    
    # Save clustered data
    print("7. Saving clustered data...")
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    clustering.save_clustered_data(wifi_data, clustering.labels_, args.output_path)
    print()
    
    print("=" * 70)
    print("Clustering completed!")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"  1. View clustered data: {args.output_path}")
    if args.visualize:
        print("  2. View visualization: visualizations/kmeans_clusters.png")
    print("  3. Analyze cluster characteristics")
    print()


if __name__ == "__main__":
    main()

