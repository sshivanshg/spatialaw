# K-Means Clustering for WiFi Data

This guide explains how to perform K-means clustering on your WiFi data to identify patterns, group similar signals, and analyze spatial signatures.

## Overview

K-means clustering groups your WiFi data into clusters based on signal characteristics. This is useful for:
- **Identifying spatial locations** based on WiFi signatures
- **Grouping similar signal patterns** (strong/weak signals, different channels)
- **Preprocessing data** for machine learning models
- **Finding optimal locations** for WiFi access points

## Quick Start

### Basic Usage

```bash
# Cluster WiFi data with 5 clusters
python scripts/run_kmeans.py --data_path data/wifi_data_20251109_150557.json --n_clusters 5

# Find optimal number of clusters automatically
python scripts/run_kmeans.py --data_path data/wifi_data_20251109_150557.json --find_optimal

# Cluster with visualization
python scripts/run_kmeans.py --data_path data/wifi_data_20251109_150557.json --n_clusters 3 --visualize
```

### Cluster Multiple Files

```bash
# Cluster all JSON files in a directory
python scripts/run_kmeans.py --data_path data/large_dataset --n_clusters 5 --visualize
```

## Feature Types

### 1. Signal Features (Default)
Clusters based on WiFi signal characteristics:
- RSSI (signal strength)
- Signal strength (0-100)
- SNR (signal-to-noise ratio)
- Channel
- Number of antennas
- Signal variance
- Noise level

```bash
python scripts/run_kmeans.py --data_path data/wifi_data.json --feature_type signal
```

### 2. CSI Features
Clusters based on Channel State Information:
- Amplitude mean/std
- Phase mean/std
- Power
- SNR from CSI

```bash
python scripts/run_kmeans.py --data_path data/wifi_data.json --feature_type csi
```

### 3. Combined Features
Clusters using both signal and CSI features:

```bash
python scripts/run_kmeans.py --data_path data/wifi_data.json --feature_type combined
```

## Finding Optimal Number of Clusters

Use the `--find_optimal` flag to automatically determine the best number of clusters:

```bash
python scripts/run_kmeans.py \
    --data_path data/wifi_data.json \
    --find_optimal \
    --k_range 2 10 \
    --visualize
```

This will:
1. Test different k values (2 to 10)
2. Calculate silhouette scores
3. Find optimal k
4. Generate elbow curve visualization

## Python API

### Basic Usage

```python
from src.analysis.kmeans_clustering import WiFiKMeansClustering

# Initialize clustering
clustering = WiFiKMeansClustering(n_clusters=5)

# Load data
wifi_data = clustering.load_wifi_data('data/wifi_data.json')

# Extract features
features = clustering.extract_signal_features(wifi_data)

# Fit clustering
clustering.fit(features)

# Evaluate
metrics = clustering.evaluate(features)
print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")

# Visualize
clustering.visualize_clusters(features, save_path='clusters.png')

# Save clustered data
clustering.save_clustered_data(wifi_data, clustering.labels_, 'clustered_data.json')
```

### Find Optimal Clusters

```python
# Find optimal number of clusters
optimal_results = clustering.find_optimal_clusters(features, k_range=(2, 10))
print(f"Optimal k: {optimal_results['optimal_k']}")

# Use optimal k
clustering.n_clusters = optimal_results['optimal_k']
clustering.kmeans = KMeans(n_clusters=clustering.n_clusters, random_state=42)
clustering.fit(features)
```

### Different Feature Types

```python
# Signal features
signal_features = clustering.extract_signal_features(wifi_data)

# CSI features
from src.data_collection.csi_processor import CSIProcessor
csi_processor = CSIProcessor()
csi_features = clustering.extract_csi_features(wifi_data, csi_processor)

# Combined features
combined_features = clustering.extract_combined_features(wifi_data, csi_processor)
```

## Output Files

### Clustered Data JSON
The script saves WiFi data with cluster labels added:

```json
[
  {
    "rssi": -59,
    "signal_strength": 58,
    "snr": 33,
    "channel": 44,
    "cluster": 0
  },
  {
    "rssi": -56,
    "signal_strength": 62,
    "snr": 36,
    "channel": 44,
    "cluster": 0
  }
]
```

### Visualization
Cluster visualization saved to `visualizations/kmeans_clusters.png`:
- 2D projection of clusters (using PCA if >2 features)
- Cluster centers marked with red X
- Color-coded clusters

### Optimal Clusters Plot
If using `--find_optimal`, generates `visualizations/optimal_clusters.png`:
- Elbow curve (inertia vs k)
- Silhouette score vs k
- Optimal k marked

## Evaluation Metrics

### Silhouette Score
- Range: -1 to 1
- Higher is better
- Measures how similar samples are to their cluster vs other clusters
- Good: > 0.5, Excellent: > 0.7

### Calinski-Harabasz Score
- Higher is better
- Ratio of between-cluster to within-cluster variance

### Davies-Bouldin Score
- Lower is better
- Average similarity ratio of clusters

### Inertia
- Lower is better
- Sum of squared distances to cluster centers

## Use Cases

### 1. Spatial Location Identification
Cluster WiFi signals to identify different locations:

```bash
# Collect data from different locations
python scripts/collect_wifi_data.py --duration 60 --output location1.json
python scripts/collect_wifi_data.py --duration 60 --output location2.json

# Cluster to identify locations
python scripts/run_kmeans.py --data_path data/ --n_clusters 5 --visualize
```

### 2. Signal Quality Analysis
Group signals by quality (strong/weak):

```bash
python scripts/run_kmeans.py \
    --data_path data/wifi_data.json \
    --n_clusters 3 \
    --feature_type signal \
    --visualize
```

### 3. Preprocessing for ML
Use cluster labels as features for machine learning:

```python
# Add cluster labels to training data
clustering = WiFiKMeansClustering(n_clusters=5)
features = clustering.extract_signal_features(wifi_data)
clustering.fit(features)
labeled_data = clustering.add_cluster_labels(wifi_data, clustering.labels_)

# Use cluster labels as feature in model training
```

### 4. Access Point Optimization
Find optimal locations for WiFi access points:

```bash
# Collect data from multiple locations
# Cluster to find similar signal patterns
# Identify areas with weak signals (separate clusters)
python scripts/run_kmeans.py --data_path data/ --find_optimal --visualize
```

## Tips

1. **Sample Size**: Need at least 10-20 samples per cluster for meaningful results
2. **Feature Selection**: Start with `signal` features, try `combined` for better results
3. **Normalization**: Keep normalization enabled (default) for best results
4. **Optimal k**: Use `--find_optimal` to find best k, but consider your use case
5. **Visualization**: Always use `--visualize` to understand cluster structure

## Troubleshooting

### Too Many/Few Clusters
- Use `--find_optimal` to find optimal k
- Adjust `--k_range` to test different ranges
- Consider your use case (e.g., 3-5 clusters for location identification)

### Poor Clustering Quality
- Try different feature types (`signal`, `csi`, `combined`)
- Ensure you have enough samples (at least 10 per cluster)
- Check if data needs preprocessing (outliers, missing values)

### Visualization Issues
- Ensure `visualizations/` directory exists
- Check that matplotlib is installed
- For high-dimensional data, PCA is automatically used

## Examples

### Example 1: Basic Clustering
```bash
python scripts/run_kmeans.py \
    --data_path data/wifi_data_20251109_150557.json \
    --n_clusters 5 \
    --feature_type signal \
    --visualize \
    --output_path data/clustered_data.json
```

### Example 2: Find Optimal Clusters
```bash
python scripts/run_kmeans.py \
    --data_path data/large_dataset \
    --find_optimal \
    --k_range 2 15 \
    --feature_type combined \
    --visualize
```

### Example 3: Cluster Multiple Files
```bash
python scripts/run_kmeans.py \
    --data_path data/ \
    --n_clusters 3 \
    --feature_type signal \
    --visualize \
    --output_path data/all_clustered.json
```

## Next Steps

1. **Analyze Clusters**: Examine cluster characteristics (mean RSSI, channels, etc.)
2. **Visualize**: Use visualization to understand cluster structure
3. **Use in ML**: Add cluster labels as features for model training
4. **Spatial Mapping**: Map clusters to physical locations
5. **Optimize**: Use clusters to optimize WiFi access point placement

