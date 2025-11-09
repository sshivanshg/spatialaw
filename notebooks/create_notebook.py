#!/usr/bin/env python3
"""Create the baseline_analysis.ipynb notebook programmatically"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase-1 Baseline Analysis: WiFi Signal Spatial Mapping\n",
                "\n",
                "This notebook demonstrates the baseline analysis pipeline for WiFi signal spatial mapping:\n",
                "- Data loading and preprocessing\n",
                "- Feature extraction\n",
                "- Visualizations (heatmaps, scatter plots, PCA)\n",
                "- Simple model training and evaluation"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Setup and Imports"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import json\n",
                "import os\n",
                "import sys\n",
                "\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                "\n",
                "# Set style\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette(\"husl\")\n",
                "%matplotlib inline\n",
                "\n",
                "# Add parent directory to path\n",
                "sys.path.insert(0, os.path.join('..'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Load Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load data (synthetic or real)\n",
                "data_path = '../data/synthetic_wifi_data.json'\n",
                "\n",
                "if os.path.exists(data_path):\n",
                "    with open(data_path, 'r') as f:\n",
                "        data = json.load(f)\n",
                "    df = pd.DataFrame(data)\n",
                "    print(f\"✅ Loaded {len(df)} samples from {data_path}\")\n",
                "else:\n",
                "    print(f\"⚠️  Data file not found: {data_path}\")\n",
                "    print(\"Please generate synthetic data first:\")\n",
                "    print(\"  python scripts/generate_synthetic_wifi_data.py\")\n",
                "    df = None"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display basic information\n",
                "if df is not None:\n",
                "    print(\"\\nData Shape:\", df.shape)\n",
                "    print(\"\\nColumns:\", df.columns.tolist())\n",
                "    print(\"\\nFirst few rows:\")\n",
                "    display(df.head())\n",
                "    print(\"\\nData Statistics:\")\n",
                "    display(df.describe())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Data Preprocessing"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Check for missing values\n",
                "    print(\"Missing values:\")\n",
                "    print(df.isnull().sum())\n",
                "    \n",
                "    # Remove any missing values\n",
                "    df_clean = df.dropna()\n",
                "    print(f\"\\nAfter cleaning: {len(df_clean)} samples\")\n",
                "    \n",
                "    # Filter outliers (optional)\n",
                "    # Remove samples with unrealistic RSSI values\n",
                "    df_clean = df_clean[(df_clean['rssi'] >= -100) & (df_clean['rssi'] <= -30)]\n",
                "    print(f\"After filtering outliers: {len(df_clean)} samples\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Feature Extraction"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Basic features\n",
                "    features = ['position_x', 'position_y']\n",
                "    target = 'rssi'\n",
                "    \n",
                "    # Extract features and target\n",
                "    X = df_clean[features].values\n",
                "    y = df_clean[target].values\n",
                "    \n",
                "    print(f\"Features: {features}\")\n",
                "    print(f\"Target: {target}\")\n",
                "    print(f\"Feature shape: {X.shape}\")\n",
                "    print(f\"Target shape: {y.shape}\")\n",
                "    \n",
                "    # Normalize features\n",
                "    scaler = StandardScaler()\n",
                "    X_scaled = scaler.fit_transform(X)\n",
                "    \n",
                "    print(f\"\\nFeature ranges after scaling:\")\n",
                "    print(f\"  X_scaled min: {X_scaled.min(axis=0)}\")\n",
                "    print(f\"  X_scaled max: {X_scaled.max(axis=0)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Visualizations"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 5.1 Position vs Signal Strength Scatter Plot"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    plt.figure(figsize=(12, 5))\n",
                "    \n",
                "    plt.subplot(1, 2, 1)\n",
                "    scatter = plt.scatter(df_clean['position_x'], df_clean['position_y'], \n",
                "                         c=df_clean['rssi'], cmap='viridis', s=50, alpha=0.6)\n",
                "    plt.colorbar(scatter, label='RSSI (dBm)')\n",
                "    plt.xlabel('Position X (meters)')\n",
                "    plt.ylabel('Position Y (meters)')\n",
                "    plt.title('Signal Strength Heatmap (RSSI)')\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    \n",
                "    plt.subplot(1, 2, 2)\n",
                "    scatter = plt.scatter(df_clean['position_x'], df_clean['position_y'], \n",
                "                         c=df_clean['signal_strength'], cmap='plasma', s=50, alpha=0.6)\n",
                "    plt.colorbar(scatter, label='Signal Strength (0-100)')\n",
                "    plt.xlabel('Position X (meters)')\n",
                "    plt.ylabel('Position Y (meters)')\n",
                "    plt.title('Signal Strength Heatmap (0-100 scale)')\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.savefig('../visualizations/baseline_position_signal_scatter.png', dpi=150, bbox_inches='tight')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 5.2 Signal Distribution Histograms"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
                "    \n",
                "    axes[0, 0].hist(df_clean['rssi'], bins=30, edgecolor='black', alpha=0.7)\n",
                "    axes[0, 0].set_xlabel('RSSI (dBm)')\n",
                "    axes[0, 0].set_ylabel('Frequency')\n",
                "    axes[0, 0].set_title('RSSI Distribution')\n",
                "    axes[0, 0].grid(True, alpha=0.3)\n",
                "    \n",
                "    axes[0, 1].hist(df_clean['snr'], bins=30, edgecolor='black', alpha=0.7, color='orange')\n",
                "    axes[0, 1].set_xlabel('SNR (dB)')\n",
                "    axes[0, 1].set_ylabel('Frequency')\n",
                "    axes[0, 1].set_title('SNR Distribution')\n",
                "    axes[0, 1].grid(True, alpha=0.3)\n",
                "    \n",
                "    axes[1, 0].hist(df_clean['signal_strength'], bins=30, edgecolor='black', alpha=0.7, color='green')\n",
                "    axes[1, 0].set_xlabel('Signal Strength (0-100)')\n",
                "    axes[1, 0].set_ylabel('Frequency')\n",
                "    axes[1, 0].set_title('Signal Strength Distribution')\n",
                "    axes[1, 0].grid(True, alpha=0.3)\n",
                "    \n",
                "    axes[1, 1].hist(df_clean['channel'], bins=20, edgecolor='black', alpha=0.7, color='red')\n",
                "    axes[1, 1].set_xlabel('Channel')\n",
                "    axes[1, 1].set_ylabel('Frequency')\n",
                "    axes[1, 1].set_title('Channel Distribution')\n",
                "    axes[1, 1].grid(True, alpha=0.3)\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.savefig('../visualizations/baseline_signal_distributions.png', dpi=150, bbox_inches='tight')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 5.3 Correlation Matrix"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Select numeric columns for correlation\n",
                "    numeric_cols = ['position_x', 'position_y', 'rssi', 'snr', 'signal_strength', 'channel', 'noise']\n",
                "    corr_matrix = df_clean[numeric_cols].corr()\n",
                "    \n",
                "    plt.figure(figsize=(10, 8))\n",
                "    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)\n",
                "    plt.title('Correlation Matrix')\n",
                "    plt.tight_layout()\n",
                "    plt.savefig('../visualizations/baseline_correlation_matrix.png', dpi=150, bbox_inches='tight')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 5.4 PCA Visualization"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Prepare features for PCA\n",
                "    pca_features = ['rssi', 'snr', 'signal_strength', 'channel', 'noise']\n",
                "    X_pca = df_clean[pca_features].values\n",
                "    \n",
                "    # Normalize\n",
                "    scaler_pca = StandardScaler()\n",
                "    X_pca_scaled = scaler_pca.fit_transform(X_pca)\n",
                "    \n",
                "    # Apply PCA\n",
                "    pca = PCA(n_components=2)\n",
                "    X_pca_2d = pca.fit_transform(X_pca_scaled)\n",
                "    \n",
                "    # Plot PCA\n",
                "    plt.figure(figsize=(10, 8))\n",
                "    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], \n",
                "                         c=df_clean['rssi'], cmap='viridis', s=50, alpha=0.6)\n",
                "    plt.colorbar(scatter, label='RSSI (dBm)')\n",
                "    plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')\n",
                "    plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')\n",
                "    plt.title('PCA Visualization of WiFi Signals')\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    plt.tight_layout()\n",
                "    plt.savefig('../visualizations/baseline_pca.png', dpi=150, bbox_inches='tight')\n",
                "    plt.show()\n",
                "    \n",
                "    print(f\"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}\")\n",
                "    print(f\"Total Explained Variance: {pca.explained_variance_ratio_.sum():.2%}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Model Training"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Split data\n",
                "    X_train, X_test, y_train, y_test = train_test_split(\n",
                "        X_scaled, y, test_size=0.2, random_state=42\n",
                "    )\n",
                "    \n",
                "    print(f\"Train samples: {len(X_train)}\")\n",
                "    print(f\"Test samples: {len(X_test)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 6.1 Random Forest Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Train Random Forest\n",
                "    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)\n",
                "    rf_model.fit(X_train, y_train)\n",
                "    \n",
                "    # Predictions\n",
                "    y_train_pred_rf = rf_model.predict(X_train)\n",
                "    y_test_pred_rf = rf_model.predict(X_test)\n",
                "    \n",
                "    # Metrics\n",
                "    train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))\n",
                "    test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))\n",
                "    train_r2_rf = r2_score(y_train, y_train_pred_rf)\n",
                "    test_r2_rf = r2_score(y_test, y_test_pred_rf)\n",
                "    \n",
                "    print(\"Random Forest Model:\")\n",
                "    print(f\"  Train RMSE: {train_rmse_rf:.4f}\")\n",
                "    print(f\"  Test RMSE: {test_rmse_rf:.4f}\")\n",
                "    print(f\"  Train R²: {train_r2_rf:.4f}\")\n",
                "    print(f\"  Test R²: {test_r2_rf:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 6.2 Linear Regression Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Train Linear Regression\n",
                "    lr_model = LinearRegression()\n",
                "    lr_model.fit(X_train, y_train)\n",
                "    \n",
                "    # Predictions\n",
                "    y_train_pred_lr = lr_model.predict(X_train)\n",
                "    y_test_pred_lr = lr_model.predict(X_test)\n",
                "    \n",
                "    # Metrics\n",
                "    train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))\n",
                "    test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))\n",
                "    train_r2_lr = r2_score(y_train, y_train_pred_lr)\n",
                "    test_r2_lr = r2_score(y_test, y_test_pred_lr)\n",
                "    \n",
                "    print(\"Linear Regression Model:\")\n",
                "    print(f\"  Train RMSE: {train_rmse_lr:.4f}\")\n",
                "    print(f\"  Test RMSE: {test_rmse_lr:.4f}\")\n",
                "    print(f\"  Train R²: {train_r2_lr:.4f}\")\n",
                "    print(f\"  Test R²: {test_r2_lr:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Model Evaluation and Visualization"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if df is not None:\n",
                "    # Plot predictions vs actual\n",
                "    fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
                "    \n",
                "    # Random Forest\n",
                "    axes[0].scatter(y_test, y_test_pred_rf, alpha=0.5)\n",
                "    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
                "    axes[0].set_xlabel('Actual RSSI (dBm)')\n",
                "    axes[0].set_ylabel('Predicted RSSI (dBm)')\n",
                "    axes[0].set_title(f'Random Forest (R² = {test_r2_rf:.3f})')\n",
                "    axes[0].grid(True, alpha=0.3)\n",
                "    \n",
                "    # Linear Regression\n",
                "    axes[1].scatter(y_test, y_test_pred_lr, alpha=0.5, color='orange')\n",
                "    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
                "    axes[1].set_xlabel('Actual RSSI (dBm)')\n",
                "    axes[1].set_ylabel('Predicted RSSI (dBm)')\n",
                "    axes[1].set_title(f'Linear Regression (R² = {test_r2_lr:.3f})')\n",
                "    axes[1].grid(True, alpha=0.3)\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.savefig('../visualizations/baseline_model_predictions.png', dpi=150, bbox_inches='tight')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Summary"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=\" * 70)\n",
                "print(\"Phase-1 Baseline Analysis Summary\")\n",
                "print(\"=\" * 70)\n",
                "if df is not None:\n",
                "    print(f\"\\nTotal samples: {len(df_clean)}\")\n",
                "    print(f\"Features: {features}\")\n",
                "    print(f\"Target: {target}\")\n",
                "    print(f\"\\nRandom Forest - Test RMSE: {test_rmse_rf:.4f}, R²: {test_r2_rf:.4f}\")\n",
                "    print(f\"Linear Regression - Test RMSE: {test_rmse_lr:.4f}, R²: {test_r2_lr:.4f}\")\n",
                "print(\"\\n✅ Analysis completed!\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save notebook
with open('../notebooks/baseline_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Notebook created successfully!")

