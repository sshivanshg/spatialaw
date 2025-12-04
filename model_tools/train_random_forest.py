# Load dataset
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Find project root
root_candidates = [Path().resolve(), Path().resolve().parent]
BINARY_DIR = None
for root in root_candidates:
    candidate = root / "data" / "processed" / "binary"
    if candidate.exists():
        BINARY_DIR = candidate
        break

if BINARY_DIR is None:
    raise FileNotFoundError("Could not locate data/processed/binary directory")

# Load features and labels
features_path = BINARY_DIR / "features.npy"
labels_path = BINARY_DIR / "labels.csv"
feature_names_path = BINARY_DIR / "feature_names.json"

X = np.load(features_path)
labels_df = pd.read_csv(labels_path)
with open(feature_names_path) as f:
    feature_names = json.load(f)

y = labels_df["label"].values

print(f"Dataset loaded:")
print(f"  Features shape: {X.shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Number of features: {len(feature_names)}")
print(f"  Label distribution:")
print(f"    Class 0 (No Activity): {np.sum(y == 0)} samples ({100*np.sum(y==0)/len(y):.1f}%)")
print(f"    Class 1 (Activity): {np.sum(y == 1)} samples ({100*np.sum(y==1)/len(y):.1f}%)")


# Split data and scale features
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# Use recording / source ID to avoid leakage between train and test
if "source" in labels_df.columns:
    groups = labels_df["source"].values
else:
    # Fallback: each sample is its own group (equivalent to random split)
    groups = np.arange(len(y))

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train_scaled.shape[0]} samples")
print(f"Test set: {X_test_scaled.shape[0]} samples")


# Train Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Create and train model
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)
print("✓ Training complete!")


# Evaluate model
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)
y_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

print("=" * 60)
print("Model Performance")
print("=" * 60)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1-Score:  {test_f1:.4f}")
print(f"Test ROC-AUC:   {test_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Activity', 'Activity']))


# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Activity', 'Activity'],
            yticklabels=['No Activity', 'Activity'])
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('True Label', fontsize=12)
axes[0, 0].set_xlabel('Predicted Label', fontsize=12)

# 2. ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {test_roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance
feature_importance = rf_model.feature_importances_
indices = np.argsort(feature_importance)[::-1][:10]  # Top 10
axes[1, 0].barh(range(len(indices)), feature_importance[indices], color='steelblue')
axes[1, 0].set_yticks(range(len(indices)))
axes[1, 0].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
axes[1, 0].set_xlabel('Importance', fontsize=12)
axes[1, 0].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()

# 4. Prediction Probability Distribution
axes[1, 1].hist(y_test_proba[y_test == 0], bins=30, alpha=0.6, label='No Activity (True)', color='red')
axes[1, 1].hist(y_test_proba[y_test == 1], bins=30, alpha=0.6, label='Activity (True)', color='green')
axes[1, 1].set_xlabel('Predicted Probability (Activity)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Decision Threshold')
axes[1, 1].legend()

plt.tight_layout()
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

plot_path = output_dir / "training_results.png"
plt.savefig(plot_path)
print(f"✓ Training plots saved to: {plot_path}")
# plt.show()


# Save model and scaler
import joblib
from pathlib import Path

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

model_path = output_dir / "presence_detector_rf.joblib"
scaler_path = output_dir / "presence_detector_scaler.joblib"

joblib.dump(rf_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"✓ Model saved to: {model_path}")
print(f"✓ Scaler saved to: {scaler_path}")

# Save metrics
metrics = {
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "test_roc_auc": float(test_roc_auc),
    "n_estimators": 150,
    "feature_names": feature_names
}

metrics_path = output_dir / "presence_detector_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Metrics saved to: {metrics_path}")

