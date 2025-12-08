# loading data
from pathlib import Path
import numpy as np
import pandas as pd
import json


root_candidates = [Path().resolve(), Path().resolve().parent]
BINARY_DIR = None
for root in root_candidates:
    candidate = root / "data" / "processed" / "binary"
    if candidate.exists():
        BINARY_DIR = candidate
        break

if BINARY_DIR is None:
    raise FileNotFoundError("Could not locate data/processed/binary directory")

# Loading features and labels
features_path = BINARY_DIR / "features.npy" #x
labels_path = BINARY_DIR / "labels.csv" #y
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


#feature scaling , encoding
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

if "source" in labels_df.columns:
    groups = labels_df["source"].values
else:
    
    groups = np.arange(len(y))

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train_scaled.shape[0]} samples")
print(f"Test set: {X_test_scaled.shape[0]} samples")

# random forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Create and training random forest model
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced' 
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)
print("✓ Training complete!")


# evaluating model
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)
y_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# calculating metrics
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
cls_report_str = classification_report(y_test, y_test_pred, target_names=['No Activity', 'Activity'])
print(cls_report_str)

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

# Save detailed artifacts: classification report, confusion matrix, ROC curve
cls_report_path = output_dir / "presence_detector_classification_report.txt"
with open(cls_report_path, "w") as f:
    f.write(cls_report_str)
print(f"✓ Classification report saved to: {cls_report_path}")

cm_path = output_dir / "presence_detector_confusion_matrix.json"
with open(cm_path, "w") as f:
    json.dump({
        "labels": ["No Activity", "Activity"],
        "matrix": cm.tolist(),
    }, f, indent=2)
print(f"✓ Confusion matrix saved to: {cm_path}")

roc_path = output_dir / "presence_detector_roc_curve.json"
with open(roc_path, "w") as f:
    json.dump({
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(test_roc_auc),
    }, f, indent=2)
print(f"✓ ROC curve data saved to: {roc_path}")

