#!/usr/bin/env python3
"""
Train a binary human presence detector on the offline WiFi CSI HAR dataset.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.csi_loader import build_feature_dataset  # noqa: E402


def train_sklearn_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a scikit-learn classifier on extracted CSI features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "logistic":
        model = LogisticRegression(random_state=random_state, max_iter=2000, class_weight="balanced")
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Training {model_type} model...")
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else 0.0,
    }

    return model, scaler, metrics, X_test, y_test, y_test_pred, y_test_proba


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dims: Optional[List[int]] = None,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train the `MotionDetector` MLP on the extracted features.
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    class_counts = np.bincount(y_train, minlength=2)
    class_counts[class_counts == 0] = 1
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MotionDetector(input_features=X.shape[1], hidden_dims=hidden_dims, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for xb, yb in val_loader:
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.numpy())
                all_labels.append(yb.numpy())

        y_val_pred = np.concatenate(all_preds)
        y_val_true = np.concatenate(all_labels)
        val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).float())
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = np.argmax(logits.numpy(), axis=1)

    metrics = {
        "val_accuracy": accuracy_score(y_val, preds),
        "val_precision": precision_score(y_val, preds, zero_division=0),
        "val_recall": recall_score(y_val, preds, zero_division=0),
        "val_f1": f1_score(y_val, preds, zero_division=0),
        "val_roc_auc": roc_auc_score(y_val, probs),
    }

    return model, scaler, metrics, X_val, y_val, preds, probs


# Plotting utilities removed for a minimalist setup.


def parse_list_argument(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    return [v.strip() for v in values if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train presence detector on WiFi CSI HAR dataset")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("WiFi CSI HAR Dataset"),
        help="Root directory containing room/session subfolders.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic", "svm"],
        help="Model family to train.",
    )
    parser.add_argument("--window_size", type=int, default=400, help="Sliding window size in packets.")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap ratio (0=no overlap, 0.5=50%% overlap).",
    )
    parser.add_argument(
        "--presence_threshold",
        type=float,
        default=0.2,
        help="Fraction of packets marked as presence required to label a window as positive.",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="statistics",
        choices=["statistics", "flatten", "flatten_magnitude"],
        help="Feature representation used for classical models.",
    )
    parser.add_argument(
        "--rooms",
        type=str,
        nargs="*",
        default=None,
        help="Subset of rooms to include (e.g. room_1 room_2).",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="*",
        default=None,
        help="Subset of session numbers to include.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"), help="Where to store checkpoints.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Hold-out fraction for evaluation.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    # NN training options removed

    args = parser.parse_args()

    print("=" * 72)
    print("WiFi CSI Presence Detector Training")
    print("=" * 72)
    print(f"Dataset root      : {args.dataset_root}")
    print(f"Model type        : {args.model_type}")
    print(f"Window size       : {args.window_size}")
    print(f"Overlap           : {args.overlap}")
    print(f"Presence threshold: {args.presence_threshold}")
    print(f"Feature mode      : {args.feature_mode}")
    print(f"Rooms filter      : {args.rooms}")
    print(f"Sessions filter   : {args.sessions}")
    print()

    # Build feature dataset
    print("Loading dataset and extracting features...")
    features, labels, metadata = build_feature_dataset(
        dataset_root=args.dataset_root,
        window_size=args.window_size,
        overlap=args.overlap,
        rooms=parse_list_argument(args.rooms),
        sessions=args.sessions,
        presence_threshold=args.presence_threshold,
        feature_mode=args.feature_mode,
        normalise=True,
    )

    print(f"\nDataset summary:")
    print(f"  Windows: {len(labels)}")
    print(f"  Feature dimension: {features.shape[1]}")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    print(f"  Presence windows: {positives} ({positives / len(labels) * 100:.1f}%)")
    print(f"  Empty windows   : {negatives} ({negatives / len(labels) * 100:.1f}%)")

    model, scaler, metrics, X_eval, y_eval, y_eval_pred, y_eval_proba = train_sklearn_model(
        features,
        labels,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = args.output_dir / f"presence_detector_{args.model_type}.pkl"
    scaler_filename = args.output_dir / f"presence_detector_{args.model_type}_scaler.pkl"
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"\n✅ Saved model to {model_filename}")
    print(f"✅ Saved scaler to {scaler_filename}")

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key:15s}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_eval, y_eval_pred, target_names=["no_presence", "presence"]))

    metrics_path = args.output_dir / f"presence_detector_{args.model_type}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"\n✅ Saved metrics to {metrics_path}")
    print("\nTraining completed.")


if __name__ == "__main__":
    main()

