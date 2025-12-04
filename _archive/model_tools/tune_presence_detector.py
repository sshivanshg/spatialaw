#!/usr/bin/env python3
"""
Hyperparameter tuning workflow for the WiAR binary presence detector.

Runs group-aware cross-validation over a small ensemble of tree models,
summarizes metrics, and saves the best-performing estimator plus scaler.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError as exc:
    raise ImportError("Install joblib (pip install joblib)") from exc

BINARY_DIR = Path("data/processed/binary")
MODELS_DIR = Path("models")
DEFAULT_CV_FOLDS = 5


@dataclass
class Candidate:
    """Container describing a model search candidate."""

    name: str
    params: Dict[str, object]

    def build(self) -> Pipeline:
        model = RandomForestClassifier(
            **self.params,
            random_state=self.params.get("random_state", 42),
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )


def default_candidates() -> List[Candidate]:
    """Return built-in hyperparameter configurations."""
    base = {
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    return [
        Candidate(
            name="rf_light",
            params={**base, "n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        ),
        Candidate(
            name="rf_medium",
            params={**base, "n_estimators": 200, "max_depth": 32, "min_samples_split": 4},
        ),
        Candidate(
            name="rf_heavy",
            params={**base, "n_estimators": 400, "max_depth": None, "min_samples_split": 2},
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=BINARY_DIR,
        help="Directory containing features.npy, labels.csv, feature_names.json.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for tuned model artifacts.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Number of GroupKFold splits.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel jobs for cross-validation (passed to cross_validate).",
    )
    return parser.parse_args()


def load_binary_dataset(binary_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    features_path = binary_dir / "features.npy"
    labels_path = binary_dir / "labels.csv"
    feature_names_path = binary_dir / "feature_names.json"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Binary dataset not found under {binary_dir}. Run scripts/process_binary_dataset.py first."
        )

    X = np.load(features_path)
    labels_df = pd.read_csv(labels_path)
    with feature_names_path.open() as f:
        feature_names = json.load(f)

    if len(X) != len(labels_df):
        raise ValueError(
            "Mismatch between features and labels lengths: "
            f"{len(X)} vs {len(labels_df)}"
        )

    y = labels_df["label"].to_numpy()
    if "source" in labels_df.columns:
        groups = labels_df["source"].fillna(labels_df.index.to_series().astype(str)).to_numpy()
    else:
        groups = labels_df.index.to_numpy(dtype=str)
    return X, y, feature_names, groups


def evaluate_candidate(
    candidate: Candidate,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv,
    n_jobs: int,
) -> Dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    pipeline = candidate.build()
    cv_result = cross_validate(
        pipeline,
        X,
        y,
        scoring=scoring,
        cv=cv,
        groups=groups,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    metrics = {metric: float(np.mean(scores)) for metric, scores in cv_result.items()}
    metrics_std = {f"{metric}_std": float(np.std(scores)) for metric, scores in cv_result.items()}
    return {**metrics, **metrics_std}


def main() -> int:
    args = parse_args()
    X, y, feature_names, groups = load_binary_dataset(args.binary_dir)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    cv = GroupKFold(n_splits=args.cv_folds)

    candidates = default_candidates()
    leaderboard: List[Dict[str, object]] = []
    best_entry: Dict[str, object] | None = None

    for candidate in candidates:
        metrics = evaluate_candidate(candidate, X, y, groups, cv, args.jobs)
        entry = {"name": candidate.name, **candidate.params, **metrics}
        leaderboard.append(entry)
        if best_entry is None or entry["test_f1"] > best_entry["test_f1"]:
            best_entry = entry
        print(
            f"[{candidate.name}] F1={entry['test_f1']:.4f} ± {entry['test_f1_std']:.4f}, "
            f"ROC-AUC={entry['test_roc_auc']:.4f}"
        )

    if best_entry is None:
        raise RuntimeError("No candidates were evaluated.")

    print(f"\nBest candidate: {best_entry['name']} (F1={best_entry['test_f1']:.4f})")

    # Refit best candidate on the full dataset
    best_candidate = next(c for c in candidates if c.name == best_entry["name"])
    best_pipeline = best_candidate.build()
    best_pipeline.fit(X, y)

    model = best_pipeline.named_steps["model"]
    scaler = best_pipeline.named_steps["scaler"]

    model_path = args.models_dir / "presence_detector_rf.joblib"
    scaler_path = args.models_dir / "presence_detector_scaler.joblib"
    pipeline_path = args.models_dir / "presence_detector_pipeline.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(best_pipeline, pipeline_path)
    print(f"✓ Saved tuned model to {model_path}")
    print(f"✓ Saved scaler to {scaler_path}")
    print(f"✓ Saved pipeline to {pipeline_path}")

    metrics_path = args.models_dir / "presence_detector_metrics.json"
    metrics_payload = {
        "best_candidate": best_entry["name"],
        "cv_folds": args.cv_folds,
        "cv_accuracy_mean": best_entry["test_accuracy"],
        "cv_accuracy_std": best_entry["test_accuracy_std"],
        "cv_precision_mean": best_entry["test_precision"],
        "cv_precision_std": best_entry["test_precision_std"],
        "cv_recall_mean": best_entry["test_recall"],
        "cv_recall_std": best_entry["test_recall_std"],
        "cv_f1_mean": best_entry["test_f1"],
        "cv_f1_std": best_entry["test_f1_std"],
        "cv_roc_auc_mean": best_entry["test_roc_auc"],
        "cv_roc_auc_std": best_entry["test_roc_auc_std"],
        "feature_names": feature_names,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"✓ Metrics saved to {metrics_path}")

    # Persist leaderboard
    leaderboard_path = args.models_dir / "tuning_results.json"
    with leaderboard_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "leaderboard": leaderboard,
                "best_candidate": best_entry,
                "cv_folds": args.cv_folds,
            },
            f,
            indent=2,
        )
    print(f"✓ Leaderboard saved to {leaderboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

