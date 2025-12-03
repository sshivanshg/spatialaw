import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _create_dummy_binary_dir(tmp_path: Path, n_samples: int = 40) -> Path:
    binary_dir = tmp_path / "binary"
    binary_dir.mkdir()

    rng = np.random.default_rng(42)
    features = rng.normal(size=(n_samples, 12)).astype(np.float32)
    np.save(binary_dir / "features.npy", features)

    labels = pd.DataFrame(
        {
            "label": rng.integers(0, 2, size=n_samples),
            "motion_score": rng.uniform(0.5, 2.0, size=n_samples),
            "source": [f"sample_{i%5}" for i in range(n_samples)],
            "original_label": ["dummy_activity"] * n_samples,
            "dataset": ["test"] * n_samples,
        }
    )
    labels.to_csv(binary_dir / "labels.csv", index=False)

    feature_names = [f"feat_{i}" for i in range(12)]
    (binary_dir / "feature_names.json").write_text(json.dumps(feature_names))

    summary = {
        "total_samples": n_samples,
        "n_features": 12,
        "motion_threshold": float(np.quantile(labels["motion_score"], 0.25)),
    }
    (binary_dir / "binary_dataset_summary.json").write_text(json.dumps(summary))

    return binary_dir


def _run_script(args, env=None):
    subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        check=True,
        env=env,
    )


def test_validate_binary_dataset_cli(tmp_path):
    binary_dir = _create_dummy_binary_dir(tmp_path)
    _run_script(["scripts/validate_binary_dataset.py", "--binary-dir", str(binary_dir)])

    report_path = binary_dir / "validation_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["total_samples"] == 40
    assert not report["split_validation"]["leakage_detected"]


def test_tune_presence_detector_cli(tmp_path):
    binary_dir = _create_dummy_binary_dir(tmp_path, n_samples=50)
    models_dir = tmp_path / "models"

    _run_script(
        [
            "model_tools/tune_presence_detector.py",
            "--binary-dir",
            str(binary_dir),
            "--models-dir",
            str(models_dir),
            "--cv-folds",
            "2",
        ]
    )

    assert (models_dir / "presence_detector_rf.joblib").exists()
    assert (models_dir / "presence_detector_scaler.joblib").exists()
    metrics = json.loads((models_dir / "presence_detector_metrics.json").read_text())
    assert metrics["best_candidate"]
    leaderboard = json.loads((models_dir / "tuning_results.json").read_text())
    assert leaderboard["leaderboard"]

