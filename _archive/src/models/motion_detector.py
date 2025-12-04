"""
Runtime helpers for loading and running the trained presence detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import joblib
import numpy as np

from src.preprocess.features import extract_fusion_features, features_to_vector


@dataclass
class DetectorArtifacts:
    """Paths to serialized model assets."""

    model_path: Path = Path("models/presence_detector_rf.joblib")
    scaler_path: Path = Path("models/presence_detector_scaler.joblib")


class SimpleMotionDetector:
    """
    Thin wrapper around the trained Random Forest presence detector.

    Expects pre-computed feature vectors (same order as `feature_names.json`).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        artifacts: DetectorArtifacts | None = None,
    ) -> None:
        if artifacts is None:
            artifacts = DetectorArtifacts(
                model_path=Path(model_path) if model_path else DetectorArtifacts.model_path,
                scaler_path=Path(scaler_path) if scaler_path else DetectorArtifacts.scaler_path,
            )
        self.model = joblib.load(artifacts.model_path)
        self.scaler = joblib.load(artifacts.scaler_path)

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Return binary predictions for a batch of feature vectors."""
        X = self._prepare(features)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Return probability of activity for each feature vector."""
        X = self._prepare(features)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    @staticmethod
    def _prepare(features: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr


class MotionDetector(SimpleMotionDetector):
    """
    End-to-end utility that converts CSI windows into feature vectors
    before running inference.
    """

    def __init__(
        self,
        feature_names: List[str],
        feature_extractor: Callable[[np.ndarray], dict] = extract_fusion_features,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_names = feature_names
        self.feature_extractor = feature_extractor

    def predict_from_windows(self, windows: Iterable[np.ndarray]) -> np.ndarray:
        """Run predictions directly on CSI windows."""
        features = self._features_from_windows(windows)
        return self.predict(features)

    def predict_proba_from_windows(self, windows: Iterable[np.ndarray]) -> np.ndarray:
        """Return probabilities for CSI windows."""
        features = self._features_from_windows(windows)
        return self.predict_proba(features)

    def _features_from_windows(self, windows: Iterable[np.ndarray]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for window in windows:
            feature_dict = self.feature_extractor(window)
            vectors.append(features_to_vector(feature_dict, feature_order=self.feature_names))
        return np.stack(vectors, axis=0)

