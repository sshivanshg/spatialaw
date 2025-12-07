"""
Feature extraction for CSI and RSS signals.

Implements statistical features mentioned in the WiAR paper:
- CSI: variance, envelope, entropy, velocity, MAD, motion period, normalized std
- RSS: peak count, peak positions, local variance
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal
from scipy.stats import entropy


def extract_csi_features(window: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from a CSI window.

    Parameters
    ----------
    window:
        CSI window array of shape ``(n_subcarriers, T)``.

    Returns
    -------
    dict
        Dictionary of feature names to values.
    """
    S, T = window.shape
    features: Dict[str, float] = {}

    # Variance (per subcarrier, then aggregate)
    var_per_sub = np.var(window, axis=1)
    features["csi_variance_mean"] = float(np.mean(var_per_sub))
    features["csi_variance_std"] = float(np.std(var_per_sub))
    features["csi_variance_max"] = float(np.max(var_per_sub))

    # Envelope (Hilbert transform envelope)
    envelope = np.abs(signal.hilbert(window, axis=1))
    features["csi_envelope_mean"] = float(np.mean(envelope))
    features["csi_envelope_std"] = float(np.std(envelope))

    # Signal entropy (Shannon entropy of amplitude distribution)
    # Flatten and bin for histogram
    hist, _ = np.histogram(window.flatten(), bins=50)
    hist = hist + 1e-10  # avoid log(0)
    hist = hist / hist.sum()
    features["csi_entropy"] = float(entropy(hist, base=2))

    # Velocity of signal change (first derivative magnitude)
    diff = np.diff(window, axis=1)
    velocity = np.abs(diff)
    features["csi_velocity_mean"] = float(np.mean(velocity))
    features["csi_velocity_max"] = float(np.max(velocity))

    # Median Absolute Deviation (MAD)
    mad_per_sub = np.median(np.abs(window - np.median(window, axis=1, keepdims=True)), axis=1)
    features["csi_mad_mean"] = float(np.mean(mad_per_sub))
    features["csi_mad_std"] = float(np.std(mad_per_sub))

    # Motion period (dominant frequency via FFT)
    fft_vals = np.abs(np.fft.rfft(window, axis=1))
    # Find peak frequency for each subcarrier
    peak_freqs = np.argmax(fft_vals[:, 1:], axis=1) + 1  # skip DC
    if len(peak_freqs) > 0:
        features["csi_motion_period_mean"] = float(np.mean(peak_freqs))
        features["csi_motion_period_std"] = float(np.std(peak_freqs))
    else:
        features["csi_motion_period_mean"] = 0.0
        features["csi_motion_period_std"] = 0.0

    # Normalized standard deviation
    mean_per_sub = np.mean(window, axis=1, keepdims=True)
    std_per_sub = np.std(window, axis=1, keepdims=True) + 1e-10
    normalized_std = std_per_sub / (np.abs(mean_per_sub) + 1e-10)
    features["csi_norm_std_mean"] = float(np.mean(normalized_std))
    features["csi_norm_std_std"] = float(np.std(normalized_std))

    return features


def extract_rss_features(rss_series: np.ndarray, window_size: int = 10) -> Dict[str, float]:
    """
    Extract RSS-based features for activity recognition.

    Parameters
    ----------
    rss_series:
        1D array of RSS values over time.
    window_size:
        Window size for local variance computation.

    Returns
    -------
    dict
        Dictionary of RSS feature names to values.
    """
    if len(rss_series) == 0:
        return {
            "rss_peak_count": 0.0,
            "rss_peak_mean_amplitude": 0.0,
            "rss_local_variance_mean": 0.0,
            "rss_local_variance_max": 0.0,
            "rss_range": 0.0,
        }

    features: Dict[str, float] = {}

    # Peak detection
    peaks, properties = signal.find_peaks(rss_series, prominence=0.5, distance=5)
    features["rss_peak_count"] = float(len(peaks))
    if len(peaks) > 0:
        peak_amplitudes = rss_series[peaks]
        features["rss_peak_mean_amplitude"] = float(np.mean(peak_amplitudes))
        features["rss_peak_max_amplitude"] = float(np.max(peak_amplitudes))
        features["rss_peak_positions_mean"] = float(np.mean(peaks))
    else:
        features["rss_peak_mean_amplitude"] = 0.0
        features["rss_peak_max_amplitude"] = 0.0
        features["rss_peak_positions_mean"] = 0.0

    # Local variance (sliding window)
    if len(rss_series) >= window_size:
        local_vars = []
        for i in range(len(rss_series) - window_size + 1):
            window = rss_series[i : i + window_size]
            local_vars.append(np.var(window))
        features["rss_local_variance_mean"] = float(np.mean(local_vars))
        features["rss_local_variance_max"] = float(np.max(local_vars))
        features["rss_local_variance_std"] = float(np.std(local_vars))
    else:
        var_all = np.var(rss_series)
        features["rss_local_variance_mean"] = float(var_all)
        features["rss_local_variance_max"] = float(var_all)
        features["rss_local_variance_std"] = 0.0

    # Range
    features["rss_range"] = float(np.max(rss_series) - np.min(rss_series))

    return features


def extract_fusion_features(
    csi_window: np.ndarray, rss_series: np.ndarray | None = None
) -> Dict[str, float]:
    """
    Extract combined CSI + RSS features for activity recognition.

    Parameters
    ----------
    csi_window:
        CSI window array of shape ``(n_subcarriers, T)``.
    rss_series:
        Optional 1D RSS array of length T. If None, only CSI features are returned.

    Returns
    -------
    dict
        Combined feature dictionary.
    """
    features = extract_csi_features(csi_window)

    if rss_series is not None:
        rss_features = extract_rss_features(rss_series)
        features.update(rss_features)

    return features


def features_to_vector(features: Dict[str, float], feature_order: list[str] | None = None) -> np.ndarray:
    """
    Convert feature dictionary to a fixed-order vector.

    Parameters
    ----------
    features:
        Feature dictionary.
    feature_order:
        Optional list specifying feature order. If None, uses sorted keys.

    Returns
    -------
    np.ndarray
        1D feature vector.
    """
    if feature_order is None:
        feature_order = sorted(features.keys())
    return np.array([features.get(key, 0.0) for key in feature_order], dtype=np.float32)

