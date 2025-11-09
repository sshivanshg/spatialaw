"""
Metrics for evaluating spatial awareness model
"""

import numpy as np
from typing import Dict

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. SSIM calculation will be limited.")


def calculate_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((pred - target) ** 2)


def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def calculate_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    # Normalize to [0, 1] if needed
    if pred.max() > 1.0 or pred.min() < -1.0:
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    if target.max() > 1.0 or target.min() < -1.0:
        target = (target - target.min()) / (target.max() - target.min())
    
    try:
        # Calculate PSNR for each image in batch
        psnr_values = []
        for i in range(pred.shape[0]):
            # Convert to [0, 1] range
            pred_img = pred[i]
            target_img = target[i]
            
            if pred_img.min() < 0:
                pred_img = (pred_img + 1) / 2
            if target_img.min() < 0:
                target_img = (target_img + 1) / 2
            
            # Calculate PSNR
            mse = np.mean((pred_img - target_img) ** 2)
            if mse == 0:
                psnr_values.append(100.0)  # Perfect reconstruction
            else:
                max_pixel = 1.0
                psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
                psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    except:
        return 0.0


def calculate_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Structural Similarity Index."""
    if not SKIMAGE_AVAILABLE:
        # Fallback: simple correlation-based similarity
        pred_norm = pred.copy()
        target_norm = target.copy()
        if pred_norm.min() < 0:
            pred_norm = (pred_norm + 1) / 2
        if target_norm.min() < 0:
            target_norm = (target_norm + 1) / 2
        
        # Simple normalized cross-correlation as proxy
        pred_flat = pred_norm.flatten()
        target_flat = target_norm.flatten()
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.0
    
    try:
        # Normalize to [0, 1] if needed
        pred_norm = pred.copy()
        target_norm = target.copy()
        
        if pred_norm.min() < 0:
            pred_norm = (pred_norm + 1) / 2
        if target_norm.min() < 0:
            target_norm = (target_norm + 1) / 2
        
        # Calculate SSIM for each image in batch
        ssim_values = []
        for i in range(pred_norm.shape[0]):
            pred_img = pred_norm[i]
            target_img = target_norm[i]
            
            # Handle multi-channel images
            if len(pred_img.shape) == 3:
                # Multi-channel (e.g., RGB)
                ssim_val = ssim(
                    target_img,
                    pred_img,
                    data_range=1.0,
                    channel_axis=0,
                    win_size=min(7, min(pred_img.shape[1], pred_img.shape[2]))
                )
            else:
                # Grayscale
                ssim_val = ssim(
                    target_img,
                    pred_img,
                    data_range=1.0,
                    win_size=min(7, min(pred_img.shape[0], pred_img.shape[1]))
                )
            
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0


def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict:
    """
    Calculate all evaluation metrics.
    
    Args:
        pred: Predictions array
        target: Target array
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': calculate_mse(pred, target),
        'mae': calculate_mae(pred, target),
        'psnr': calculate_psnr(pred, target),
        'ssim': calculate_ssim(pred, target)
    }
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    pred = np.random.rand(4, 3, 64, 64)  # Batch of RGB images
    target = np.random.rand(4, 3, 64, 64)
    
    metrics = calculate_metrics(pred, target)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

