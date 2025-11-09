"""
Loss Functions for Spatial Awareness Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for image/spatial reconstruction from CSI.
    Combines MSE and perceptual loss.
    """
    
    def __init__(self, mse_weight: float = 1.0, perceptual_weight: float = 0.1):
        super(ReconstructionLoss, self).__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss."""
        # MSE loss
        mse = self.mse_loss(pred, target)
        
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Combined loss
        loss = self.mse_weight * mse + (1 - self.mse_weight) * l1
        
        return loss


class SpatialLoss(nn.Module):
    """
    Spatial loss that considers spatial relationships in the output.
    """
    
    def __init__(self, spatial_weight: float = 0.5):
        super(SpatialLoss, self).__init__()
        self.spatial_weight = spatial_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate spatial loss."""
        # Pixel-wise loss
        pixel_loss = self.mse_loss(pred, target)
        
        # Spatial gradient loss (encourages smooth spatial transitions)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        spatial_loss = self.mse_loss(pred_grad_x, target_grad_x) + \
                      self.mse_loss(pred_grad_y, target_grad_y)
        
        # Combined loss
        loss = pixel_loss + self.spatial_weight * spatial_loss
        
        return loss


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity Index) loss.
    """
    
    def __init__(self, window_size: int = 11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
                             for x in range(window_size)])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
              window_size: int, channel: int, size_average: bool = True) -> torch.Tensor:
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Calculate SSIM loss."""
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)
        
        channel = img1.size(1)
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        ssim_value = self._ssim(img1, img2, window, self.window_size, channel)
        return 1 - ssim_value  # Return loss (1 - SSIM)


class CombinedLoss(nn.Module):
    """
    Combined loss function using multiple loss components.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.1,
        spatial_weight: float = 0.1
    ):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.spatial_weight = spatial_weight
        
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.spatial_loss = SpatialLoss(spatial_weight=1.0)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss."""
        mse = self.mse_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        spatial = self.spatial_loss(pred, target)
        
        loss = self.mse_weight * mse + \
               self.ssim_weight * ssim + \
               self.spatial_weight * spatial
        
        return loss

