import numpy as np
from dataclasses import dataclass
from typing import Iterator, Tuple


@dataclass
class CSISensorConfig:
    window_size: int = 256
    stride: int = 64
    fs_hz: float = 100.0


class CSISensor:
    """Iterates CSI windows from numpy arrays for detection."""

    def __init__(self, config: CSISensorConfig | None = None):
        self.cfg = config or CSISensorConfig()

    def windows(self, csi: np.ndarray) -> Iterator[Tuple[np.ndarray, int]]:
        """Yield (window, index) where index is the window number."""
        # Expect csi as (time, subcarriers) or (subcarriers, time)
        x = csi
        if x.ndim != 2:
            raise ValueError("CSI must be 2D array")
        if x.shape[0] in (30, 60) and x.shape[1] >= self.cfg.window_size:
            x = x.T

        T = self.cfg.window_size
        S = self.cfg.stride
        if x.shape[0] < T:
            pad = T - x.shape[0]
            x = np.pad(x, ((0, pad), (0, 0)), mode="edge")

        n = (x.shape[0] - T) // S + 1
        for i in range(n):
            start = i * S
            yield x[start : start + T, :], i

