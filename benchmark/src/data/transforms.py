from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray], eps: float = 1e-8) -> "StandardScaler":
        data = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.maximum(std, eps)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class DictTransform:
    def __init__(
        self,
        imu: Optional[StandardScaler] = None,
        pressure: Optional[StandardScaler] = None,
        fpv: Optional[StandardScaler] = None,
        gt: Optional[StandardScaler] = None,
    ) -> None:
        self.imu = imu
        self.pressure = pressure
        self.fpv = fpv
        self.gt = gt

    def __call__(self, item: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.imu is not None and "imu" in item:
            item["imu"] = self.imu.transform(item["imu"])
        if self.pressure is not None and "pressure" in item:
            item["pressure"] = self.pressure.transform(item["pressure"])
        if self.fpv is not None and "fpv" in item:
            item["fpv"] = self.fpv.transform(item["fpv"])
        if self.gt is not None and "gt" in item:
            item["gt"] = self.gt.transform(item["gt"])
        return item
