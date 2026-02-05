from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _pad_array(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[0] == length:
        return arr
    pad_width = [(0, length - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="constant")


def collate_batch(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    lengths = np.array([b["imu"].shape[0] for b in batch], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) > 0 else 0

    out: Dict[str, np.ndarray] = {
        "lengths": lengths,
        "subject": [b["subject"] for b in batch],
        "swing": [b["swing"] for b in batch],
    }

    for key in ["time_s", "imu", "gt", "pressure", "fpv"]:
        if key not in batch[0]:
            continue
        padded = [_pad_array(b[key], max_len) for b in batch]
        out[key] = np.stack(padded, axis=0)

    if torch is not None:
        for key in ["time_s", "imu", "gt", "pressure", "fpv", "lengths"]:
            if key in out and not isinstance(out[key], list):
                out[key] = torch.from_numpy(out[key])
    return out
