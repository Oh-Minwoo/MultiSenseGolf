from __future__ import annotations


def get_device(device: str | None = None) -> str:
    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
