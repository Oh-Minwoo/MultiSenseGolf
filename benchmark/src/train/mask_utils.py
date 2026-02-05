from __future__ import annotations

import torch


def make_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    lengths = lengths.to(device=lengths.device, dtype=torch.long)
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
