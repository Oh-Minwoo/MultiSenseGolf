from __future__ import annotations

import torch

from src.train.mask_utils import make_length_mask


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.norm(diff.view(diff.shape[0], diff.shape[1], -1, 3), dim=-1))


def mpjpe_masked(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape_mismatch pred={pred.shape} target={target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"expected (B,T,3J), got pred.ndim={pred.ndim}")
    if pred.shape[-1] % 3 != 0:
        raise ValueError(f"last_dim_must_be_multiple_of_3, got d={pred.shape[-1]}")

    diff = (pred - target).view(pred.shape[0], pred.shape[1], -1, 3)  # (B, T, J, 3)
    err = torch.norm(diff, dim=-1)  # (B, T, J)
    j = err.size(-1)
    max_len = err.size(1)
    mask = make_length_mask(lengths.to(err.device), max_len).to(dtype=err.dtype)  # (B, T)
    denom = (mask.sum() * j).clamp_min(1.0)
    return (err * mask[:, :, None]).sum() / denom


def mpjve(
    pred: torch.Tensor,
    target: torch.Tensor,
    dt: float = 1.0,
    lengths: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape_mismatch pred={pred.shape} target={target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"expected (B,T,3J), got pred.ndim={pred.ndim}")

    b, t, d = pred.shape
    if d % 3 != 0:
        raise ValueError(f"last_dim_must_be_multiple_of_3, got d={d}")
    j = d // 3

    pred_xyz = pred.view(b, t, j, 3)
    target_xyz = target.view(b, t, j, 3)

    dt_t = torch.as_tensor(dt, device=pred.device, dtype=pred.dtype).clamp_min(eps)

    pred_vel = (pred_xyz[:, 1:] - pred_xyz[:, :-1]) / dt_t
    target_vel = (target_xyz[:, 1:] - target_xyz[:, :-1]) / dt_t

    vel_err = torch.norm(pred_vel - target_vel, dim=-1)  # (B, T-1, J)

    if lengths is None:
        return vel_err.mean()

    if lengths.ndim != 1 or lengths.shape[0] != b:
        raise ValueError(f"lengths must be (B,), got {lengths.shape}")

    lengths = lengths.to(device=pred.device)
    valid_steps = (lengths - 1).clamp_min(0)  # velocity has T-1 steps
    time_idx = torch.arange(t - 1, device=pred.device).view(1, -1)  # (1, T-1)
    mask = (time_idx < valid_steps.view(-1, 1)).unsqueeze(-1)  # (B, T-1, 1)
    mask = mask.to(dtype=vel_err.dtype)

    denom = (mask.sum() * j).clamp_min(1.0)
    return (vel_err * mask).sum() / denom



def jitter_error(
    pred: torch.Tensor,
    dt: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.ndim != 3:
        raise ValueError(f"expected (B,T,D), got pred.ndim={pred.ndim}")
    if pred.shape[1] < 4:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    if pred.shape[2] % 3 != 0:
        raise ValueError(f"expected D to be multiple of 3 (xyz per joint), got D={pred.shape[2]}")

    dt_t = torch.as_tensor(dt, device=pred.device, dtype=pred.dtype).clamp_min(eps)

    B, T, D = pred.shape
    J = D // 3
    x = pred.view(B, T, J, 3)  # (B, T, J, 3)

    jerk = x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]  # (B, T-3, J, 3)

    # Time scaling: jerk ~ d^3x/dt^3
    jerk = jerk / (dt_t ** 3)

    # Unit conversion: cm -> m
    jerk = jerk * 0.01

    # Scale down by 1e2
    jerk = jerk / 1e2

    jerk_mag = torch.linalg.norm(jerk, dim=-1)  # (B, T-3, J)
    return jerk_mag.mean()


def jitter_error_masked(
    pred: torch.Tensor,
    lengths: torch.Tensor,
    dt: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.ndim != 3:
        raise ValueError(f"expected (B,T,D), got pred.ndim={pred.ndim}")
    if pred.shape[1] < 4:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    if pred.shape[2] % 3 != 0:
        raise ValueError(f"expected D to be multiple of 3 (xyz per joint), got D={pred.shape[2]}")

    dt_t = torch.as_tensor(dt, device=pred.device, dtype=pred.dtype).clamp_min(eps)

    B, T, D = pred.shape
    J = D // 3
    x = pred.view(B, T, J, 3)

    jerk = x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]  # (B, T-3, J, 3)
    jerk = jerk / (dt_t ** 3)
    jerk = jerk * 0.01
    jerk = jerk / 1e2

    jerk_mag = torch.linalg.norm(jerk, dim=-1)  # (B, T-3, J)
    max_len = jerk_mag.size(1)
    mask = make_length_mask((lengths - 3).clamp_min(0).to(jerk_mag.device), max_len).to(dtype=jerk_mag.dtype)  # (B, T-3)
    denom = (mask.sum() * J).clamp_min(1.0)
    return (jerk_mag * mask[:, :, None]).sum() / denom