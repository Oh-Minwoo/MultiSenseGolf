from __future__ import annotations

import torch


def mpjpe_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
    # pred/target: (B, T, J*3)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape [B, T, out_dim].")
    diff = (pred - target).view(pred.shape[0], pred.shape[1], -1, 3)  # (B, T, J, 3)
    err = torch.linalg.norm(diff, dim=-1)  # (B, T, J)
    if lengths is None:
        return err.mean()

    b, t, j = err.shape
    lengths = lengths.to(device=pred.device)
    time_idx = torch.arange(t, device=pred.device).view(1, -1)  # (1, T)
    mask = (time_idx < lengths.view(-1, 1)).to(dtype=err.dtype)  # (B, T)
    denom = (mask.sum() * j).clamp_min(1.0)
    return (err * mask[:, :, None]).sum() / denom

def mpjve_loss(pred: torch.Tensor, target: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    pred, target: [B, T, out_dim], out_dim = 3*J
    dt: time step (seconds). Use 1/fps if frames are uniformly sampled.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape [B, T, out_dim].")
    if pred.ndim != 3:
        raise ValueError("Expected pred/target shape [B, T, out_dim].")
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    if pred.shape[2] % 3 != 0:
        raise ValueError("out_dim must be divisible by 3.")

    B, T, out_dim = pred.shape
    J = out_dim // 3

    p = pred.view(B, T, J, 3)
    y = target.view(B, T, J, 3)

    v_p = (p[:, 1:] - p[:, :-1]) / dt      # [B, T-1, J, 3]
    v_y = (y[:, 1:] - y[:, :-1]) / dt

    err = torch.linalg.norm(v_p - v_y, dim=-1)  # [B, T-1, J]
    return err.mean()


def fourth_derivative_smooth_loss(
    pred: torch.Tensor,
    dt: float = 1.0,
    squared: bool = False,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    4th derivative based smoothness loss.
    EgoBody3M style: L_smooth = L2(p''''_t) regularization on prediction only.

    pred: [B, T, out_dim] where out_dim = 3*J
    dt: time step in seconds (e.g., 1/30)
    squared: if True, uses mean(||snap||^2); else mean(||snap||)

    returns: scalar tensor
    """
    if pred.ndim != 3:
        raise ValueError("pred must have shape [B, T, out_dim].")
    B, T, out_dim = pred.shape
    if out_dim % 3 != 0:
        raise ValueError("out_dim must be divisible by 3.")
    if T < 5:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    J = out_dim // 3
    p = pred.view(B, T, J, 3)

    # Finite differences to approximate derivatives
    v = (p[:, 1:] - p[:, :-1]) / dt          # [B, T-1, J, 3]
    a = (v[:, 1:] - v[:, :-1]) / dt          # [B, T-2, J, 3]
    j = (a[:, 1:] - a[:, :-1]) / dt          # [B, T-3, J, 3]
    snap = (j[:, 1:] - j[:, :-1]) / dt       # [B, T-4, J, 3]  (4th derivative)

    snap_mag = torch.linalg.norm(snap, dim=-1)  # [B, T-4, J]

    if lengths is None:
        return (snap_mag ** 2).mean() if squared else snap_mag.mean()

    lengths = lengths.to(device=pred.device)
    valid_steps = (lengths - 4).clamp_min(0)  # snap has T-4 steps
    time_idx = torch.arange(T - 4, device=pred.device).view(1, -1)  # (1, T-4)
    mask = (time_idx < valid_steps.view(-1, 1)).to(dtype=snap_mag.dtype)  # (B, T-4)
    denom = (mask.sum() * J).clamp_min(1.0)
    val = snap_mag ** 2 if squared else snap_mag
    return (val * mask[:, :, None]).sum() / denom
