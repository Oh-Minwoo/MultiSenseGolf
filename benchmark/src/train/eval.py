from __future__ import annotations

from typing import Dict

import torch

from src.train.metrics import (
    jitter_error,
    jitter_error_masked,
    mpjpe,
    mpjpe_masked,
    mpjve,
)


@torch.no_grad()
def run_eval(
    model,
    loader,
    device: str,
    gt_scaler=None,
    target_fps=None
) -> Dict[str, float]:
    model.eval()
    if target_fps is None:
        raise ValueError("target_fps_required_for_eval")
    total_mpjpe = 0.0
    total_mpjve = 0.0
    total_jitter = 0.0
    n = 0
    for batch in loader:
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        pred = model(batch)
        target = batch["gt"]
        lengths = batch.get("lengths")
        if gt_scaler is not None:
            pred = torch.from_numpy(gt_scaler.inverse_transform(pred.cpu().numpy())).to(device)
            target = torch.from_numpy(gt_scaler.inverse_transform(target.cpu().numpy())).to(device)
        if lengths is None:
            total_mpjpe += mpjpe(pred, target).item()
            total_mpjve += mpjve(pred, target, dt=1 / target_fps).item()
            total_jitter += jitter_error(pred, dt=1 / target_fps).item()
        else:
            total_mpjpe += mpjpe_masked(pred, target, lengths).item()
            total_mpjve += mpjve(pred, target, dt=1 / target_fps, lengths=lengths).item()
            total_jitter += jitter_error_masked(pred, lengths, dt=1 / target_fps).item()
        n += 1
    if n == 0:
        return {"mpjpe": 0.0, "mpjve": 0.0, "jitter": 0.0}
    return {
        "mpjpe": total_mpjpe / n,
        "mpjve": total_mpjve / n,
        "jitter": total_jitter / n,
    }
