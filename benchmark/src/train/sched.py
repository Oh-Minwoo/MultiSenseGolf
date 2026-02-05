from __future__ import annotations

from typing import Dict


def build_scheduler(optimizer, cfg: Dict):
    name = cfg.get("name")
    if name is None:
        return None
    name = name.lower()
    if name == "steplr":
        step_size = cfg.get("step_size", 10)
        gamma = cfg.get("gamma", 0.1)
        return __import__("torch").optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        t_max = cfg.get("t_max", 30)
        return __import__("torch").optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    raise ValueError(f"unknown_scheduler: {name}")
