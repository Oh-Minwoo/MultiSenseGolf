from __future__ import annotations

from typing import Dict

import torch


def build_optimizer(params, cfg: Dict):
    name = cfg.get("name", "adam").lower()
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unknown_optimizer: {name}")
