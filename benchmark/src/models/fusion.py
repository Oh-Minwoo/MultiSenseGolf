from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Use LazyLinear so modality input dims can change without requiring
        # config updates (e.g., IMU feature redesigns).
        self.proj = nn.LazyLinear(out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.gate = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x * self.gate


class EarlyFusion(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        latent_dim: int,
        modality_order: Optional[List[str]] = None,
        dropout: float = 0.0,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.modality_order = modality_order or list(modality_dims.keys())
        self.modality_dropout = modality_dropout
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(modality_dims[name], latent_dim, dropout)
                for name in self.modality_order
            }
        )
        self.out_dim = latent_dim * len(self.modality_order)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        for name in self.modality_order:
            if name not in batch:
                continue
            x = self.projectors[name](batch[name])
            if self.training and self.modality_dropout > 0.0:
                keep_prob = 1.0 - self.modality_dropout
                mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
                x = x * mask / keep_prob
            feats.append(x)
        if len(feats) == 0:
            raise ValueError("no_modalities_in_batch")
        return torch.cat(feats, dim=-1)



class FusionModel(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        latent_dim: int,
        backbone: nn.Module,
        head: nn.Module,
        modality_order: Optional[List[str]] = None,
        dropout: float = 0.0,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion = EarlyFusion(
                modality_dims=modality_dims,
                latent_dim=latent_dim,
                modality_order=modality_order,
                dropout=dropout,
                modality_dropout=modality_dropout,
            )
        self.backbone = backbone
        self.head = head

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.fusion(batch)
        if "lengths" in batch:
            x = self.backbone(x, lengths=batch["lengths"])
        else:
            x = self.backbone(x)
        return self.head(x)
