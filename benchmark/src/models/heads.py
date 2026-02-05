from __future__ import annotations

from torch import nn


class JointRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.fc(x)
