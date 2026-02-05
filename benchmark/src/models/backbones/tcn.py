from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res = None
        if in_ch != out_ch:
            self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.relu(y)
        y = self.dropout(y)
        res = x if self.res is None else self.res(x)
        return y + res


class TCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        ch_in = in_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(_TCNBlock(ch_in, hidden_dim, dilation=dilation, dropout=dropout))
            ch_in = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)
