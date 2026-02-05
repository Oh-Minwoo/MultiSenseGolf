from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2000) -> None:
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 2000,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, model_dim) if in_dim != model_dim else nn.Identity()
        self.pos = _SinusoidalPositionalEncoding(model_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            key_padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        return self.encoder(x, src_key_padding_mask=key_padding_mask)
