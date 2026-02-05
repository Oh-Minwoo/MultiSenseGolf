from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        lstm_hidden = hidden_dim // 2 if bidirectional else hidden_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is None:
            out, _ = self.lstm(x)
            return out
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out
