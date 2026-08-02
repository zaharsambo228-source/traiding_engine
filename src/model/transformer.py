import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class TransformerTradingModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.2,
        num_symbols: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        self.symbol_embedding = nn.Embedding(num_symbols, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x, symbol_ids=None):
        # x shape: (batch_size, seq_len, input_dim)
        # symbol_ids shape: (batch_size,)
        x = self.input_projection(x) * math.sqrt(self.d_model)

        if symbol_ids is not None:
            # Add symbol embedding to all timesteps
            sym_emb = self.symbol_embedding(symbol_ids).unsqueeze(1)  # shape: (batch_size, 1, d_model)
            x = x + sym_emb

        x = self.pos_encoder(x)

        # Pass through transformer
        out = self.transformer_encoder(x)

        # Take the output of the last sequence element
        out = self.fc(out[:, -1, :])
        return out
