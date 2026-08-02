import torch
import torch.nn as nn


class LSTMTradingModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3, dropout: float = 0.2
    ):
        """
        Output dim = 3 for classification (Short, Flat, Long)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out
