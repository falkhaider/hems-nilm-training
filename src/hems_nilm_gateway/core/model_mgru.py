import torch
import torch.nn as nn


class MGRUNetMultiSeq2Seq(nn.Module):
    # GRU + Head für Seq2Seq Modell
    def __init__(
        self,
        num_devices: int,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.0,
        in_channels: int = 2,
    ):
        super().__init__()

        # Anzahl der Heads = Anzahl Zielgeräte
        self.num_devices = int(num_devices)

        # Rekurrenter Encoder über das Zeitfenster (B,T,C) -> (B,T,H)
        self.rnn = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,  # Dropout nur bei >1 Layer
        )

        # Projektion: Hidden-State je Zeitschritt -> Logits pro Gerät (B,T,D)
        self.head = nn.Linear(hidden, self.num_devices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward: Sequenz durch GRU, dann Head auf jedem Zeitschritt
        o, _ = self.rnn(x)      # Hidden-Repräsentation (B, T, H)
        logits = self.head(o)   # Logits je Zeitschritt und Gerät (B, T, D)
        return logits
