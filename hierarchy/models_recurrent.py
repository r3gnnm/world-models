import torch
import torch.nn as nn


class RecurrentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2,
                 hidden: int = 256):
        super().__init__()
        self.hidden_size = hidden
        self.cell = nn.GRUCell(latent_dim + action_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor):
        h = self.cell(torch.cat([z, a], -1), h)
        z_next = z + self.head(torch.cat([h, z], -1))  
        return z_next, h


class MLPPredictorSeq(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2,
                 hidden: int = 256):
        super().__init__()
        self.hidden_size = hidden
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor):
        return z + self.net(torch.cat([z, a], -1)), h
