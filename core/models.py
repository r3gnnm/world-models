"""Энкодер и предиктор для action-conditioned JEPA.

Encoder:   картинка (1, 64, 64) -> латент z размерности latent_dim.
Predictor: (z_t, a_t) -> предсказанный z_{t+1}.

Размеры маленькие намеренно: на этой среде модель обучается за минуты,
что позволяет быстро итерироваться. Когда перейдёшь на сложные среды,
замени энкодер на ResNet/ViT — интерфейс останется тем же.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),# 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """MLP-динамика: конкатенирует латент и действие, предсказывает следующий латент.

    Идея: предсказываем дельту (z' = z + f(z, a)) — так проще выучить
    тождество "ничего не изменилось" и стабильнее multi-step rollout.
    """
    def __init__(self, latent_dim: int = 128, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))
