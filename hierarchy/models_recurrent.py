"""Рекуррентный предиктор: латентная динамика с памятью.

Зачем: при частичной наблюдаемости (эгоцентрический вид) одного кадра
недостаточно — по чёрному пятну в центре комнаты нельзя понять, где ты.
Безпамятный предиктор z_t -> z_{t+1} упирается в этот предел.

Решение: скрытое состояние h_t, которое накапливает историю наблюдений
и действий. Предсказание опирается на (z_t, a_t, h_t), а не только на кадр.
Это упрощённый аналог RSSM из DreamerV3: там латентное состояние тоже
разделено на детерминированную (рекуррентную) и наблюдаемую части.

Ключевой момент для probe: у рекуррентной модели "состояние мира" живёт
не в z, а в паре (z, h). Позицию агента надо восстанавливать из обоих.
"""
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
        """Один шаг. Возвращает (предсказанный z_{t+1}, новое h)."""
        h = self.cell(torch.cat([z, a], -1), h)
        z_next = z + self.head(torch.cat([h, z], -1))   # предсказываем дельту
        return z_next, h


class MLPPredictorSeq(nn.Module):
    """Безпамятный предиктор с тем же интерфейсом — для честного сравнения.

    Принимает и возвращает h, но игнорирует его. Так обе модели обучаются
    одним и тем же кодом на одних и тех же последовательностях, и разница
    в метриках объясняется ТОЛЬКО наличием памяти.
    """
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
