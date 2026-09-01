"""Иерархическая JEPA: два уровня с разными масштабами времени.

  Уровень 1 (быстрый): z1 = Encoder(o), предсказание на КАЖДЫЙ шаг,
                       обусловленное контекстом сверху: (z1, a, z2) -> z1'
  Уровень 2 (медленный): z2 = Abstractor(z1), предсказание ПРЫЖКОМ через k
                       шагов: (z2, сводка действий) -> z2_{t+k}

Ключевая идея (H-JEPA, LeCun 2022): верхний уровень должен выучить абстракцию,
меняющуюся медленно, и подсказывать нижнему контекст. Главный риск — верхний
уровень схлопывается в декорацию: нижний справляется сам, а z2 не несёт
информации. Поэтому в train_hier.py встроены диагностики, проверяющие это явно.
"""
import torch
import torch.nn as nn


class Abstractor(nn.Module):
    """z1_t, z2_{t-1} -> z2_t: РЕКУРРЕНТНОЕ накопление медленной абстракции.

    ВАЖНО (урок из эксперимента): первая версия была мгновенной проекцией
    z2 = MLP(z1) без памяти. На эгоцентрической среде это провалилось —
    номер комнаты принципиально не восстановим из одного кадра, его можно
    узнать только интегрируя историю переходов через двери. Абстрактор
    обязан быть рекуррентным, иначе он не может знать больше, чем z1.
    """
    def __init__(self, latent1=128, latent2=32, hidden=64):
        super().__init__()
        self.latent2 = latent2
        self.cell = nn.GRUCell(latent1, latent2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.latent2, device=device)

    def forward(self, z1, z2_prev):
        return self.cell(z1, z2_prev)


class Level1Predictor(nn.Module):
    """Локальная динамика, ОБУСЛОВЛЕННАЯ контекстом верхнего уровня."""
    def __init__(self, latent1=128, latent2=32, action_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent1 + action_dim + latent2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent1),
        )

    def forward(self, z1, a, z2):
        return z1 + self.net(torch.cat([z1, a, z2], -1))


class Level2Predictor(nn.Module):
    """Прыжковое предсказание: где окажется абстракция через k шагов.

    На вход — сводка действий за интервал (сумма и норма), а не отдельные
    действия: верхний уровень работает с агрегированным намерением,
    а не с микродвижениями.
    """
    def __init__(self, latent2=32, action_summary=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent2 + action_summary, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent2),
        )

    def forward(self, z2, a_summary):
        return z2 + self.net(torch.cat([z2, a_summary], -1))


class FlatPredictor(nn.Module):
    """Плоский бейзлайн БЕЗ иерархии — контроль на равное число параметров.

    Принимает z2 и игнорирует его, поэтому обучается тем же кодом.
    Разница в метриках относительно иерархической модели объясняется
    ТОЛЬКО наличием уровня 2.
    """
    def __init__(self, latent1=128, latent2=32, action_dim=2, hidden=288):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent1 + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent1),
        )

    def forward(self, z1, a, z2=None):
        return z1 + self.net(torch.cat([z1, a], -1))


def action_summary(actions):
    """Сводка последовательности действий: (сумма_x, сумма_y, суммарный путь).

    actions: (B, k, 2) -> (B, 3)
    """
    s = actions.sum(1)
    dist = actions.norm(dim=-1).sum(1, keepdim=True)
    return torch.cat([s, dist], -1)
