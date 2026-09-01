"""Рекуррентный энкодер: восстанавливает скорости (vx, vz, omega), которые
принципиально не видны из одного кадра.

Обоснование (подтверждено экспериментом): после 150 эпох обучения x/z/theta
кодируются всё лучше, а probe по vx/vz/omega остаётся низким и почти не
растёт с числом эпох — потому что дело не в недообученности, а в том, что
мгновенная скорость физически не присутствует в статичном кадре. Прямая
параллель с находкой из иерархической части проекта: то, чего нет в
наблюдении, не появится от более долгого обучения — нужен механизм,
интегрирующий историю.

Архитектура: тот же CNN-энкодер на кадр, что и раньше (Encoder из models.py),
плюс GRU поверх последовательности z1_t, дающий h_t — расширенное состояние,
из которого скорости должны восстанавливаться заметно лучше.

    z1_t = Encoder(o_t)                  # как раньше, кадр -> латент
    h_t  = GRU(z1_t, h_{t-1})            # новое: накопленная история

Diagnostics: сравниваем probe(z1) против probe([z1; h]) на все 6 переменных
состояния — ожидание: x/z/theta почти не меняются (уже видны в кадре),
vx/vz/omega заметно растут (нужна была история).
"""
import torch
import torch.nn as nn


class RecurrentQuadEncoder(nn.Module):
    """Оборачивает Encoder дополнительным GRU поверх последовательности
    латентов кадров. Сам Encoder не меняется — это надстройка, совместимая
    с уже обученным чекпоинтом (encoder можно инициализировать из него)."""

    def __init__(self, base_encoder: nn.Module, latent_dim: int = 128,
                hidden_dim: int = 64):
        super().__init__()
        self.encoder = base_encoder
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(latent_dim, hidden_dim)

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def encode_frame(self, o: torch.Tensor) -> torch.Tensor:
        """Кадр -> z1, без истории (совместимо со старым использованием)."""
        return self.encoder(o)

    def step(self, z1: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Один шаг рекуррентного накопления: z1_t, h_{t-1} -> h_t."""
        return self.gru(z1, h_prev)

    def encode_sequence(self, obs_seq: torch.Tensor) -> tuple:
        """obs_seq: (B, T, 1, 64, 64) -> (z1_all: (B,T,latent), h_all: (B,T,hidden))."""
        B, T = obs_seq.shape[0], obs_seq.shape[1]
        z1_all = self.encoder(obs_seq.reshape(B * T, *obs_seq.shape[2:])).view(B, T, -1)
        h = self.init_hidden(B, obs_seq.device)
        h_list = []
        for t in range(T):
            h = self.step(z1_all[:, t], h)
            h_list.append(h)
        h_all = torch.stack(h_list, dim=1)
        return z1_all, h_all


@torch.no_grad()
def probe_with_and_without_memory(rec_enc, obs, states, device, bs=8):
    """Ridge-probe: сравнивает R² по каждой переменной состояния из z1
    (мгновенное) против [z1; h] (с историей). Прямой тест гипотезы, что
    скорости нужны именно памяти, а не большей ёмкости z1."""
    import numpy as np
    n_ep, T = obs.shape[0], obs.shape[1]
    Z1, H = [], []
    for i in range(0, n_ep, bs):
        o = torch.from_numpy(obs[i:i + bs]).to(device)
        z1, h = rec_enc.encode_sequence(o)
        Z1.append(z1.cpu()); H.append(h.cpu())
    Z1 = torch.cat(Z1).reshape(-1, Z1[0].shape[-1])
    H = torch.cat(H).reshape(-1, H[0].shape[-1])
    Y = torch.from_numpy(states.reshape(-1, states.shape[-1]))

    def ridge_r2(X, Y, device):
        n_tr = int(0.8 * len(X))
        X, Y = X.to(device), Y.to(device)
        Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
        Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
        reg = 1e-3 * torch.eye(Xtr.shape[1], device=device)
        w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
        P = Xte @ w
        ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
        ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
        return (1 - ss_res / ss_tot).numpy()

    r2_z1 = ridge_r2(Z1, Y, device)
    r2_z1h = ridge_r2(torch.cat([Z1, H], dim=1), Y, device)
    return r2_z1, r2_z1h


class InstantaneousExpansion(nn.Module):
    """Случайная МГНОВЕННАЯ (не рекуррентная) проекция z1 в то же число
    измерений, что и h — контроль третьего рода. Отличие от случайного GRU:
    здесь нет памяти вообще, каждый выход зависит только от текущего кадра.

    Зачем: случайный GRU в первом контроле мог давать прирост по двум разным
    причинам — (а) рекуррентность сама по себе что-то интегрирует (эффект
    reservoir computing), или (б) линейной регрессии просто помогают ЛЮБЫЕ
    лишние случайные признаки, безотносительно к тому, несут ли они историю.
    Если этот контроль (без памяти) даёт прирост, сравнимый со случайным
    GRU — дело в (б), и рекуррентность как таковая не важна. Если прирост
    здесь намного меньше — дело в (а), и именно рекуррентность даёт эффект.
    """
    def __init__(self, latent_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z1_all: torch.Tensor) -> torch.Tensor:
        """z1_all: (B, T, latent) -> (B, T, hidden), поэлементно, без памяти."""
        return self.net(z1_all)


@torch.no_grad()
def probe_three_way(base_enc, rec_enc, obs, states, device, latent_dim=128, bs=8):
    """Три контроля разом: только z1, z1+h(случайный GRU),
    z1+h(случайная мгновенная проекция той же размерности)."""
    import numpy as np
    n_ep, T = obs.shape[0], obs.shape[1]
    inst = InstantaneousExpansion(latent_dim, rec_enc.hidden_dim).to(device).eval()

    Z1, Hrnd, Hinst = [], [], []
    for i in range(0, n_ep, bs):
        o = torch.from_numpy(obs[i:i + bs]).to(device)
        z1, h_rnd = rec_enc.encode_sequence(o)
        h_inst = inst(z1)
        Z1.append(z1.cpu()); Hrnd.append(h_rnd.cpu()); Hinst.append(h_inst.cpu())
    Z1 = torch.cat(Z1).reshape(-1, Z1[0].shape[-1])
    Hrnd = torch.cat(Hrnd).reshape(-1, Hrnd[0].shape[-1])
    Hinst = torch.cat(Hinst).reshape(-1, Hinst[0].shape[-1])
    Y = torch.from_numpy(states.reshape(-1, states.shape[-1]))

    def ridge_r2(X, Y, device):
        n_tr = int(0.8 * len(X))
        X, Y = X.to(device), Y.to(device)
        Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
        Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
        reg = 1e-3 * torch.eye(Xtr.shape[1], device=device)
        w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
        P = Xte @ w
        ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
        ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
        return (1 - ss_res / ss_tot).numpy()

    r2_z1 = ridge_r2(Z1, Y, device)
    r2_z1_rndgru = ridge_r2(torch.cat([Z1, Hrnd], dim=1), Y, device)
    r2_z1_rndinst = ridge_r2(torch.cat([Z1, Hinst], dim=1), Y, device)
    return r2_z1, r2_z1_rndgru, r2_z1_rndinst
