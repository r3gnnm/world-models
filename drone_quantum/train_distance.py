"""Обучение temporal-distance метрики (правильное решение проблемы L2).

Идея: L2-расстояние в латенте не соответствует реальной достижимости
(два состояния по разные стороны стены близки в пикселях, но далеки по числу
шагов). Учим сеть d(z_a, z_b) ~ сколько шагов между состояниями.

Обучающий сигнал берём бесплатно из траекторий: пары состояний, разделённые
k шагами в одном эпизоде, должны иметь расстояние ~k. Отрицательные пары
(из разных эпизодов / далёкие) — большое расстояние. Это self-supervised:
разметка не нужна, всё из собранных переходов.

Использование: замени в планировщике L2-cost на dist_net(z, z_goal).

Запуск:  python train_distance.py --ckpt checkpoints/jepa_step8.pt
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Encoder


class DistanceNet(nn.Module):
    """Симметричная обучаемая метрика: d(z_a, z_b) >= 0."""
    def __init__(self, latent_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Softplus(),          # неотрицательный выход
        )

    def forward(self, za, zb):
        # симметризуем: d(a,b) = d(b,a)
        return 0.5 * (self.net(torch.cat([za, zb], -1)).squeeze(-1)
                      + self.net(torch.cat([zb, za], -1)).squeeze(-1))


@torch.no_grad()
def encode_episodes(enc, obs, ep_len, device, bs=512):
    zs = []
    for i in range(0, len(obs), bs):
        zs.append(enc(obs[i:i + bs].to(device)).cpu())
    z = torch.cat(zs)
    n_ep = len(z) // ep_len
    return z[: n_ep * ep_len].view(n_ep, ep_len, -1)      # (эпизоды, шаги, D)


def sample_batch(z_ep, max_k, device, bs=256):
    n_ep, ep_len, _ = z_ep.shape
    ei = torch.randint(0, n_ep, (bs,))
    ti = torch.randint(0, ep_len, (bs,))
    # положительные: разделены k шагами внутри эпизода (target = точное k)
    k = torch.randint(1, max_k + 1, (bs,))
    tj = torch.clamp(ti + k, max=ep_len - 1)
    real_k = (tj - ti).float()
    za = z_ep[ei, ti].to(device)
    zb = z_ep[ei, tj].to(device)
    # отрицательные: из другого эпизода. НЕ форсируем точное расстояние
    # (многие такие пары реально близки из-за случайного reset),
    # а лишь требуем "не ближе margin" через hinge — см. loss ниже.
    ej = torch.randint(0, n_ep, (bs,))
    zn = z_ep[ej, torch.randint(0, ep_len, (bs,))].to(device)
    return za, zb, real_k.to(device), zn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/transitions_step8.npz")
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa_step8.pt")
    p.add_argument("--ep-len", type=int, default=200)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", type=str, default="checkpoints/distance.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval(); enc.load_state_dict(ck["encoder"])

    d = np.load(args.data)
    obs = torch.from_numpy(d["obs"])
    z_ep = encode_episodes(enc, obs, args.ep_len, device)
    print(f"эпизодов: {z_ep.shape[0]}, длина: {z_ep.shape[1]}")

    dist = DistanceNet(ck["latent_dim"]).to(device)
    opt = torch.optim.AdamW(dist.parameters(), lr=1e-3, weight_decay=1e-5)

    margin = float(args.max_k)
    for step in range(1, args.steps + 1):
        za, zb, k, zn = sample_batch(z_ep, args.max_k, device)
        pos = F.smooth_l1_loss(dist(za, zb), k)              # точное k для позитивов
        neg = torch.relu(margin - dist(za, zn)).mean()      # hinge: не ближе margin
        loss = pos + neg
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} "
                  f"(pos {pos.item():.3f}, neg {neg.item():.3f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"dist": dist.state_dict(), "latent_dim": ck["latent_dim"],
                "encoder_ckpt": args.ckpt}, args.out)
    print(f"Метрика расстояния сохранена в {args.out}")

    # --- Встроенная диагностика: коррелирует ли метрика с реальным расстоянием? ---
    diagnose(enc, dist, args.data, device)


@torch.no_grad()
def diagnose(enc, dist, data_path, device, n_goals=5):
    """Печатает корреляцию обученной метрики и L2 с реальным пространственным
    расстоянием. Если обученная заметно выше L2 — метрика полезна для планирования."""
    import itertools
    from env import TwoRoomsEnv
    d = np.load(data_path)
    rng = np.random.default_rng(0)
    goal_idx = rng.integers(0, len(d["states"]), n_goals)
    grid = [(x, y) for x, y in itertools.product(range(6, 60, 3), range(6, 60, 3))]

    l2_all, learned_all, spa_all = [], [], []
    for gi in goal_idx:
        gp = d["states"][gi] * 64
        ge = TwoRoomsEnv(); ge.pos = gp.astype(np.float32)
        zg = enc(torch.from_numpy(ge.render()[None]).to(device))
        for x, y in grid:
            e = TwoRoomsEnv()
            if e._collides(np.array([x, y], np.float32)):
                continue
            e.pos = np.array([x, y], np.float32)
            z = enc(torch.from_numpy(e.render()[None]).to(device))
            l2_all.append(((z - zg) ** 2).mean().item())
            learned_all.append(dist(z, zg).item())
            spa_all.append(float(np.linalg.norm([x - gp[0], y - gp[1]])))
    l2_all, learned_all, spa_all = map(np.array, (l2_all, learned_all, spa_all))
    r_l2 = float(np.corrcoef(l2_all, spa_all)[0, 1])
    r_learned = float(np.corrcoef(learned_all, spa_all)[0, 1])
    print("\n--- Диагностика метрики (корреляция с реальным расстоянием) ---")
    print(f"  L2 в латенте      : {r_l2:.3f}")
    print(f"  обученная метрика : {r_learned:.3f}")
    if r_learned > r_l2 + 0.05:
        print("  OK: метрика лучше L2, можно планировать через plan_distance.py")
    else:
        print("  ВНИМАНИЕ: метрика не лучше L2. Увеличь --steps или --max-k.")


if __name__ == "__main__":
    main()
