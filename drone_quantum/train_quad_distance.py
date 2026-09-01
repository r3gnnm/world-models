"""Temporal-distance метрика для квадрокоптера — то же лечение, что уже
дважды применялось в проекте (комнаты, открытый мир).

Диагноз (измерено, не предположено): корреляция сырого L2-расстояния в
латенте с реальной пространственной дистанцией составляет 0.537. В
комнатных средах даже 0.67 не позволяло CEM планировать (агент застревал);
обученная temporal-distance метрика поднимала это до 0.86, и планирование
начинало работать. Здесь исходная ситуация ещё хуже, что полностью
объясняет 0/10 успешных долётов модельного планировщика против 10/10 на
истинной физике.

Обучающий сигнал — тот же self-supervised приём: пары состояний из одной
траектории, разделённые k шагами, должны иметь расстояние ~k; пары из
разных эпизодов получают hinge-штраф (не жёсткую цель — многие такие пары
случайно близки, и жёсткая цель дестабилизирует обучение, это выяснилось
ещё в первой реализации для комнат).

Запуск:  python train_quad_distance.py --ckpt checkpoints/quad2d_150ep.pt \
             --data data/quad2d.npz --steps 5000
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Encoder
from env_quad2d import Quad2DEnv, SIZE, GROUND_Z, CEILING_Z


class QuadDistanceNet(nn.Module):
    """Симметричная обучаемая метрика d(z_a, z_b) >= 0."""
    def __init__(self, latent_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Softplus(),
        )

    def forward(self, za, zb):
        return 0.5 * (self.net(torch.cat([za, zb], -1)).squeeze(-1)
                      + self.net(torch.cat([zb, za], -1)).squeeze(-1))


@torch.no_grad()
def encode_episodes(enc, obs, device, bs=8):
    """obs: (n_ep, T, 1, 64, 64) -> (n_ep, T, latent)"""
    n_ep, T = obs.shape[0], obs.shape[1]
    Z = []
    for i in range(0, n_ep, bs):
        o = torch.from_numpy(obs[i:i + bs]).to(device)
        B = o.shape[0]
        Z.append(enc(o.reshape(B * T, *o.shape[2:])).view(B, T, -1).cpu())
    return torch.cat(Z)


def sample_batch(z_ep, max_k, device, bs=256):
    n_ep, T, _ = z_ep.shape
    ei = torch.randint(0, n_ep, (bs,))
    ti = torch.randint(0, T, (bs,))
    k = torch.randint(1, max_k + 1, (bs,))
    tj = torch.clamp(ti + k, max=T - 1)
    real_k = (tj - ti).float()
    za = z_ep[ei, ti].to(device)
    zb = z_ep[ei, tj].to(device)
    ej = torch.randint(0, n_ep, (bs,))
    zn = z_ep[ej, torch.randint(0, T, (bs,))].to(device)
    return za, zb, real_k.to(device), zn


@torch.no_grad()
def diagnose(enc, dist_net, device, n_goals=4, n_probe=120, seed=0):
    """Сравнивает корреляцию сырого L2 и обученной метрики с реальной
    пространственной дистанцией — тот же диагностический протокол, что
    применялся для комнатных сред."""
    rng = np.random.default_rng(seed)
    l2_all, learned_all, real_all = [], [], []
    for g in range(n_goals):
        env = Quad2DEnv(seed=g); env.reset()
        for _ in range(300):
            target = rng.uniform([20, GROUND_Z + 15], [SIZE - 20, CEILING_Z - 15])
            if not env._collides(target):
                break
        goal_env = Quad2DEnv(seed=0, obstacles=False)
        goal_env.obs_pos, goal_env.obs_r = env.obs_pos, env.obs_r
        goal_env.pos = target.astype(np.float32)
        goal_env.vel = np.zeros(2, np.float32); goal_env.theta = 0.; goal_env.omega = 0.
        z_goal = enc(torch.from_numpy(goal_env.render()[None]).to(device))

        for _ in range(n_probe):
            p = rng.uniform([15, GROUND_Z + 10], [SIZE - 15, CEILING_Z - 10])
            if env._collides(p):
                continue
            env.pos = p.astype(np.float32)
            env.vel = np.zeros(2, np.float32); env.theta = 0.; env.omega = 0.
            z = enc(torch.from_numpy(env.render()[None]).to(device))
            l2_all.append(((z - z_goal) ** 2).mean().item())
            learned_all.append(dist_net(z, z_goal).item())
            real_all.append(float(np.linalg.norm(p - target)))

    l2_all, learned_all, real_all = map(np.array, (l2_all, learned_all, real_all))
    r_l2 = float(np.corrcoef(l2_all, real_all)[0, 1])
    r_learned = float(np.corrcoef(learned_all, real_all)[0, 1])
    return r_l2, r_learned


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/quad2d_150ep.pt")
    p.add_argument("--data", type=str, default="data/quad2d.npz")
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--out", type=str, default="checkpoints/quad2d_distance.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval()
    enc.load_state_dict(ck["encoder"])

    d = np.load(args.data)
    print(f"device: {device} | кодирую {d['obs'].shape[0]} эпизодов...")
    z_ep = encode_episodes(enc, d["obs"], device)

    dist_net = QuadDistanceNet(ck["latent_dim"]).to(device)
    opt = torch.optim.AdamW(dist_net.parameters(), lr=1e-3, weight_decay=1e-5)
    margin = float(args.max_k)

    for step in range(1, args.steps + 1):
        za, zb, k, zn = sample_batch(z_ep, args.max_k, device)
        pos_loss = F.smooth_l1_loss(dist_net(za, zb), k)
        neg_loss = torch.relu(margin - dist_net(za, zn)).mean()
        loss = pos_loss + neg_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} "
                  f"(pos {pos_loss.item():.3f}, neg {neg_loss.item():.3f})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"dist": dist_net.state_dict(), "latent_dim": ck["latent_dim"],
                "base_ckpt": args.ckpt}, args.out)
    print(f"\nСохранено в {args.out}")

    dist_net.eval()
    r_l2, r_learned = diagnose(enc, dist_net, device)
    print("\n--- Диагностика метрики (корреляция с реальной дистанцией) ---")
    print(f"  сырой L2 в латенте : {r_l2:.3f}")
    print(f"  обученная метрика  : {r_learned:.3f}")
    if r_learned > r_l2 + 0.05:
        print("  OK: метрика лучше L2 — есть смысл подключать к планировщику")
    else:
        print("  ВНИМАНИЕ: улучшения нет. Увеличь --steps или --max-k.")
