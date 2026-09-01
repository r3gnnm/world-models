"""Калибровка неопределённости ансамбля.

Вопрос: совпадает ли РАЗБРОС ансамбля (то, где модель "сомневается")
с РЕАЛЬНОЙ ошибкой (то, где она действительно ошибается)?

Три анализа:
  1. Одношаговая калибровка: корреляция (Пирсон + Спирмен) между
     disagreement и фактической ошибкой на тестовых переходах.
  2. Пространственная карта: средний disagreement по позициям агента.
     Гипотеза: неуверенность концентрируется у проёма в перегородке.
  3. Multi-step: рост disagreement и ошибки с горизонтом воображения.

Запуск:  python eval_uncertainty.py --ensemble checkpoints/ensemble.pt
Выход:   calibration_scatter.png, uncertainty_map.png, rollout_uncertainty.png
"""
import argparse
import numpy as np
import torch

from env import TwoRoomsEnv, SIZE
from models import Encoder, Predictor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def one_step_analysis(enc, members, data, device, n_test=5000):
    obs = torch.from_numpy(data["obs"][-n_test:])
    a = torch.from_numpy(data["actions"][-n_test:]).to(device)
    next_obs = torch.from_numpy(data["next_obs"][-n_test:])
    states = data["states"][-n_test:]

    z = enc(obs.to(device))
    z1 = enc(next_obs.to(device))
    preds = torch.stack([m(z, a) for m in members])          # (K, N, D)

    disagreement = preds.std(dim=0).mean(dim=-1)             # (N,)
    error = ((preds.mean(0) - z1) ** 2).mean(dim=-1)         # (N,)
    dis, err = disagreement.cpu().numpy(), error.cpu().numpy()

    pearson = float(np.corrcoef(dis, err)[0, 1])
    rho = spearman(dis, err)
    print(f"Одношаговая калибровка: Pearson r = {pearson:.3f}, "
          f"Spearman rho = {rho:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(dis, err, s=3, alpha=0.15)
    ax.set_xlabel("disagreement ансамбля")
    ax.set_ylabel("реальная ошибка (MSE)")
    ax.set_title(f"Калибровка: r={pearson:.2f}, rho={rho:.2f}")
    fig.tight_layout(); fig.savefig("calibration_scatter.png", dpi=150)

    return dis, states


def uncertainty_map(dis, states, bins=16):
    grid = np.zeros((bins, bins)); cnt = np.zeros((bins, bins))
    for d, (x, y) in zip(dis, states):
        i, j = min(int(y * bins), bins - 1), min(int(x * bins), bins - 1)
        grid[i, j] += d; cnt[i, j] += 1
    grid = np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(grid, cmap="inferno", origin="upper",
                   extent=[0, SIZE, SIZE, 0])
    walls = TwoRoomsEnv().walls
    ax.contour(walls, levels=[0.5], colors="cyan", linewidths=1,
               extent=[0, SIZE, 0, SIZE], origin="upper")
    fig.colorbar(im, label="средний disagreement")
    ax.set_title("Карта неуверенности модели динамики")
    fig.tight_layout(); fig.savefig("uncertainty_map.png", dpi=150)
    print("Сохранено: uncertainty_map.png")


@torch.no_grad()
def rollout_analysis(enc, members, device, horizon=30, n_episodes=40, seed=7):
    env = TwoRoomsEnv(seed=seed)
    rng = np.random.default_rng(seed)
    dis_curve = np.zeros(horizon); err_curve = np.zeros(horizon)
    for _ in range(n_episodes):
        o = env.reset()
        z0 = enc(torch.from_numpy(o[None]).to(device))
        zs = [z0.clone() for _ in members]        # каждый член катит свой латент
        a = rng.uniform(-1, 1, size=2)
        for t in range(horizon):
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1).astype(np.float32)
            at = torch.from_numpy(a[None]).to(device)
            zs = [m(zk, at) for m, zk in zip(members, zs)]
            stack = torch.stack(zs)                              # (K, 1, D)
            o = env.step(a)
            z_real = enc(torch.from_numpy(o[None]).to(device))
            dis_curve[t] += stack.std(0).mean().item()
            err_curve[t] += ((stack.mean(0) - z_real) ** 2).mean().item()
    dis_curve /= n_episodes; err_curve /= n_episodes

    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax1.plot(err_curve, color="tab:red", label="реальная ошибка")
    ax1.set_xlabel("шаг воображения"); ax1.set_ylabel("MSE", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(dis_curve, color="tab:blue", label="disagreement")
    ax2.set_ylabel("disagreement", color="tab:blue")
    ax1.set_title("Неопределённость и ошибка растут вместе?")
    fig.tight_layout(); fig.savefig("rollout_uncertainty.png", dpi=150)
    r = float(np.corrcoef(dis_curve, err_curve)[0, 1])
    print(f"Multi-step: корреляция кривых disagreement/ошибка r = {r:.3f}")
    print("Сохранено: rollout_uncertainty.png, calibration_scatter.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/transitions.npz")
    p.add_argument("--ensemble", type=str, default="checkpoints/ensemble.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ens = torch.load(args.ensemble, map_location=device, weights_only=True)
    enc_ckpt = torch.load(ens["encoder_ckpt"], map_location=device,
                          weights_only=True)
    enc = Encoder(ens["latent_dim"]).to(device).eval()
    enc.load_state_dict(enc_ckpt["encoder"])
    members = []
    for sd in ens["members"]:
        m = Predictor(ens["latent_dim"]).to(device).eval()
        m.load_state_dict(sd)
        members.append(m)

    data = np.load(args.data)
    dis, states = one_step_analysis(enc, members, data, device)
    uncertainty_map(dis, states)
    rollout_analysis(enc, members, device)
