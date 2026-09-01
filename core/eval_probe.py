"""Оценка выученной world model.

1) Linear probe: линейная регрессия из латента z в истинную позицию (x, y).
   Высокий R^2 => представление содержит состояние мира.
2) Multi-step rollout: катим предиктор на k шагов вперёд только по действиям
   и меряем ошибку против латентов реальных кадров. Показывает compounding error.

Запуск:  python eval_probe.py --data data/transitions.npz --ckpt checkpoints/jepa.pt
"""
import argparse
import numpy as np
import torch

from models import Encoder, Predictor
from env import TwoRoomsEnv


@torch.no_grad()
def linear_probe(enc, data, device, n_train=20_000, n_test=5_000):
    n_total = len(data["obs"])
    n_train = min(n_train, int(0.8 * n_total))
    n_test = min(n_test, n_total - n_train)
    obs = torch.from_numpy(data["obs"][: n_train + n_test]).to(device)
    states = torch.from_numpy(data["states"][: n_train + n_test]).to(device)
    zs = torch.cat([enc(obs[i:i + 512]) for i in range(0, len(obs), 512)])

    z_tr, z_te = zs[:n_train], zs[n_train:]
    s_tr, s_te = states[:n_train], states[n_train:]

    # ridge-регрессия в закрытой форме
    z_tr_b = torch.cat([z_tr, torch.ones(len(z_tr), 1, device=device)], 1)
    z_te_b = torch.cat([z_te, torch.ones(len(z_te), 1, device=device)], 1)
    reg = 1e-3 * torch.eye(z_tr_b.shape[1], device=device)
    w = torch.linalg.solve(z_tr_b.T @ z_tr_b + reg, z_tr_b.T @ s_tr)

    pred = z_te_b @ w
    ss_res = ((s_te - pred) ** 2).sum(0)
    ss_tot = ((s_te - s_te.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).cpu().numpy()
    return r2  # (R^2 по x, R^2 по y)


@torch.no_grad()
def rollout_error(enc, pred, device, horizon=30, n_episodes=50, seed=123):
    env = TwoRoomsEnv(seed=seed)
    rng = np.random.default_rng(seed)
    errs = np.zeros(horizon)
    for _ in range(n_episodes):
        o = env.reset()
        z = enc(torch.from_numpy(o[None]).to(device))
        a = rng.uniform(-1, 1, size=2)
        for k in range(horizon):
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1).astype(np.float32)
            o = env.step(a)
            z = pred(z, torch.from_numpy(a[None]).to(device))     # воображение
            z_real = enc(torch.from_numpy(o[None]).to(device))    # реальность
            errs[k] += torch.nn.functional.mse_loss(z, z_real).item()
    return errs / n_episodes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/transitions.npz")
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ckpt["latent_dim"]).to(device).eval()
    pred = Predictor(ckpt["latent_dim"]).to(device).eval()
    enc.load_state_dict(ckpt["encoder"])
    pred.load_state_dict(ckpt["predictor"])

    data = np.load(args.data)
    r2 = linear_probe(enc, data, device)
    print(f"Linear probe R^2: x = {r2[0]:.3f}, y = {r2[1]:.3f}  (цель: > 0.9)")

    errs = rollout_error(enc, pred, device)
    for k in (0, 4, 9, 19, 29):
        print(f"rollout step {k + 1:2d}: latent MSE = {errs[k]:.4f}")
