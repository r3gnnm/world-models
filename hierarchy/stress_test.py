"""Стресс-тест: обучает и оценивает мировую модель на каждом варианте среды.

Для каждого варианта: собирает данные -> обучает JEPA -> меряет метрики.
Итог — таблица, показывающая, ЧТО именно ломается в каждом усложнении.

Метрики:
  action_gap  — использует ли предиктор действие (главный показатель здоровья)
  probe R^2   — восстановима ли позиция агента из латента
  rollout@10  — ошибка воображения на 10 шагов
  z_std       — нет ли коллапса представлений

Ожидания (гипотезы для проверки):
  three_rooms — метрики близки к base; топология сложнее, но задача та же
  distractor  — probe R^2 может слегка просесть; ключевое: action_gap НЕ должен
                рухнуть (JEPA игнорирует непредсказуемое)
  fog         — probe R^2 должен заметно упасть: по окну вокруг агента нельзя
                восстановить глобальную позицию без памяти

Запуск:  python stress_test.py --transitions 20000 --epochs 20
         python stress_test.py --only fog --epochs 30
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from env_variants import VARIANTS
from models import Encoder, Predictor
from losses import vicreg_loss


def collect(env_cls, n_transitions, episode_len=200, seed=0):
    env = env_cls(seed=seed)
    rng = np.random.default_rng(seed)
    O, A, O1, S = [], [], [], []
    while len(O) < n_transitions:
        o = env.reset()
        a = rng.uniform(-1, 1, 2)
        for _ in range(episode_len):
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1)
            s = env.state.copy()
            o1 = env.step(a)
            O.append(o); A.append(a.astype(np.float32)); O1.append(o1); S.append(s)
            o = o1
            if len(O) >= n_transitions:
                break
    return (np.stack(O), np.stack(A), np.stack(O1), np.stack(S))


def train(obs, acts, next_obs, epochs, latent_dim, device, bs=256, lr=3e-4):
    ds = TensorDataset(torch.from_numpy(obs), torch.from_numpy(acts),
                       torch.from_numpy(next_obs))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    enc, pred = Encoder(latent_dim).to(device), Predictor(latent_dim).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=1e-5)
    gap = z_std = 0.0
    for ep in range(1, epochs + 1):
        gap = z_std = 0.0
        for o, a, o1 in loader:
            o, a, o1 = o.to(device), a.to(device), o1.to(device)
            z, z1 = enc(o), enc(o1)
            z1_hat = pred(z, a)
            loss, _ = vicreg_loss(z1_hat, z1)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                z1_rand = pred(z, a[torch.randperm(len(a))])
                gap += (F.mse_loss(z1_rand, z1) - F.mse_loss(z1_hat, z1)).item()
                z_std += z.std(0).mean().item()
        gap /= len(loader); z_std /= len(loader)
        if ep % 5 == 0 or ep == epochs:
            print(f"    epoch {ep:3d} | action_gap {gap:.4f} | z_std {z_std:.3f}",
                  flush=True)
    return enc, pred, gap, z_std


@torch.no_grad()
def probe_r2(enc, obs, states, device, split=0.8):
    n = len(obs); n_tr = int(split * n)
    x = torch.from_numpy(obs).to(device)
    zs = torch.cat([enc(x[i:i + 512]) for i in range(0, n, 512)])
    s = torch.from_numpy(states).to(device)
    ztr = torch.cat([zs[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
    zte = torch.cat([zs[n_tr:], torch.ones(n - n_tr, 1, device=device)], 1)
    reg = 1e-3 * torch.eye(ztr.shape[1], device=device)
    w = torch.linalg.solve(ztr.T @ ztr + reg, ztr.T @ s[:n_tr])
    p = zte @ w
    ss_res = ((s[n_tr:] - p) ** 2).sum(0)
    ss_tot = ((s[n_tr:] - s[n_tr:].mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).cpu().numpy()


@torch.no_grad()
def rollout_err(enc, pred, env_cls, device, horizon=10, n_ep=30, seed=99):
    env = env_cls(seed=seed)
    rng = np.random.default_rng(seed)
    errs = np.zeros(horizon)
    for _ in range(n_ep):
        o = env.reset()
        z = enc(torch.from_numpy(o[None]).to(device))
        a = rng.uniform(-1, 1, 2)
        for k in range(horizon):
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1).astype(np.float32)
            o = env.step(a)
            z = pred(z, torch.from_numpy(a[None]).to(device))
            z_real = enc(torch.from_numpy(o[None]).to(device))
            errs[k] += F.mse_loss(z, z_real).item()
    return errs / n_ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--only", type=str, default=None,
                   help="прогнать один вариант: base|three_rooms|distractor|fog")
    p.add_argument("--out", type=str, default="stress_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [args.only] if args.only else list(VARIANTS)
    results = {}

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        env_cls = VARIANTS[name]
        obs, acts, next_obs, states = collect(env_cls, args.transitions)
        enc, pred, gap, z_std = train(obs, acts, next_obs, args.epochs,
                                      args.latent_dim, device)
        enc.eval(); pred.eval()
        r2 = probe_r2(enc, obs, states, device)
        errs = rollout_err(enc, pred, env_cls, device)
        results[name] = {"action_gap": round(gap, 4), "z_std": round(z_std, 3),
                         "probe_r2_x": round(float(r2[0]), 3),
                         "probe_r2_y": round(float(r2[1]), 3),
                         "rollout_1": round(float(errs[0]), 4),
                         "rollout_10": round(float(errs[-1]), 4)}
        print(f"  -> {results[name]}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'вариант':<14}{'action_gap':>12}{'z_std':>8}{'probe R2 x':>12}"
          f"{'probe R2 y':>12}{'rollout@10':>12}")
    print("-" * 78)
    for k, v in results.items():
        print(f"{k:<14}{v['action_gap']:>12.4f}{v['z_std']:>8.3f}"
              f"{v['probe_r2_x']:>12.3f}{v['probe_r2_y']:>12.3f}"
              f"{v['rollout_10']:>12.4f}")
    print("=" * 78)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены в {args.out}")


if __name__ == "__main__":
    main()
