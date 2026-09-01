"""Пост-хок диагностика: использует ли предиктор действие, если сравнивать
ЧЕСТНО — внутри одной и той же траектории, а не между разными эпизодами.

Почему train_quad2d.py's action_gap_h может обманывать на этом масштабе мира.
Та аблация перемешивает действия МЕЖДУ элементами батча — то есть сравнивает
"эта траектория с своим действием" против "эта траектория с чужим действием
из совсем другого полёта". В маленьком мире комнатных сред это работало,
потому что весь мир был сравним по масштабу с одним действием. Здесь мир
128×128, 300 разных эпизодов — межтраекторная изменчивость (где вообще
летит дрон) огромна по сравнению с эффектом одного действия, и гасит сигнал
в знаменателе, даже если модель прекрасно использует действие.

Правильная проверка: один и тот же старт z0, один и тот же РЕАЛЬНЫЙ горизонт
действий против ЗАМЕНЁННОГО (обнулённого или развёрнутого) — то есть то,
что менялось, это только действие, а не "какой это вообще полёт".

Запуск:  python eval_quad2d.py --ckpt checkpoints/quad2d_jepa.pt --horizon 15
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from models import Encoder, Predictor


@torch.no_grad()
def same_trajectory_gap(enc, pred, obs, acts, horizon, device, n_samples=200):
    """Для каждой из n_samples стартовых точек: катим РЕАЛЬНЫЕ действия,
    ОБНУЛЁННЫЕ (=держать текущее состояние, для квадрокоптера это разумное
    действие, а не мусор — слабый контраст) и СЛУЧАЙНЫЕ (равномерно из
    [-1,1] на каждом шаге — резкий, недвусмысленный контраст) — из одного
    и того же z0, сравниваем с реальным будущим состоянием той же
    траектории."""
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    rng = np.random.default_rng(0)

    err_true, err_zero, err_random = [], [], []
    for _ in range(n_samples):
        ep = rng.integers(0, n_ep)
        t0 = rng.integers(0, max(T - horizon, 1))
        o = torch.from_numpy(obs[ep, t0:t0 + horizon + 1]).to(device)
        a = torch.from_numpy(acts[ep, t0:t0 + horizon]).to(device)

        z_all = enc(o)
        z0 = z_all[0:1]
        z_target = z_all[horizon:horizon + 1]

        z_true = z0.clone()
        for h in range(horizon):
            z_true = pred(z_true, a[h:h + 1])

        z_zero = z0.clone()
        zero_a = torch.zeros(1, a.shape[-1], device=device)
        for h in range(horizon):
            z_zero = pred(z_zero, zero_a)

        z_rnd = z0.clone()
        for h in range(horizon):
            rnd_a = torch.from_numpy(
                rng.uniform(-1, 1, (1, a.shape[-1])).astype(np.float32)).to(device)
            z_rnd = pred(z_rnd, rnd_a)

        err_true.append(F.mse_loss(z_true, z_target).item())
        err_zero.append(F.mse_loss(z_zero, z_target).item())
        err_random.append(F.mse_loss(z_rnd, z_target).item())

    return np.array(err_true), np.array(err_zero), np.array(err_random)


@torch.no_grad()
def fit_probe(enc, obs, states, device):
    """Ridge-регрессия z -> состояние, для декодирования предсказанных
    латентов в интерпретируемые единицы (позиция, угол) для визуализации."""
    n_ep, T1 = obs.shape[0], obs.shape[1]
    Z = []
    for i in range(0, n_ep, 8):
        o = torch.from_numpy(obs[i:i + 8]).to(device)
        B = o.shape[0]
        Z.append(enc(o.view(B * T1, *o.shape[2:])).view(B, T1, -1).cpu())
    Z = torch.cat(Z).reshape(-1, Z[0].shape[-1]).to(device)
    Y = torch.from_numpy(states.reshape(-1, states.shape[-1])).to(device)
    Zb = torch.cat([Z, torch.ones(len(Z), 1, device=device)], 1)
    reg = 1e-3 * torch.eye(Zb.shape[1], device=device)
    w = torch.linalg.solve(Zb.T @ Zb + reg, Zb.T @ Y)
    return w


@torch.no_grad()
def visualize_rollout(enc, pred, w, obs, acts, states, horizon, device,
                      out="quad2d_rollout.png", n_episodes=4, seed=1):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    fig, axes = plt.subplots(n_episodes, 3, figsize=(11, 2.6 * n_episodes))

    for row in range(n_episodes):
        ep = rng.integers(0, n_ep)
        t0 = rng.integers(0, max(T - horizon, 1))
        o = torch.from_numpy(obs[ep, t0:t0 + horizon + 1]).to(device)
        a = torch.from_numpy(acts[ep, t0:t0 + horizon]).to(device)
        true_state = states[ep, t0:t0 + horizon + 1]   # (horizon+1, 6)

        z = enc(o[0:1])
        pred_states = [torch.cat([z, torch.ones(1, 1, device=device)], 1) @ w]
        for h in range(horizon):
            z = pred(z, a[h:h + 1])
            pred_states.append(torch.cat([z, torch.ones(1, 1, device=device)], 1) @ w)
        pred_states = torch.cat(pred_states).cpu().numpy()   # (horizon+1, 6)

        for col, (name, idx) in enumerate([("x", 0), ("z", 1), ("theta", 4)]):
            ax = axes[row, col]
            ax.plot(true_state[:, idx], "-o", ms=3, color="tab:blue", label="реальность")
            ax.plot(pred_states[:, idx], "-o", ms=3, color="tab:red", label="воображение")
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"ep{ep} t{t0}")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Предсказанная (воображение) vs реальная траектория, в единицах probe")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/quad2d_jepa.pt")
    p.add_argument("--data", type=str, default="data/quad2d.npz")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--n-samples", type=int, default=200)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval(); enc.load_state_dict(ck["encoder"])
    pred = Predictor(ck["latent_dim"], action_dim=ck.get("action_dim", 2)).to(device).eval()
    pred.load_state_dict(ck["predictor"])

    d = np.load(args.data)
    err_true, err_zero, err_rnd = same_trajectory_gap(
        enc, pred, d["obs"], d["actions"], args.horizon, device, args.n_samples)

    print(f"Горизонт {args.horizon} шагов, {args.n_samples} стартовых точек "
          f"из ОДНОЙ и той же траектории каждая (не между эпизодами):\n")
    print(f"  реальные действия   -> MSE {err_true.mean():.4f} (±{err_true.std():.4f})")
    print(f"  обнулённые действия -> MSE {err_zero.mean():.4f} (±{err_zero.std():.4f})")
    print(f"  СЛУЧАЙНЫЕ действия  -> MSE {err_rnd.mean():.4f} (±{err_rnd.std():.4f})")

    gap_zero = (err_zero.mean() - err_true.mean()) / max(err_zero.mean(), 1e-8)
    gap_rnd = (err_rnd.mean() - err_true.mean()) / max(err_rnd.mean(), 1e-8)
    print(f"\n  gap vs обнулённые: {gap_zero:+.3f} "
          f"(слабый контраст: a=0 — тоже разумная команда для дрона)")
    print(f"  gap vs случайные:  {gap_rnd:+.3f} "
          f"(резкий контраст: главный, самый показательный тест)")

    print("\nСтрою визуализацию rollout (воображение vs реальность)...")
    w = fit_probe(enc, d["obs"], d["states"], device)
    visualize_rollout(enc, pred, w, d["obs"], d["actions"], d["states"],
                      args.horizon, device)
