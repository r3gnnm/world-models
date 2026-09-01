"""Обучение JEPA на недоуправляемой 2.5D-динамике с MULTI-STEP loss.

Почему multi-step обязателен здесь, а не был нужен раньше. Во всех
предыдущих средах проекта действие задавало скорость напрямую — эффект на
следующий кадр был мгновенным, и одношагового loss хватало. У квадрокоптера
действие влияет на УСКОРЕНИЕ (и на угловое ускорение), поэтому видимое
смещение появляется лишь через несколько шагов интегрирования. При
одношаговом loss градиент от действия к наблюдаемому эффекту почти исчезает,
и предиктор выучивает "инерцию" (следующий кадр ≈ текущий), игнорируя
управление.

Решение: раскатываем предиктор на H шагов в латенте и штрафуем расхождение
на КАЖДОМ шаге, а не только на первом.

Диагностики (адаптированы под 2.5D):
  action_gap_h    — нормированный gap на горизонте H (а не на 1 шаге!)
  probe по 6 переменным — x, z, vx, vz, theta, omega по отдельности:
                    видно, что именно модель кодирует, а что теряет
  z_std           — контроль коллапса, как раньше

Запуск:  python train_quad2d.py --epochs 30 --horizon 5
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from models import Encoder, Predictor
from losses import vicreg_loss

STATE_NAMES = ["x", "z", "vx", "vz", "theta", "omega"]


def train(obs, acts, epochs, horizon, device, latent_dim=128, action_dim=2,
          bs=16, lr=3e-4, seed=0, sim_w=25.0, var_w=25.0, cov_w=1.0,
          in_channels=1):
    torch.manual_seed(seed)
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    enc = Encoder(latent_dim, in_channels=in_channels).to(device)
    pred = Predictor(latent_dim, action_dim=action_dim).to(device)
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    obs_t, acts_t = torch.from_numpy(obs), torch.from_numpy(acts)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_ep)
        agg = {"loss": 0.0, "gap": 0.0, "z_std": 0.0, "n": 0}
        for i in range(0, n_ep, bs):
            idx = perm[i:i + bs]
            o = obs_t[idx].to(device)
            a = acts_t[idx].to(device)
            B = o.shape[0]
            z_all = enc(o.view(B * T1, *o.shape[2:])).view(B, T1, -1)

            # случайное стартовое смещение, чтобы покрыть весь эпизод
            t0 = int(torch.randint(0, max(T - horizon, 1), (1,)).item())
            z = z_all[:, t0]
            loss = 0.0
            for h in range(horizon):
                z = pred(z, a[:, t0 + h])
                l, _ = vicreg_loss(z, z_all[:, t0 + h + 1],
                                   sim_w=sim_w, var_w=var_w, cov_w=cov_w)
                loss = loss + l
            loss = loss / horizon

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            with torch.no_grad():
                # НОРМИРОВАННЫЙ gap на горизонте H: катим с настоящими vs
                # перемешанными действиями и сравниваем ошибку в конце
                z_true = z_all[:, t0].clone()
                z_rand = z_all[:, t0].clone()
                perm_b = torch.randperm(B)
                for h in range(horizon):
                    z_true = pred(z_true, a[:, t0 + h])
                    z_rand = pred(z_rand, a[perm_b, t0 + h])
                tgt = z_all[:, t0 + horizon]
                e_true = F.mse_loss(z_true, tgt).item()
                e_rand = F.mse_loss(z_rand, tgt).item()
                agg["gap"] += (e_rand - e_true) / max(e_rand, 1e-8)
                agg["z_std"] += z_all[:, t0].std(0).mean().item()

            agg["loss"] += loss.item(); agg["n"] += 1

        n = agg["n"]
        if ep % 5 == 0 or ep == epochs:
            print(f"    epoch {ep:3d} | loss {agg['loss']/n:7.3f} "
                  f"| action_gap_h {agg['gap']/n:.4f} | z_std {agg['z_std']/n:.3f}",
                  flush=True)
    return enc, pred, agg["gap"] / agg["n"]


@torch.no_grad()
def probe_states(enc, obs, states, device, bs=8):
    """Ridge-probe по КАЖДОЙ из 6 переменных состояния отдельно —
    показывает, что модель закодировала (позицию? скорость? наклон?),
    а что потеряла."""
    n_ep, T1 = obs.shape[0], obs.shape[1]
    Z = []
    for i in range(0, n_ep, bs):
        o = torch.from_numpy(obs[i:i + bs]).to(device)
        B = o.shape[0]
        Z.append(enc(o.view(B * T1, *o.shape[2:])).view(B, T1, -1).cpu())
    Z = torch.cat(Z).reshape(-1, Z[0].shape[-1])
    Y = torch.from_numpy(states.reshape(-1, states.shape[-1]))

    n_tr = int(0.8 * len(Z))
    X, Y = Z.to(device), Y.to(device)
    Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
    Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
    reg = 1e-3 * torch.eye(Xtr.shape[1], device=device)
    w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
    P = Xte @ w
    ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
    ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).cpu().numpy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/quad2d.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-w", type=float, default=25.0,
                   help="вес точности предсказания — выше => меньше затухание амплитуды")
    p.add_argument("--var-w", type=float, default=25.0)
    p.add_argument("--cov-w", type=float, default=1.0)
    p.add_argument("--out", type=str, default="checkpoints/quad2d_jepa.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data)
    obs, acts, states = d["obs"], d["actions"], d["states"]
    print(f"device: {device} | данные: {obs.shape[0]} эпизодов × {obs.shape[1]} кадров"
          f" | горизонт multi-step: {args.horizon}")

    in_ch = obs.shape[2]
    print(f'каналов наблюдения: {in_ch}')
    enc, pred, gap = train(obs, acts, args.epochs, args.horizon, device,
                          latent_dim=args.latent_dim, seed=args.seed,
                          sim_w=args.sim_w, var_w=args.var_w, cov_w=args.cov_w,
                          in_channels=in_ch)
    enc.eval(); pred.eval()

    r2 = probe_states(enc, obs, states, device)
    print("\n" + "=" * 56)
    print("PROBE по переменным состояния (R²):")
    for name, v in zip(STATE_NAMES, r2):
        bar = "#" * int(max(v, 0) * 30)
        print(f"  {name:6s} {v:+.3f}  {bar}")
    print("-" * 56)
    print(f"  action_gap на горизонте {args.horizon}: {gap:.4f}")
    print("  (у недоуправляемой динамики одношаговый gap слаб по построению —")
    print("   смотреть надо именно на многошаговый)")
    print("=" * 56)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "predictor": pred.state_dict(),
                "latent_dim": args.latent_dim, "action_dim": 2,
                "in_channels": in_ch}, args.out)
    print(f"\nСохранено в {args.out}")
