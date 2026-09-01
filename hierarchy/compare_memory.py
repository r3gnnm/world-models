"""Нужна ли память? Сравнение безпамятного (MLP) и рекуррентного (GRU)
предикторов на частично наблюдаемой среде.

Обе модели обучаются ОДНИМ кодом на ОДНИХ последовательностях, отличаясь
только наличием скрытого состояния — значит разница в метриках объясняется
именно памятью.

Методологические улучшения против stress_test.py:
  * action_gap НОРМИРОВАН (делится на ошибку при случайном действии),
    поэтому сравним между средами с разным уровнем шума.
  * probe считается ДВУМЯ способами: из z (что видно в кадре) и из (z, h)
    (что модель помнит). Разница между ними — вклад памяти.
  * ошибка probe разбивается ПО ПОЗИЦИЯМ — карта "где модель понимает,
    где она находится".

Запуск:
  python compare_memory.py --env egocentric --episodes 150 --epochs 20
  python compare_memory.py --env base --episodes 150 --epochs 20
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from env_variants import VARIANTS
from models import Encoder
from models_recurrent import RecurrentPredictor, MLPPredictorSeq
from losses import vicreg_loss


def collect_sequences(env_cls, n_episodes, ep_len, seed=0):
    """Возвращает (obs, actions, states) формы (эпизоды, шаги, ...)."""
    env = env_cls(seed=seed)
    rng = np.random.default_rng(seed)
    O, A, S = [], [], []
    for _ in range(n_episodes):
        o = env.reset()
        obs_ep, act_ep, st_ep = [o], [], [env.state.copy()]
        a = rng.uniform(-1, 1, 2)
        for _ in range(ep_len):
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1)
            o = env.step(a)
            obs_ep.append(o); act_ep.append(a.astype(np.float32))
            st_ep.append(env.state.copy())
        O.append(np.stack(obs_ep)); A.append(np.stack(act_ep)); S.append(np.stack(st_ep))
    return np.stack(O), np.stack(A), np.stack(S)


def train(pred_cls, obs, acts, epochs, latent_dim, device, bs=32, lr=3e-4):
    """Обучение энкодера + предиктора на последовательностях (BPTT)."""
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    enc = Encoder(latent_dim).to(device)
    pred = pred_cls(latent_dim).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=1e-5)
    obs_t = torch.from_numpy(obs)
    acts_t = torch.from_numpy(acts)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_ep)
        tot_gap = tot_loss = n_batches = 0
        for i in range(0, n_ep, bs):
            idx = perm[i:i + bs]
            o = obs_t[idx].to(device)            # (B, T+1, 1, 64, 64)
            a = acts_t[idx].to(device)           # (B, T, 2)
            B = o.shape[0]
            z_all = enc(o.view(B * (T + 1), *o.shape[2:])).view(B, T + 1, -1)
            h = pred.init_hidden(B, device)
            loss = 0.0
            gap_num = gap_den = 0.0
            for t in range(T):
                z_hat, h = pred(z_all[:, t], a[:, t], h)
                l, _ = vicreg_loss(z_hat, z_all[:, t + 1])
                loss = loss + l
                with torch.no_grad():
                    z_rand, _ = pred(z_all[:, t], a[torch.randperm(B), t], h)
                    e_rand = F.mse_loss(z_rand, z_all[:, t + 1]).item()
                    e_true = F.mse_loss(z_hat, z_all[:, t + 1]).item()
                    gap_num += e_rand - e_true; gap_den += e_rand
            loss = loss / T
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 5.0)
            opt.step()
            tot_loss += loss.item()
            tot_gap += gap_num / max(gap_den, 1e-8)   # НОРМИРОВАННЫЙ gap
            n_batches += 1
        if ep % 5 == 0 or ep == epochs:
            print(f"    epoch {ep:3d} | loss {tot_loss/n_batches:.4f} "
                  f"| action_gap_norm {tot_gap/n_batches:.4f}", flush=True)
    return enc, pred, tot_gap / n_batches


@torch.no_grad()
def probe(enc, pred, obs, acts, states, device, use_memory: bool):
    """Ridge-probe позиции. use_memory=True -> признаки (z, h), иначе только z."""
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    feats, targets = [], []
    for i in range(0, n_ep, 16):
        o = torch.from_numpy(obs[i:i + 16]).to(device)
        a = torch.from_numpy(acts[i:i + 16]).to(device)
        B = o.shape[0]
        z_all = enc(o.view(B * (T + 1), *o.shape[2:])).view(B, T + 1, -1)
        h = pred.init_hidden(B, device)
        for t in range(T):
            f = torch.cat([z_all[:, t], h], -1) if use_memory else z_all[:, t]
            feats.append(f.cpu())
            targets.append(torch.from_numpy(states[i:i + 16, t]))
            _, h = pred(z_all[:, t], a[:, t], h)
    X = torch.cat(feats).to(device)
    Y = torch.cat(targets).to(device)
    n_tr = int(0.8 * len(X))
    Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
    Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
    reg = 1e-3 * torch.eye(Xtr.shape[1], device=device)
    w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
    P = Xte @ w
    ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
    ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).cpu().numpy()
    per_point = ((Y[n_tr:] - P) ** 2).sum(-1).sqrt().cpu().numpy()
    return r2, Y[n_tr:].cpu().numpy(), per_point


def probe_error_map(states, errors, out, title, bins=14):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.zeros((bins, bins)); cnt = np.zeros((bins, bins))
    for (x, y), e in zip(states, errors):
        i, j = min(int(y * bins), bins - 1), min(int(x * bins), bins - 1)
        grid[i, j] += e; cnt[i, j] += 1
    grid = np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(grid, cmap="inferno", origin="upper", extent=[0, 64, 64, 0])
    fig.colorbar(im, label="ошибка probe (позиция)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"    сохранено: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="egocentric")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--ep-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--out", type=str, default="memory_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | среда: {args.env}")
    obs, acts, states = collect_sequences(VARIANTS[args.env], args.episodes,
                                          args.ep_len)
    print(f"данные: {obs.shape[0]} эпизодов по {obs.shape[1]} кадров")

    results = {}
    for label, cls in [("MLP (без памяти)", MLPPredictorSeq),
                       ("GRU (с памятью)", RecurrentPredictor)]:
        print(f"\n--- {label} ---")
        enc, pred, gap = train(cls, obs, acts, args.epochs, args.latent_dim, device)
        enc.eval(); pred.eval()
        r2_z, st, err_z = probe(enc, pred, obs, acts, states, device, use_memory=False)
        r2_zh, _, err_zh = probe(enc, pred, obs, acts, states, device, use_memory=True)
        key = "mlp" if "MLP" in label else "gru"
        results[key] = {"action_gap_norm": round(float(gap), 4),
                        "probe_r2_from_z": [round(float(v), 3) for v in r2_z],
                        "probe_r2_from_z_and_h": [round(float(v), 3) for v in r2_zh]}
        print(f"    probe R2 из z      : x={r2_z[0]:.3f} y={r2_z[1]:.3f}")
        print(f"    probe R2 из (z, h) : x={r2_zh[0]:.3f} y={r2_zh[1]:.3f}")
        probe_error_map(st, err_zh, f"probe_map_{key}.png",
                        f"{label}: где модель знает, где она")

    print("\n" + "=" * 66)
    print(f"{'модель':<20}{'gap_norm':>12}{'R2 из z':>16}{'R2 из (z,h)':>16}")
    print("-" * 66)
    for k, v in results.items():
        rz = np.mean(v["probe_r2_from_z"]); rzh = np.mean(v["probe_r2_from_z_and_h"])
        print(f"{k:<20}{v['action_gap_norm']:>12.4f}{rz:>16.3f}{rzh:>16.3f}")
    print("=" * 66)
    print("Если у GRU R2 из (z,h) заметно выше, чем R2 из z — память реально"
          "\nвосстанавливает то, чего нет в одном кадре.")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nСохранено в {args.out}")


if __name__ == "__main__":
    main()
