"""Обучение GRU поверх замороженного encoder — чтобы h накопил историю,
нужную для восстановления скоростей (vx, vz, omega).

Задача для GRU: предсказывать z1_{t+1} используя (z1_t, h_t, a_t), а не
только (z1_t, a_t) как раньше. Если h действительно накопил что-то полезное
(историю движения, из которой выводится скорость), предиктор с контекстом
должен предсказывать лучше — и, что важнее для нас, из самого h должна
линейно восстанавливаться скорость, которую z1 сам по себе не содержит.

Энкодер ЗАМОРОЖЕН (уже обучен на x/z/theta) — тренируется только GRU и
небольшой предиктор поверх него, аналогично тому, как в hierarchy/
Abstractor обучался поверх замороженного z1.

Запуск:  python train_quad_recurrent.py --ckpt checkpoints/quad2d_150ep.pt \
                 --data data/quad2d.npz --epochs 40
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Encoder
from models_quad_recurrent import (RecurrentQuadEncoder, probe_three_way,
                                   InstantaneousExpansion)

STATE_NAMES = ["x", "z", "vx", "vz", "theta", "omega"]


class ContextPredictor(nn.Module):
    """(z1_t, h_t, a_t) -> z1_{t+1}. Расширенная версия Predictor с
    дополнительным входом h — контекстом из GRU."""
    def __init__(self, latent_dim=128, hidden_dim=64, action_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z1, h, a):
        return z1 + self.net(torch.cat([z1, h, a], -1))


def train_recurrent(base_enc, obs, acts, epochs, device, hidden_dim=64,
                    latent_dim=128, action_dim=2, bs=16, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    for p in base_enc.parameters():
        p.requires_grad_(False)   # encoder заморожен

    rec_enc = RecurrentQuadEncoder(base_enc, latent_dim, hidden_dim).to(device)
    pred = ContextPredictor(latent_dim, hidden_dim, action_dim).to(device)
    params = list(rec_enc.gru.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    obs_t, acts_t = torch.from_numpy(obs), torch.from_numpy(acts)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_ep)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n_ep, bs):
            idx = perm[i:i + bs]
            o = obs_t[idx].to(device)
            a = acts_t[idx].to(device)
            B = o.shape[0]

            with torch.no_grad():
                z1_all = base_enc(o.reshape(B * T1, *o.shape[2:])).view(B, T1, -1)

            h = rec_enc.init_hidden(B, device)
            loss = 0.0
            for t in range(T):
                h = rec_enc.step(z1_all[:, t], h)
                z1_pred = pred(z1_all[:, t], h, a[:, t])
                loss = loss + F.mse_loss(z1_pred, z1_all[:, t + 1])
            loss = loss / T

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total_loss += loss.item(); n_batches += 1

        if ep % 5 == 0 or ep == epochs:
            print(f"    epoch {ep:3d} | loss {total_loss / n_batches:.4f}", flush=True)

    return rec_enc, pred


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/quad2d_jepa.pt")
    p.add_argument("--data", type=str, default="data/quad2d.npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/quad2d_recurrent.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    base_enc = Encoder(ck["latent_dim"],
                       in_channels=ck.get("in_channels", 1)).to(device)
    base_enc.load_state_dict(ck["encoder"])

    d = np.load(args.data)
    print(f"device: {device} | данные: {d['obs'].shape[0]} эпизодов, "
          f"encoder заморожен (из {args.ckpt})")

    rec_enc, pred = train_recurrent(base_enc, d["obs"], d["actions"], args.epochs,
                                    device, hidden_dim=args.hidden_dim,
                                    latent_dim=ck["latent_dim"], seed=args.seed)
    rec_enc.eval()

    print("\nТри контроля: z1 (мгновенное) | z1+h обученный GRU | "
          "z1+h случайный GRU | z1+h случайная МГНОВЕННАЯ проекция —")
    print("ожидание: x/z/theta почти не меняются везде; vx/vz/omega растут")
    print("сильнее всего у обученного GRU, заметно слабее у случайного GRU,")
    print("и почти не растут у мгновенной проекции (если рекуррентность —")
    print("реальный, а не просто 'больше признаков' эффект)\n")

    r2_z1, r2_z1h = None, None
    # ВАЖНО: для случайного контроля нужен ОТДЕЛЬНЫЙ необученный GRU,
    # а не обученный rec_enc — иначе колонки "обучен"/"случаен" совпадут
    rnd_rec_enc = RecurrentQuadEncoder(base_enc, ck["latent_dim"],
                                       args.hidden_dim).to(device).eval()
    r2_z1, r2_rndgru, r2_rndinst = probe_three_way(
        base_enc, rnd_rec_enc, d["obs"], d["states"], device,
        latent_dim=ck["latent_dim"])

    # обученный GRU считаем тем же способом, что и раньше, для четвёртой колонки
    @torch.no_grad()
    def ridge_r2_single(rec_enc_local, obs, states, device, bs=8):
        n_ep, T = obs.shape[0], obs.shape[1]
        Z1, H = [], []
        for i in range(0, n_ep, bs):
            o = torch.from_numpy(obs[i:i + bs]).to(device)
            z1, h = rec_enc_local.encode_sequence(o)
            Z1.append(z1.cpu()); H.append(h.cpu())
        Z1 = torch.cat(Z1).reshape(-1, Z1[0].shape[-1])
        H = torch.cat(H).reshape(-1, H[0].shape[-1])
        Y = torch.from_numpy(states.reshape(-1, states.shape[-1]))
        X = torch.cat([Z1, H], dim=1)
        n_tr = int(0.8 * len(X))
        X, Y = X.to(device), Y.to(device)
        Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
        Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
        reg = 1e-3 * torch.eye(Xtr.shape[1], device=device)
        w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
        P = Xte @ w
        ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
        ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
        import numpy as np
        return (1 - ss_res / ss_tot).cpu().numpy()

    r2_trained = ridge_r2_single(rec_enc, d["obs"], d["states"], device)

    print(f"{'переменная':<10}{'z1':>9}{'+GRU обуч':>11}{'+GRU случ':>11}"
          f"{'+проекция':>11}{'GRU>проекц?':>13}")
    print("-" * 65)
    for name, a, tr, rg, ri in zip(STATE_NAMES, r2_z1, r2_trained, r2_rndgru, r2_rndinst):
        verdict = "да" if (tr - a) > 1.5 * (ri - a) else "неясно"
        print(f"{name:<10}{a:>9.3f}{tr:>11.3f}{rg:>11.3f}{ri:>11.3f}{verdict:>13}")
    print("\n'GRU>проекц?' = да, если прирост от ОБУЧЕННОГО GRU минимум в 1.5")
    print("раза больше прироста от случайной мгновенной проекции той же")
    print("размерности — то есть эффект от рекуррентности, а не просто")
    print("от лишних признаков.")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"gru": rec_enc.gru.state_dict(), "predictor": pred.state_dict(),
                "hidden_dim": args.hidden_dim, "latent_dim": ck["latent_dim"],
                "in_channels": ck.get("in_channels", 1),
                "base_ckpt": args.ckpt}, args.out)
    print(f"\nСохранено в {args.out}")
