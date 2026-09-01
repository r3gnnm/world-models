"""Ансамбль предикторов для оценки неопределённости динамики.

Рецепт:
  1. Берём обученный чекпоинт, ЗАМОРАЖИВАЕМ энкодер.
  2. Предвычисляем латенты z_t, z_{t+1} для всего датасета (один проход).
  3. Обучаем K предикторов с разными сидами на бутстрап-подвыборках.
     Коллапс невозможен по построению: цели (латенты) фиксированы,
     поэтому достаточно обычного MSE, VICReg не нужен.

Разброс предсказаний членов ансамбля = эпистемическая неопределённость
модели динамики: "мы по-разному выучились там, где данных мало или
динамика сложна".

Запуск:  python train_ensemble.py --ckpt checkpoints/jepa.pt --k 5
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from models import Encoder, Predictor


@torch.no_grad()
def precompute_latents(enc, obs, next_obs, device, bs=512):
    zs, z1s = [], []
    for i in range(0, len(obs), bs):
        zs.append(enc(obs[i:i + bs].to(device)).cpu())
        z1s.append(enc(next_obs[i:i + bs].to(device)).cpu())
    return torch.cat(zs), torch.cat(z1s)


def train_one_predictor(z, a, z1, latent_dim, seed, epochs, device, bs=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(z), size=len(z))          # бутстрап-подвыборка
    zb, ab, z1b = z[boot].to(device), a[boot].to(device), z1[boot].to(device)

    pred = Predictor(latent_dim).to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(epochs):
        perm = torch.randperm(len(zb), device=device)
        for i in range(0, len(zb), bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(pred(zb[idx], ab[idx]), z1b[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return pred, loss.item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/transitions.npz")
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa.pt")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", type=str, default="checkpoints/ensemble.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ckpt["latent_dim"]).to(device).eval()
    enc.load_state_dict(ckpt["encoder"])

    d = np.load(args.data)
    obs = torch.from_numpy(d["obs"])
    a = torch.from_numpy(d["actions"])
    next_obs = torch.from_numpy(d["next_obs"])

    print("предвычисляю латенты (энкодер заморожен)...")
    z, z1 = precompute_latents(enc, obs, next_obs, device)

    members = []
    for k in range(args.k):
        pred, final = train_one_predictor(z, a, z1, ckpt["latent_dim"],
                                          seed=1000 + k, epochs=args.epochs,
                                          device=device)
        members.append(pred.state_dict())
        print(f"member {k}: final MSE {final:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"members": members, "latent_dim": ckpt["latent_dim"],
                "encoder_ckpt": args.ckpt}, args.out)
    print(f"Ансамбль из {args.k} предикторов сохранён в {args.out}")
