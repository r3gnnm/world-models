"""Обучение action-conditioned JEPA.

Запуск:  python train.py --data data/transitions.npz --epochs 20

Что логируется и зачем:
  sim / var / cov  — компоненты VICReg; если var растёт к нулю, а sim падает
                     к нулю подозрительно быстро — начался коллапс.
  z_std            — средний std латентов по батчу; здоровое значение ~1.
  action_gap       — насколько хуже предсказание со случайно перемешанными
                     действиями. Если gap ~ 0, модель игнорирует действие
                     и выучила только пассивную динамику. Должен расти.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import Encoder, Predictor
from losses import vicreg_loss


def make_loader(path: str, batch_size: int):
    d = np.load(path)
    ds = TensorDataset(torch.from_numpy(d["obs"]),
                       torch.from_numpy(d["actions"]),
                       torch.from_numpy(d["next_obs"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/transitions.npz")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--out", type=str, default="checkpoints/jepa.pt")
    p.add_argument("--init", type=str, default=None,
                   help="чекпоинт для продолжения обучения")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    loader = make_loader(args.data, args.batch_size)
    enc = Encoder(args.latent_dim).to(device)
    pred = Predictor(args.latent_dim).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=True)
        enc.load_state_dict(ckpt["encoder"])
        pred.load_state_dict(ckpt["predictor"])
        print(f"продолжаю с {args.init}")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        logs = {"sim": 0.0, "var": 0.0, "cov": 0.0,
                "z_std": 0.0, "action_gap": 0.0}
        for o, a, o1 in loader:
            o, a, o1 = o.to(device), a.to(device), o1.to(device)

            z = enc(o)
            z1 = enc(o1)
            z1_hat = pred(z, a)

            loss, parts = vicreg_loss(z1_hat, z1)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                # ablation: предсказание с перемешанными действиями
                z1_rand = pred(z, a[torch.randperm(len(a))])
                gap = (torch.nn.functional.mse_loss(z1_rand, z1)
                       - torch.nn.functional.mse_loss(z1_hat, z1)).item()
                logs["action_gap"] += gap
                logs["z_std"] += z.std(dim=0).mean().item()
            for k in ("sim", "var", "cov"):
                logs[k] += parts[k]

        n = len(loader)
        print(f"epoch {epoch:3d} | sim {logs['sim']/n:.4f} "
              f"| var {logs['var']/n:.4f} | cov {logs['cov']/n:.4f} "
              f"| z_std {logs['z_std']/n:.3f} "
              f"| action_gap {logs['action_gap']/n:.4f}", flush=True)

        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"encoder": enc.state_dict(),
                    "predictor": pred.state_dict(),
                    "latent_dim": args.latent_dim}, args.out)
    print(f"Сохранено в {args.out}")


if __name__ == "__main__":
    main()
