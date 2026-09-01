"""Декодер латента в картинку (probe на замороженном энкодере).

Важно: энкодер ЗАМОРОЖЕН. Декодер — это probe, он не влияет на представление,
а лишь позволяет заглянуть внутрь: "как выглядит мир, который модель держит
в латенте". JEPA специально обучалась без пиксельного лосса, поэтому декодер
здесь — диагностический инструмент, а не часть модели.

Запуск:  python train_decoder.py --ckpt checkpoints/jepa.pt --epochs 15
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import Encoder


class Decoder(nn.Module):
    """Зеркало энкодера: латент -> (1, 64, 64)."""
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 32 -> 64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 128, 4, 4)
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/transitions.npz")
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=str, default="checkpoints/decoder.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval()
    enc.load_state_dict(ck["encoder"])
    for prm in enc.parameters():          # заморозка энкодера
        prm.requires_grad_(False)

    d = np.load(args.data)
    obs = torch.from_numpy(d["obs"])
    loader = DataLoader(TensorDataset(obs), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    dec = Decoder(ck["latent_dim"]).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for (o,) in loader:
            o = o.to(device)
            with torch.no_grad():
                z = enc(o)
            recon = dec(z)
            loss = F.mse_loss(recon, o)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"epoch {epoch:3d} | recon MSE {total / len(loader):.5f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"decoder": dec.state_dict(), "latent_dim": ck["latent_dim"],
                "encoder_ckpt": args.ckpt}, args.out)
    print(f"Декодер сохранён в {args.out}")


if __name__ == "__main__":
    main()
