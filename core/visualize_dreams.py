"""Визуализация "снов" модели: воображаемый rollout против реальности.

Кодируем стартовый кадр, дальше катим ТОЛЬКО предиктор по действиям
(модель воображает, не видя настоящих кадров) и декодируем каждый латент.
Рядом — реальные кадры из среды при тех же действиях. Видно, как воображение
расходится с реальностью по мере накопления ошибки.

Запуск:  python visualize_dreams.py --ckpt checkpoints/jepa.pt \
                 --decoder checkpoints/decoder.pt --steps 20
Выход:   dreams.png (сетка кадров) + dreams.gif (анимация)
"""
import argparse
import numpy as np
import torch

from env import TwoRoomsEnv
from models import Encoder, Predictor
from train_decoder import Decoder


@torch.no_grad()
def rollout_dream(enc, pred, dec, env, actions, device):
    """Возвращает (воображаемые кадры, реальные кадры) по списку действий."""
    o = env.render()
    z = enc(torch.from_numpy(o[None]).to(device))
    imagined, real = [dec(z).cpu().numpy()[0, 0]], [o[0]]
    for a in actions:
        z = pred(z, torch.from_numpy(a[None]).to(device))     # воображение
        o = env.step(a)                                        # реальность
        imagined.append(dec(z).cpu().numpy()[0, 0])
        real.append(o[0])
    return np.array(imagined), np.array(real)


def make_grid(imagined, real, out="dreams.png", every=2):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idxs = list(range(0, len(imagined), every))
    fig, axes = plt.subplots(2, len(idxs), figsize=(1.5 * len(idxs), 3.2))
    for col, t in enumerate(idxs):
        axes[0, col].imshow(real[t], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"t={t}", fontsize=8)
        axes[1, col].imshow(imagined[t], cmap="gray", vmin=0, vmax=1)
        for row in (0, 1):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("реальность", fontsize=9)
    axes[1, 0].set_ylabel("воображение", fontsize=9)
    fig.suptitle("Сны модели: воображаемый rollout vs реальность", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Сохранено: {out}")


def make_gif(imagined, real, out="dreams.gif"):
    try:
        import imageio
    except ImportError:
        print("imageio не установлен — пропускаю gif (pip install imageio)")
        return
    frames = []
    for t in range(len(imagined)):
        pair = np.concatenate([real[t], np.ones((64, 2)), imagined[t]], axis=1)
        frames.append((pair * 255).astype(np.uint8))
    imageio.mimsave(out, frames, fps=5)
    print(f"Сохранено: {out} (слева реальность, справа воображение)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa.pt")
    p.add_argument("--decoder", type=str, default="checkpoints/decoder.pt")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval(); enc.load_state_dict(ck["encoder"])
    pred = Predictor(ck["latent_dim"]).to(device).eval(); pred.load_state_dict(ck["predictor"])
    dck = torch.load(args.decoder, map_location=device, weights_only=True)
    dec = Decoder(dck["latent_dim"]).to(device).eval(); dec.load_state_dict(dck["decoder"])

    rng = np.random.default_rng(args.seed)
    env = TwoRoomsEnv(seed=args.seed); env.reset()
    a = rng.uniform(-1, 1, size=2)
    actions = []
    for _ in range(args.steps):
        a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1).astype(np.float32)
        actions.append(a.copy())

    imagined, real = rollout_dream(enc, pred, dec, env, actions, device)
    make_grid(imagined, real)
    make_gif(imagined, real)
