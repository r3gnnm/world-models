"""Планировщик на обученной temporal-distance метрике (правильное решение).

Отличие от plan_greedy.py: cost = dist_net(z, z_goal) вместо ((z-z_goal)**2).
Обученная метрика соответствует ЧИСЛУ ШАГОВ между состояниями, а не евклидову
расстоянию в искривлённом латенте — поэтому у неё нет ложных локальных минимумов,
в которых застревает L2-планировщик.

Предпосылка: обучены и модель (train.py), и метрика (train_distance.py).

Запуск:  python plan_distance.py --ckpt checkpoints/jepa.pt \
                 --dist checkpoints/distance.pt --start 14 20 --goal 24 45
"""
import argparse
import numpy as np
import torch

from env import TwoRoomsEnv, SIZE
from models import Encoder, Predictor
from train_distance import DistanceNet


@torch.no_grad()
def plan_action(pred, dist_net, z, z_goal, horizon=5, pop=512, iters=5,
                n_elites=64, device="cpu"):
    """CEM с обученной метрикой в качестве cost. Возвращает первое действие."""
    mean = torch.zeros(horizon, 2, device=device)
    std = torch.full((horizon, 2), 0.8, device=device)
    zg = z_goal.expand(pop, -1)
    for _ in range(iters):
        acts = (mean + std * torch.randn(pop, horizon, 2, device=device)).clamp(-1, 1)
        zc = z.expand(pop, -1)
        cost = torch.zeros(pop, device=device)
        for t in range(horizon):
            zc = pred(zc, acts[:, t])
            cost = cost + 0.2 * dist_net(zc, zg)      # шейпинг по обученной метрике
        cost = cost + dist_net(zc, zg)                # финальное расстояние
        elite = acts[cost.topk(n_elites, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean[0]


@torch.no_grad()
def run(enc, pred, dist_net, start, goal, device, max_steps=100, success_dist=4.0):
    env = TwoRoomsEnv(seed=0); env.reset()
    env.pos = np.array(start, dtype=np.float32)
    g = TwoRoomsEnv(); g.pos = np.array(goal, dtype=np.float32)
    z_goal = enc(torch.from_numpy(g.render()[None]).to(device))

    traj = [env.pos.copy()]
    for step in range(max_steps):
        z = enc(torch.from_numpy(env.render()[None]).to(device))
        a = plan_action(pred, dist_net, z, z_goal, device=device)
        env.step(a.cpu().numpy())
        traj.append(env.pos.copy())
        dist = float(np.linalg.norm(env.pos - goal))
        if step % 10 == 0 or dist < success_dist:
            print(f"step {step:3d} | позиция {env.pos.round(1)} | до цели {dist:.1f}")
        if dist < success_dist:
            print(f"ЦЕЛЬ ДОСТИГНУТА за {step + 1} шагов")
            return np.array(traj), True
    print("Лимит шагов исчерпан")
    return np.array(traj), False


def plot(traj, start, goal, out="distance_trajectory.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(TwoRoomsEnv().walls, cmap="gray_r", origin="upper")
    ax.plot(traj[:, 0], traj[:, 1], "-o", color="tab:blue", ms=2, lw=1.2)
    ax.plot(*start, "s", color="tab:green", ms=10, label="старт")
    ax.plot(*goal, "*", color="tab:red", ms=16, label="цель")
    ax.legend(fontsize=8); ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
    ax.set_title("Планировщик на temporal-distance метрике")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa.pt")
    p.add_argument("--dist", type=str, default="checkpoints/distance.pt")
    p.add_argument("--start", type=float, nargs=2, default=[14, 20])
    p.add_argument("--goal", type=float, nargs=2, default=[24, 45])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval(); enc.load_state_dict(ck["encoder"])
    pred = Predictor(ck["latent_dim"]).to(device).eval(); pred.load_state_dict(ck["predictor"])

    dck = torch.load(args.dist, map_location=device, weights_only=True)
    dist_net = DistanceNet(dck["latent_dim"]).to(device).eval()
    dist_net.load_state_dict(dck["dist"])

    traj, ok = run(enc, pred, dist_net, args.start, args.goal, device)
    plot(traj, args.start, args.goal)
