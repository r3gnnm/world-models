"""Жадный планировщик с коротким горизонтом (обход ловушки L2).

Проблема, которую решаем: при длинном горизонте и L2-расстоянии в латенте
"стоять на месте" часто выгоднее правильного движения, потому что промежуточные
состояния кодируются далеко от цели (кривизна латентного многообразия).

Решение здесь простое и прагматичное: оптимизируем расстояние через 1-2 шага,
а не через 15, и перепланируем на каждом шаге (жадный MPC). Короткий горизонт
не даёт кривизне накопиться. Это НЕ полное решение (см. train_distance.py для
правильной temporal-distance метрики), но обычно этого хватает, чтобы агент
поехал — и чтобы отделить "планировщик сломан" от "L2 плохая метрика".

Запуск:  python plan_greedy.py --ckpt checkpoints/jepa_step8.pt --start 14 20 --goal 45 20
"""
import argparse
import numpy as np
import torch

from env import TwoRoomsEnv, SIZE
from models import Encoder, Predictor


@torch.no_grad()
def greedy_action(pred, z, z_goal, horizon=2, pop=512, iters=4,
                  n_elites=64, device="cpu"):
    """CEM на коротком горизонте. Возвращает первое действие лучшего плана."""
    mean = torch.zeros(horizon, 2, device=device)
    std = torch.full((horizon, 2), 0.8, device=device)
    for _ in range(iters):
        acts = (mean + std * torch.randn(pop, horizon, 2, device=device)).clamp(-1, 1)
        zc = z.expand(pop, -1)
        for t in range(horizon):
            zc = pred(zc, acts[:, t])
        cost = ((zc - z_goal) ** 2).mean(-1)                 # расстояние в конце
        elite = acts[cost.topk(n_elites, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean[0]


@torch.no_grad()
def run(enc, pred, start, goal, device, max_steps=80, success_dist=4.0):
    env = TwoRoomsEnv(seed=0); env.reset()
    env.pos = np.array(start, dtype=np.float32)
    g = TwoRoomsEnv(); g.pos = np.array(goal, dtype=np.float32)
    z_goal = enc(torch.from_numpy(g.render()[None]).to(device))

    traj = [env.pos.copy()]
    for step in range(max_steps):
        z = enc(torch.from_numpy(env.render()[None]).to(device))
        a = greedy_action(pred, z, z_goal, device=device)
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


def plot(traj, start, goal, out="greedy_trajectory.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(TwoRoomsEnv().walls, cmap="gray_r", origin="upper")
    ax.plot(traj[:, 0], traj[:, 1], "-o", color="tab:blue", ms=2, lw=1.2)
    ax.plot(*start, "s", color="tab:green", ms=10, label="старт")
    ax.plot(*goal, "*", color="tab:red", ms=16, label="цель")
    ax.legend(fontsize=8); ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
    ax.set_title("Жадный планировщик в латенте")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa_step8.pt")
    p.add_argument("--start", type=float, nargs=2, default=[14, 20])
    p.add_argument("--goal", type=float, nargs=2, default=[45, 20])
    p.add_argument("--horizon", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval(); enc.load_state_dict(ck["encoder"])
    pred = Predictor(ck["latent_dim"]).to(device).eval(); pred.load_state_dict(ck["predictor"])

    traj, ok = run(enc, pred, args.start, args.goal, device)
    plot(traj, args.start, args.goal)
