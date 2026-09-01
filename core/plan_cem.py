"""Планирование в латентном пространстве: CEM + MPC.

Идея: цель задаётся КАРТИНКОЙ. Кодируем её в z_goal, затем Cross-Entropy Method
подбирает последовательность действий, минимизирующую латентное расстояние
до цели, прокатывая кандидатов через предиктор ("воображение"). Исполняем
первое действие в реальной среде, перепланируем — классический MPC.

Запуск:  python plan_cem.py --ckpt checkpoints/jepa.pt
Результат: печать прогресса + mpc_trajectory.png с маршрутом агента.
"""
import argparse
import numpy as np
import torch

from env import TwoRoomsEnv, SIZE
from models import Encoder, Predictor


@torch.no_grad()
def cem_plan(pred, z0, z_goal, horizon=15, pop=256, n_elites=32, iters=5,
             device="cpu"):
    """Возвращает лучшую последовательность действий (horizon, 2)."""
    mean = torch.zeros(horizon, 2, device=device)
    std = torch.full((horizon, 2), 0.7, device=device)
    for _ in range(iters):
        acts = (mean + std * torch.randn(pop, horizon, 2, device=device)
                ).clamp(-1, 1)
        z = z0.expand(pop, -1)
        cost = torch.zeros(pop, device=device)
        for t in range(horizon):
            z = pred(z, acts[:, t])
            cost += 0.05 * ((z - z_goal) ** 2).mean(-1)   # промежут. шейпинг
        cost += ((z - z_goal) ** 2).mean(-1)              # финальное расстояние
        elite = acts[cost.topk(n_elites, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean


@torch.no_grad()
def run_mpc(enc, pred, start, goal, device, max_steps=60, success_dist=4.0):
    env = TwoRoomsEnv(seed=0)
    env.reset()
    env.pos = np.array(start, dtype=np.float32)

    goal_env = TwoRoomsEnv()
    goal_env.pos = np.array(goal, dtype=np.float32)
    z_goal = enc(torch.from_numpy(goal_env.render()[None]).to(device))

    traj = [env.pos.copy()]
    for step in range(max_steps):
        z = enc(torch.from_numpy(env.render()[None]).to(device))
        plan = cem_plan(pred, z, z_goal, device=device)
        env.step(plan[0].cpu().numpy())                   # MPC: только 1-е действие
        traj.append(env.pos.copy())
        dist = float(np.linalg.norm(env.pos - goal))
        if step % 10 == 0 or dist < success_dist:
            print(f"step {step:3d} | позиция {env.pos.round(1)} | до цели {dist:.1f}")
        if dist < success_dist:
            print(f"ЦЕЛЬ ДОСТИГНУТА за {step + 1} шагов")
            return np.array(traj), True
    print("Лимит шагов исчерпан")
    return np.array(traj), False


def plot_trajectory(traj, start, goal, walls, out="mpc_trajectory.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(walls, cmap="gray_r", origin="upper")
    ax.plot(traj[:, 0], traj[:, 1], "-o", color="tab:blue", ms=2, lw=1.2,
            label="маршрут")
    ax.plot(*start, "s", color="tab:green", ms=10, label="старт")
    ax.plot(*goal, "*", color="tab:red", ms=16, label="цель")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("MPC в латентном пространстве")
    ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/jepa.pt")
    p.add_argument("--start", type=float, nargs=2, default=[14, 32])
    p.add_argument("--goal", type=float, nargs=2, default=[50, 32])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ckpt["latent_dim"]).to(device).eval()
    pred = Predictor(ckpt["latent_dim"]).to(device).eval()
    enc.load_state_dict(ckpt["encoder"])
    pred.load_state_dict(ckpt["predictor"])

    traj, ok = run_mpc(enc, pred, args.start, args.goal, device)
    plot_trajectory(traj, args.start, args.goal, TwoRoomsEnv().walls)
