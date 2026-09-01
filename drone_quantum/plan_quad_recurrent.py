"""CEM-планировщик, использующий РЕКУРРЕНТНЫЙ контекст — закрывает последний
измеренный информационный дефицит.

Логика всей серии экспериментов, приведшая сюда:
  эгоцентрический вид  -> наклон есть (0.92), позиции нет (0.53)  -> 0/10
  глобальный, разные карты -> shortcut на статичный фон           -> 0/10
  глобальный, одна карта   -> позиция есть (0.99), наклона нет    -> 1/10
  два канала               -> позиция И наклон есть (0.98/0.94)   -> 2/10
                              но СКОРОСТЬ всё ещё слаба (vx 0.28)

Скорость критична для недоуправляемой системы с инерцией: чтобы прилететь
в точку, надо затормозить. В baseline на истинной физике cost содержал
явный штраф за остаточную скорость (vel_penalty=0.3) — без него дрон
проскакивал цель. В латентной версии такого члена не было, потому что
скорости в латенте почти нет.

Отдельный эксперимент показал, что рекуррентность это чинит: GRU поверх
замороженного энкодера поднимал probe по vx с 0.164 до 0.680, причём
контроль случайной МГНОВЕННОЙ проекцией дал ровно нулевой прирост — то
есть эффект именно от интегрирования истории, а не от лишней размерности.

Здесь мы соединяем оба: двухканальный вход (позиция + наклон) и
рекуррентный контекст h (скорость). Планировщик катит ContextPredictor,
принимающий (z1, h, a), и h обновляется вдоль воображаемой траектории.

Запуск:
  python plan_quad_recurrent.py --ckpt checkpoints/quad2d_dual.pt \
      --recurrent checkpoints/quad2d_dual_rec.pt --dual-view --trials 10
"""
import argparse
import numpy as np
import torch

from env_quad2d import Quad2DEnv, SIZE, GROUND_Z, CEILING_Z
from models import Encoder
from models_quad_recurrent import RecurrentQuadEncoder
from train_quad_recurrent import ContextPredictor


@torch.no_grad()
def cem_plan_recurrent(rec_enc, ctx_pred, env, z_goal, h_now, device,
                       horizon=15, pop=256, iters=6, n_elites=32, rng=None):
    """CEM, катящий рекуррентный предиктор. h обновляется вдоль воображаемой
    траектории — так планировщик учитывает накопленную динамику (скорость),
    а не только мгновенный кадр."""
    if rng is None:
        rng = np.random.default_rng(0)
    mean = np.zeros((horizon, 2))
    std = np.full((horizon, 2), 0.6)

    o = env.render()
    z0 = rec_enc.encode_frame(torch.from_numpy(o[None]).to(device))

    for _ in range(iters):
        cand = np.clip(mean[None] + std[None] * rng.normal(size=(pop, horizon, 2)),
                      -1, 1).astype(np.float32)
        cand_t = torch.from_numpy(cand).to(device)
        z = z0.expand(pop, -1).contiguous()
        h = h_now.expand(pop, -1).contiguous()
        for t in range(horizon):
            h = rec_enc.step(z, h)
            z = ctx_pred(z, h, cand_t[:, t])
        cost = ((z - z_goal) ** 2).mean(-1).cpu().numpy()
        elite = cand[np.argsort(cost)[:n_elites]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean


def make_goal_obs(env, target):
    g = Quad2DEnv(seed=0, obstacles=False,
                  dual_view=getattr(env, "dual_view", False),
                  egocentric=getattr(env, "egocentric", True))
    g.obs_pos, g.obs_r = env.obs_pos, env.obs_r
    g.pos = np.array(target, dtype=np.float32)
    g.vel = np.zeros(2, dtype=np.float32)
    g.theta, g.omega = 0.0, 0.0
    return g.render()


@torch.no_grad()
def run_trial(rec_enc, ctx_pred, seed, horizon, max_steps, success_dist,
              replan_every, device, rng, env_kwargs):
    env = Quad2DEnv(seed=seed, **env_kwargs)
    env.reset()
    for _ in range(200):
        target = rng.uniform([20, GROUND_Z + 15], [SIZE - 20, CEILING_Z - 15])
        if not env._collides(target) and 20 < np.linalg.norm(target - env.pos) < 60:
            break

    z_goal = rec_enc.encode_frame(
        torch.from_numpy(make_goal_obs(env, target)[None]).to(device))

    # h накапливается по РЕАЛЬНОЙ траектории полёта, отдельно от воображаемой
    h = rec_enc.init_hidden(1, device)
    plan = None
    min_dist = np.linalg.norm(env.pos - target)
    for step in range(max_steps):
        z_now = rec_enc.encode_frame(
            torch.from_numpy(env.render()[None]).to(device))
        h = rec_enc.step(z_now, h)
        if step % replan_every == 0:
            plan = cem_plan_recurrent(rec_enc, ctx_pred, env, z_goal, h,
                                      device, horizon=horizon, rng=rng)
        env.step(plan[step % replan_every])
        d = np.linalg.norm(env.pos - target)
        min_dist = min(min_dist, d)
        if d < success_dist:
            return True, step + 1, min_dist
    return False, max_steps, min_dist


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/quad2d_dual.pt")
    p.add_argument("--recurrent", type=str, default="checkpoints/quad2d_dual_rec.pt")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--success-dist", type=float, default=8.0)
    p.add_argument("--replan-every", type=int, default=5)
    p.add_argument("--dual-view", action="store_true")
    p.add_argument("--global-view", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    in_ch = ck.get("in_channels", 1)
    base_enc = Encoder(ck["latent_dim"], in_channels=in_ch).to(device).eval()
    base_enc.load_state_dict(ck["encoder"])

    rck = torch.load(args.recurrent, map_location=device, weights_only=True)
    rec_enc = RecurrentQuadEncoder(base_enc, ck["latent_dim"],
                                   rck["hidden_dim"]).to(device).eval()
    rec_enc.gru.load_state_dict(rck["gru"])
    ctx_pred = ContextPredictor(ck["latent_dim"], rck["hidden_dim"]).to(device).eval()
    ctx_pred.load_state_dict(rck["predictor"])

    print(f"device: {device} | база: {args.ckpt} ({in_ch} кан.) | "
          f"рекуррентный: {args.recurrent}")
    print("baseline: истинная физика 10/10 | двухканальный без памяти 2/10\n")

    env_kwargs = {"dual_view": args.dual_view, "egocentric": not args.global_view}
    rng = np.random.default_rng(0)
    results, dists = [], []
    for t in range(args.trials):
        ok, steps, min_d = run_trial(rec_enc, ctx_pred, t, args.horizon,
                                     args.max_steps, args.success_dist,
                                     args.replan_every, device, rng, env_kwargs)
        results.append(ok); dists.append(min_d)
        status = f"долетел за {steps}" if ok else f"НЕ долетел (мин. дист. {min_d:.1f})"
        print(f"  trial {t}: {status}")

    print(f"\nCEM с рекуррентным контекстом: {sum(results)}/{args.trials} успешных")
    print(f"  средняя минимальная дистанция: {np.mean(dists):.1f} "
          f"(без памяти было 20.6)")
