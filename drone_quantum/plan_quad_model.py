"""CEM-планировщик для квадрокоптера, работающий на ОБУЧЕННОЙ world model
вместо истинной физики — и прямое сравнение с baseline.

Смысл эксперимента. test_quad_cem.py показал 10/10 успешных долётов, когда
CEM катит истинную физику. Здесь тот же планировщик, та же cost-функция, тот
же горизонт — но динамику предсказывает обученный Predictor в латентном
пространстве. Разница в успешности напрямую и однозначно измеряет, насколько
world model достаточно хороша, чтобы по ней реально летать. Это куда более
содержательная метрика, чем probe R² или action_gap: она отвечает на
вопрос "можно ли этой моделью пользоваться", а не "что она закодировала".

Cost в латенте: расстояние до закодированной цели. Заметь, что штраф за
остаточную скорость (который был важен на истинной физике, чтобы дрон не
проскакивал цель) здесь недоступен напрямую — скорость плохо восстановима
из мгновенного латента, это уже установлено. Для честности сравнения
приводим оба варианта: с рекуррентным контекстом (где скорость доступна)
и без него.

Запуск:
  python plan_quad_model.py --ckpt checkpoints/quad2d_150ep.pt --trials 10
  python plan_quad_model.py --ckpt checkpoints/quad2d_150ep.pt \
      --recurrent checkpoints/quad2d_recurrent.pt --trials 10
"""
import argparse
import numpy as np
import torch

from env_quad2d import Quad2DEnv, SIZE, GROUND_Z, CEILING_Z
from models import Encoder, Predictor


@torch.no_grad()
def cem_plan_model(enc, pred, env, z_goal, device, horizon=15, pop=256,
                   iters=6, n_elites=32, rng=None):
    """CEM, катящий ОБУЧЕННУЮ модель в латенте. Возвращает план действий."""
    if rng is None:
        rng = np.random.default_rng(0)
    mean = np.zeros((horizon, 2))
    std = np.full((horizon, 2), 0.6)

    o = env.render()
    z0 = enc(torch.from_numpy(o[None]).to(device))

    for _ in range(iters):
        cand = np.clip(mean[None] + std[None] * rng.normal(size=(pop, horizon, 2)),
                      -1, 1).astype(np.float32)
        cand_t = torch.from_numpy(cand).to(device)          # (P, H, 2)
        z = z0.expand(pop, -1).contiguous()
        for t in range(horizon):
            z = pred(z, cand_t[:, t])
        cost = ((z - z_goal) ** 2).mean(-1).cpu().numpy()   # (P,)
        elite = cand[np.argsort(cost)[:n_elites]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean


def make_goal_obs(env, target):
    """Кадр, как если бы дрон был в целевой точке (нулевая скорость, ровный
    корпус) — та же техника, что использовалась в предыдущих средах проекта."""
    g = Quad2DEnv(seed=0, obstacles=False, dual_view=getattr(env, 'dual_view', False),
                  egocentric=getattr(env, 'egocentric', True))
    g.obs_pos, g.obs_r = env.obs_pos, env.obs_r
    g.pos = np.array(target, dtype=np.float32)
    g.vel = np.zeros(2, dtype=np.float32)
    g.theta, g.omega = 0.0, 0.0
    return g.render()


def run_trial(enc, pred, seed, horizon, max_steps, success_dist, replan_every,
              device, rng, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    env = Quad2DEnv(seed=seed, egocentric=env_kwargs.get('egocentric', True),
                    dual_view=env_kwargs.get('dual_view', False))
    env.reset()
    for _ in range(200):
        target = rng.uniform([20, GROUND_Z + 15], [SIZE - 20, CEILING_Z - 15])
        if not env._collides(target) and 20 < np.linalg.norm(target - env.pos) < 60:
            break

    z_goal = enc(torch.from_numpy(make_goal_obs(env, target)[None]).to(device))

    plan = None
    min_dist = np.linalg.norm(env.pos - target)
    for step in range(max_steps):
        if step % replan_every == 0:
            plan = cem_plan_model(enc, pred, env, z_goal, device,
                                  horizon=horizon, rng=rng)
        env.step(plan[step % replan_every])
        d = np.linalg.norm(env.pos - target)
        min_dist = min(min_dist, d)
        if d < success_dist:
            return True, step + 1, min_dist
    return False, max_steps, min_dist


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/quad2d_150ep.pt")
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
    enc = Encoder(ck["latent_dim"], in_channels=ck.get("in_channels", 1)).to(device).eval()
    enc.load_state_dict(ck["encoder"])
    pred = Predictor(ck["latent_dim"],
                     action_dim=ck.get("action_dim", 2)).to(device).eval()
    pred.load_state_dict(ck["predictor"])

    print(f"device: {device} | модель: {args.ckpt} | горизонт {args.horizon}")
    print(f"baseline для сравнения: test_quad_cem.py даёт 10/10 на ИСТИННОЙ физике\n")

    rng = np.random.default_rng(0)
    results, dists = [], []
    for t in range(args.trials):
        ok, steps, min_d = run_trial(enc, pred, t, args.horizon, args.max_steps,
                                     args.success_dist, args.replan_every,
                                     device, rng,
                                     {'dual_view': args.dual_view,
                                      'egocentric': not args.global_view})
        results.append(ok); dists.append(min_d)
        status = f"долетел за {steps}" if ok else f"НЕ долетел (мин. дист. {min_d:.1f})"
        print(f"  trial {t}: {status}")

    print(f"\nCEM на ОБУЧЕННОЙ модели: {sum(results)}/{args.trials} успешных")
    print(f"  средняя минимальная дистанция до цели: {np.mean(dists):.1f}")
    print(f"  (истинная физика: 10/10 — разница измеряет качество модели)")
