"""Sanity-check ПЕРЕД переносом closed-loop на квадрокоптер: способен ли
CEM вообще довести недоуправляемый дрон до точки, если дать ему ИСТИННУЮ
физику вместо обученной модели?

Зачем этот тест первым. Если CEM не справляется даже с идеальной динамикой
(симулятором), то любые неудачи с обученной world model будут неотличимы
от неудач самой постановки задачи — и мы будем месяц чинить модель там,
где сломан планировщик. Это прямой урок из истории с L2-метрикой ранее в
проекте: сначала изолируй, потом чини.

Ключевые отличия от навигации в предыдущих средах:
  - действие = (тяга, команда наклона), а не (dx, dy) — горизонтальное
    движение возникает ТОЛЬКО через наклон корпуса (недоуправляемость)
  - горизонт CEM должен быть заметно длиннее: измеренный эффект от разных
    действий проявляется на ~15 шагах (1.2 с), а не на 3-5
  - цель — долететь и по возможности стабилизироваться, а не просто
    коснуться точки: у дрона есть инерция, он проскакивает цель

Запуск:  python test_quad_cem.py --trials 10 --horizon 15
"""
import argparse
import numpy as np

from env_quad2d import Quad2DEnv, SIZE, GROUND_Z, CEILING_Z


def rollout_batch_true_physics(state, actions_batch, obs_pos, obs_r):
    """ВЕКТОРИЗОВАННАЯ прокатка истинной физики для всей популяции CEM разом.

    Дублирует формулы из Quad2DEnv.step, но батчем по numpy — цикл по 256
    отдельным экземплярам среды слишком медленный для CEM на CPU
    (замерено: полный прогон не укладывался в разумное время).

    actions_batch: (P, H, 2) -> возвращает (final_pos (P,2), final_vel (P,2),
    crashed (P,) bool)
    """
    from env_quad2d import (DT, GRAVITY, MASS, THRUST_HOVER, THRUST_RANGE,
                            MAX_PITCH_RATE, MAX_PITCH, DRAG, ANG_DRAG, AGENT_R)
    P, H = actions_batch.shape[0], actions_batch.shape[1]
    pos = np.tile(state["pos"], (P, 1)).astype(np.float64)
    vel = np.tile(state["vel"], (P, 1)).astype(np.float64)
    theta = np.full(P, state["theta"], dtype=np.float64)
    omega = np.full(P, state["omega"], dtype=np.float64)
    crashed = np.zeros(P, dtype=bool)

    for t in range(H):
        a = np.clip(actions_batch[:, t], -1, 1)
        thrust = THRUST_HOVER * (1.0 + THRUST_RANGE * a[:, 0])
        pitch_cmd = MAX_PITCH_RATE * a[:, 1]

        omega += (pitch_cmd - ANG_DRAG * omega) * DT
        theta = np.clip(theta + omega * DT, -MAX_PITCH, MAX_PITCH)

        ax = (thrust / MASS) * np.sin(theta) - DRAG * vel[:, 0]
        az = (thrust / MASS) * np.cos(theta) - GRAVITY - DRAG * vel[:, 1]
        vel = vel + np.stack([ax, az], axis=1) * DT
        new_pos = pos + vel * DT * 10.0

        # коллизии: границы + препятствия, батчем
        hit = ((new_pos[:, 1] < GROUND_Z) | (new_pos[:, 1] > CEILING_Z)
               | (new_pos[:, 0] < AGENT_R) | (new_pos[:, 0] > SIZE - AGENT_R))
        if len(obs_pos) > 0:
            d = np.linalg.norm(new_pos[:, None, :] - obs_pos[None, :, :], axis=2)
            hit |= np.any(d < (obs_r + AGENT_R)[None, :], axis=1)

        vel[hit] *= -0.4
        pos[~hit] = new_pos[~hit]
        crashed |= hit

    return pos, vel, crashed


def rollout_true_physics(env_state, actions, obs_pos, obs_r):
    """Одиночная прокатка (для отладки/совместимости)."""
    pos, vel, crashed = rollout_batch_true_physics(
        env_state, actions[None], obs_pos, obs_r)
    return pos[0], vel[0], bool(crashed[0])


def cem_plan_true(env, target, horizon=15, pop=256, iters=6, n_elites=32,
                  vel_penalty=0.3, crash_penalty=50.0, rng=None):
    """CEM на истинной физике. cost = расстояние до цели в конце
    + штраф за остаточную скорость (чтобы не проскакивал) + штраф за столкновение."""
    if rng is None:
        rng = np.random.default_rng(0)
    mean = np.zeros((horizon, 2))
    std = np.full((horizon, 2), 0.6)
    state = {"pos": env.pos.copy(), "vel": env.vel.copy(),
             "theta": env.theta, "omega": env.omega}

    for _ in range(iters):
        cand = np.clip(mean[None] + std[None] * rng.normal(size=(pop, horizon, 2)),
                      -1, 1)
        final_pos, final_vel, crashed = rollout_batch_true_physics(
            state, cand, env.obs_pos, env.obs_r)
        costs = (np.linalg.norm(final_pos - target, axis=1)
                + vel_penalty * np.linalg.norm(final_vel, axis=1)
                + crash_penalty * crashed)
        elite = cand[np.argsort(costs)[:n_elites]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean


def run_trial(seed, horizon, max_steps, success_dist, replan_every, rng):
    env = Quad2DEnv(seed=seed)
    env.reset()
    # цель на разумном расстоянии, не в препятствии
    for _ in range(200):
        target = rng.uniform([20, GROUND_Z + 15], [SIZE - 20, CEILING_Z - 15])
        if not env._collides(target) and 20 < np.linalg.norm(target - env.pos) < 60:
            break

    traj = [env.pos.copy()]
    plan = None
    for step in range(max_steps):
        if step % replan_every == 0:
            plan = cem_plan_true(env, target, horizon=horizon, rng=rng)
        env.step(plan[step % replan_every])
        traj.append(env.pos.copy())
        if np.linalg.norm(env.pos - target) < success_dist:
            return True, step + 1, np.array(traj), target
    return False, max_steps, np.array(traj), target


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--success-dist", type=float, default=8.0)
    p.add_argument("--replan-every", type=int, default=5)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    results = []
    for t in range(args.trials):
        ok, steps, traj, target = run_trial(t, args.horizon, args.max_steps,
                                            args.success_dist, args.replan_every, rng)
        results.append(ok)
        status = f"долетел за {steps}" if ok else f"НЕ долетел ({args.max_steps} шагов)"
        print(f"  trial {t}: {status}")

    print(f"\nCEM на ИСТИННОЙ физике: {sum(results)}/{args.trials} успешных")
    print("Если здесь низкий процент — проблема в постановке задачи "
          "(горизонт, cost, недоуправляемость),")
    print("а НЕ в качестве обученной world model. Чинить надо это, прежде "
          "чем подключать модель.")
