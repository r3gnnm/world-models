"""Замкнутый цикл автономной разведки: собирает воедино всё, что уже
провалидировано по отдельности.

Каждый раунд:
  1. Сэмплируем K кандидатных точек, не пересекающих препятствия и не
     дублирующих уже исследованное (минимальная дистанция до visited).
  2. Оцениваем неопределённость каждой через ансамбль (rollout в воображении).
  3. QUBO выбирает M точек, избегая избыточных пар (уже провалидировано:
     0 избыточных пар на 10/10 seed, -2.6% цены разведки).
  4. Дрон летит к каждой выбранной точке по очереди (ближайшая сначала),
     навигация — короткогоризонтный CEM в латентном пространстве (та же
     идея, что в plan_greedy.py, самодостаточная реализация здесь).
  5. Точки помечаются как исследованные, раунд повторяется с учётом этого.

Запуск:  python closed_loop_exploration.py --rounds 5 --waypoints-per-round 3
"""
import argparse
import numpy as np
import torch

from env_openworld import OpenWorldEnv, SIZE
from models import Encoder, Predictor
from frontier_qubo import candidate_uncertainty, build_qubo, solve
from train_distance import DistanceNet


@torch.no_grad()
def greedy_nav_step(pred, dist_net, z, z_goal, device, horizon=10, pop=384,
                    iters=6, n_elites=48):
    """Короткогоризонтный CEM на ОБУЧЕННОЙ temporal-distance метрике вместо
    сырого L2 (см. диагностику: L2 даёт слабый/обманчивый градиент при
    наличии препятствий между дроном и целью — та же ловушка, что и в
    первой версии планировщика для room-среды)."""
    mean = torch.zeros(horizon, 2, device=device)
    std = torch.full((horizon, 2), 0.8, device=device)
    zg = z_goal.expand(pop, -1)
    for _ in range(iters):
        acts = (mean + std * torch.randn(pop, horizon, 2, device=device)).clamp(-1, 1)
        zc = z.expand(pop, -1)
        cost = torch.zeros(pop, device=device)
        for t in range(horizon):
            zc = pred(zc, acts[:, t])
            cost = cost + 0.2 * dist_net(zc, zg)
        cost = cost + dist_net(zc, zg)
        elite = acts[cost.topk(n_elites, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean[0]


def make_subgoals(env, start, target, max_leg=25.0):
    """Разбивает дальний перелёт на цепочку промежуточных точек не дальше
    max_leg друг от друга — короткий горизонт CEM не может 'разглядеть'
    обход крупных препятствий на дистанции 80-100 юнитов (см. диагностику:
    неудачи дают ОГРОМНЫЙ остаток, а не 'почти долетел' — типичный
    признак застревания сразу за первым препятствием на пути). Цепочка
    коротких отрезков — стандартный приём (глобальный грубый маршрут +
    локальный исполнитель), прямая практическая параллель с иерархией."""
    start, target = np.asarray(start), np.asarray(target)
    dist = np.linalg.norm(target - start)
    n = max(1, int(np.ceil(dist / max_leg)))
    pts = [start + (target - start) * (i / n) for i in range(1, n + 1)]
    # если промежуточная точка попала в препятствие — небольшой перпендикулярный сдвиг
    fixed = []
    perp = np.array([-(target - start)[1], (target - start)[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-8)
    for p in pts:
        q = p.copy()
        for sign in (0, 1, -1, 2, -2):
            cand = p + perp * sign * 8.0
            if not env._collides(cand):
                q = cand
                break
        fixed.append(q)
    return fixed


def navigate_chained(enc, pred, dist_net, env, target_xy, device,
                     max_leg=25.0, per_leg_steps=40):
    """Ведёт агента к дальней цели через цепочку промежуточных точек —
    каждый отрезок решается отдельным, компактным navigate_to."""
    subgoals = make_subgoals(env, env.pos, target_xy, max_leg)
    full_traj = [env.pos.copy()]
    reached_final = False
    for i, sg in enumerate(subgoals):
        is_last = (i == len(subgoals) - 1)
        traj, reached = navigate_to(enc, pred, dist_net, env, sg, device,
                                    max_steps=per_leg_steps,
                                    success_dist=6.0 if is_last else 10.0)
        full_traj.extend(traj[1:].tolist())
        if is_last:
            reached_final = reached
    return np.array(full_traj), reached_final


@torch.no_grad()
def navigate_to(enc, pred, dist_net, env, target_xy, device, max_steps=60,
                success_dist=6.0):
    """Направляет агента к точке через реальную среду (MPC: перепланирование
    на каждом шаге), используя обученную метрику расстояния как cost.
    Возвращает (траектория, добрался_ли_за_отведённые_шаги)."""
    goal_env = OpenWorldEnv(seed=0)
    goal_env.obs_pos, goal_env.obs_r = env.obs_pos, env.obs_r  # тот же мир
    goal_env.pos = np.array(target_xy, dtype=np.float32)
    z_goal = enc(torch.from_numpy(goal_env.render()[None]).to(device))

    traj = [env.pos.copy()]
    reached = False
    for _ in range(max_steps):
        z = enc(torch.from_numpy(env.render()[None]).to(device))
        a = greedy_nav_step(pred, dist_net, z, z_goal, device)
        env.step(a.cpu().numpy())
        traj.append(env.pos.copy())
        if np.linalg.norm(env.pos - np.array(target_xy)) < success_dist:
            reached = True
            break
    return np.array(traj), reached


def sample_candidates(env, visited, n_candidates, min_dist_visited, rng,
                      max_travel_dist=55.0):
    """max_travel_dist: не рассматриваем точки дальше разумного вылета за
    раз — реалистичное ограничение дальности для дрона, и оно же убирает
    сверхдальние перелёты сквозь плотные кластеры препятствий, которые
    короткогоризонтная навигация не может надёжно преодолеть за один присест
    (см. диагностику: застревание именно на перелётах 60-100+ юнитов через
    плотные скопления препятствий, а не на разумных дистанциях)."""
    cand = []
    attempts = 0
    while len(cand) < n_candidates and attempts < n_candidates * 40:
        attempts += 1
        p_ = rng.uniform(12, SIZE - 12, size=2)
        if env._collides(p_):
            continue
        if np.linalg.norm(p_ - env.pos) > max_travel_dist:
            continue
        if any(np.linalg.norm(p_ - v) < min_dist_visited for v in visited):
            continue
        cand.append(p_)
    return np.array(cand) if cand else np.zeros((0, 2))


def run_exploration(enc, pred, dist_net, members, env, device, rounds,
                    n_candidates, m_select, min_dist_visited=15.0,
                    max_travel_dist=55.0, seed=0):
    rng = np.random.default_rng(seed)
    visited = [env.pos.copy()]
    full_traj = [env.pos.copy()]
    round_log = []

    for r in range(rounds):
        cand = sample_candidates(env, visited, n_candidates, min_dist_visited,
                                 rng, max_travel_dist=max_travel_dist)
        if len(cand) < m_select:
            print(f"раунд {r+1}: недостаточно новых кандидатов, останавливаюсь")
            break

        value, risk = candidate_uncertainty(enc, members, env, cand, device,
                                            return_path_risk=True)
        bqm = build_qubo(cand, value, M=min(m_select, len(cand)),
                         current_pos=env.pos, risk=risk,
                         max_travel_dist=max_travel_dist)
        mask = solve(bqm)
        waypoints = cand[mask]

        order = []
        remaining = list(range(len(waypoints)))
        cur = env.pos.copy()
        while remaining:
            d = [np.linalg.norm(waypoints[i] - cur) for i in remaining]
            nxt = remaining[int(np.argmin(d))]
            order.append(nxt); remaining.remove(nxt); cur = waypoints[nxt]

        print(f"раунд {r+1}: выбрано {len(waypoints)} точек, "
              f"ценность={value[mask].mean():.3f}, риск={risk[mask].mean():.3f}, "
              f"макс. дистанция маршрута={max((np.linalg.norm(waypoints[i]-waypoints[j]) for i in range(len(waypoints)) for j in range(i+1,len(waypoints))), default=0):.1f}")

        for idx in order:
            wp = waypoints[idx]
            traj, reached = navigate_chained(enc, pred, dist_net, env, wp, device)
            full_traj.extend(traj[1:].tolist())
            visited.append(env.pos.copy())
            dist_left = np.linalg.norm(env.pos - wp)
            status = "долетел" if reached else f"НЕ долетел (осталось {dist_left:.1f})"
            print(f"    -> точка {wp.round(1)}: {status} за {len(traj)-1} шагов")

        round_log.append({"round": r + 1, "waypoints": waypoints.copy(),
                          "uncertainty": value[mask].copy()})

    return np.array(full_traj), visited, round_log


def plot_exploration(env, full_traj, visited, round_log, out="exploration_map.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    for (ox, oy), r in zip(env.obs_pos, env.obs_r):
        ax.add_patch(plt.Circle((ox, oy), r, color="gray", alpha=0.5))

    ax.plot(full_traj[:, 0], full_traj[:, 1], "-", color="tab:blue", lw=1.2,
           alpha=0.8, label="траектория дрона")
    cmap = plt.cm.plasma
    for i, rl in enumerate(round_log):
        c = cmap(i / max(len(round_log) - 1, 1))
        ax.scatter(rl["waypoints"][:, 0], rl["waypoints"][:, 1],
                  s=90, color=c, edgecolors="k", zorder=5,
                  label=f"раунд {rl['round']}")
    start = visited[0]
    ax.plot(*start, "s", color="lime", ms=14, mec="k", zorder=6, label="старт")
    ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
    ax.set_title("Автономная разведка: JEPA + ансамбль неопределённости + QUBO")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/openworld_jepa.pt")
    p.add_argument("--ensemble", type=str, default="checkpoints/openworld_ensemble.pt")
    p.add_argument("--dist", type=str, default="checkpoints/openworld_distance.pt")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--n-candidates", type=int, default=20)
    p.add_argument("--waypoints-per-round", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-travel-dist", type=float, default=55.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval(); enc.load_state_dict(ck["encoder"])
    pred = Predictor(ck["latent_dim"]).to(device).eval(); pred.load_state_dict(ck["predictor"])
    dck = torch.load(args.dist, map_location=device, weights_only=True)
    dist_net = DistanceNet(dck["latent_dim"]).to(device).eval()
    dist_net.load_state_dict(dck["dist"])
    ens = torch.load(args.ensemble, map_location=device, weights_only=True)
    members = []
    for sd in ens["members"]:
        m = Predictor(ens["latent_dim"]).to(device).eval()
        m.load_state_dict(sd)
        members.append(m)

    env = OpenWorldEnv(seed=args.seed)
    env.reset()

    traj, visited, round_log = run_exploration(
        enc, pred, dist_net, members, env, device, args.rounds,
        args.n_candidates, args.waypoints_per_round,
        max_travel_dist=args.max_travel_dist, seed=args.seed)

    print(f"\nВсего пройдено точек траектории: {len(traj)}, "
          f"раундов завершено: {len(round_log)}")
    plot_exploration(env, traj, visited, round_log)
