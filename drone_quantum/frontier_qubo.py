"""Frontier selection как QUBO: какие точки разведки выбрать дроном.

Задача: из K кандидатных точек вокруг агента выбрать M самых ценных для
разведки. "Ценность" точки — предсказанная неопределённость модели на пути
к ней (чем больше расходятся члены ансамбля, тем меньше модель знает про
это направление, тем ценнее там реально пролететь). Наивный жадный выбор
(взять top-M по ценности) не учитывает ИЗБЫТОЧНОСТЬ: если две ценные точки
рядом, разведка одной почти наверняка даёт всю ту же информацию, что и
разведка соседней — жадный алгоритм всё равно выберет обе, а QUBO способен
учесть это взаимодействие между кандидатами и предпочесть более
рассредоточенный набор.

QUBO-формулировка (бинарные переменные x_i = "выбрать точку i"):

  H(x) = -sum_i u_i * x_i                         # максимизируем ценность
         + P * (sum_i x_i - M)^2                  # ровно M точек (мягкое ограничение)
         + R * sum_{i<j, near(i,j)} x_i * x_j      # штраф за близкие пары (избыточность)

Решаем классическим симулированным отжигом (dwave-neal) — тем же
интерфейсом (dimod.BinaryQuadraticModel), которым решают и на настоящем
квантовом отжигающем устройстве (D-Wave). Смена SimulatedAnnealingSampler
на DWaveSampler — единственное, что меняется при переходе на реальное
железо через корейских партнёров.

Запуск:  python frontier_qubo.py --ckpt checkpoints/openworld_jepa.pt \
                 --ensemble checkpoints/openworld_ensemble.pt
"""
import argparse
import numpy as np
import torch
import dimod
from neal import SimulatedAnnealingSampler

from env_openworld import OpenWorldEnv, SIZE
from models import Encoder, Predictor


@torch.no_grad()
def candidate_uncertainty(enc, members, env, cand_points, device, horizon_cap=8,
                          return_path_risk=False):
    """Для каждой кандидатной точки: катим ансамбль предикторов по прямой
    от текущей позиции агента и меряем разброс предсказаний в конце —
    та же методология, что в eval_uncertainty.rollout_analysis (r≈0.98
    корреляция с реальной ошибкой на building-среде).

    Если return_path_risk=True — дополнительно возвращает risk: средний
    разброс ансамбля НА ВСЁМ ПУТИ (не только в конце). Это отдельное от
    "ценности" понятие: value = сколько нового мы узнаем, долетев туда;
    risk = насколько модель вообще уверена, что она правильно предсказывает
    происходящее по дороге. Дальняя точка может быть очень информативной
    (высокий value) и при этом опасной, если путь к ней лежит через зону,
    где модели сильно расходятся (высокий risk) — это разные критерии, и
    полноценный QUBO должен учитывать оба, а не только value."""
    o = env.render()
    z0 = enc(torch.from_numpy(o[None]).to(device))
    values, risks = [], []
    for cx, cy in cand_points:
        direction = np.array([cx, cy]) - env.pos
        dist = np.linalg.norm(direction)
        steps = min(int(np.ceil(dist / 4.0)), horizon_cap)   # STEP=4.0 в openworld
        a = np.clip(direction / max(dist, 1e-6), -1, 1).astype(np.float32)
        a_t = torch.from_numpy(a[None]).to(device)
        zs = [z0.clone() for _ in members]
        step_disagreements = []
        for _ in range(max(steps, 1)):
            zs = [m(zk, a_t) for m, zk in zip(members, zs)]
            step_disagreements.append(torch.stack(zs).std(0).mean().item())
        values.append(step_disagreements[-1])              # неопределённость В ЦЕЛИ
        risks.append(float(np.mean(step_disagreements)))    # средний риск ПО ПУТИ
    if return_path_risk:
        return np.array(values), np.array(risks)
    return np.array(values)


def build_qubo(cand_points, value, M, current_pos=None, risk=None,
               max_travel_dist=55.0, penalty_budget=5.0,
               penalty_redundancy=3.0, redundancy_radius=20.0,
               w_energy=1.0, w_risk=1.0, w_transit=2.0):
    """Полная формулировка с ценой перелёта, риском и компактностью маршрута.

    Cost(x) = -value_i * x_i                                  # инф. ценность (максимизируем)
              + w_energy * energy_i * x_i                     # цена перелёта (дистанция от дрона)
              + w_risk   * risk_i   * x_i                      # риск (неуверенность ПО ПУТИ)
              + P * (sum x_i - M)^2                            # бюджет: ровно M точек
              + R * sum_{i<j, близко} x_i*x_j                  # штраф за избыточность (дублирование)
              + T * sum_{i<j} (dist_ij/max_travel_dist) x_i*x_j  # штраф за разбросанность маршрута

    Последнее слагаемое — новое: оно решает и задачу "учесть цену
    перелёта между точками", и попутно устраняет баг с порядком облёта
    (раньше QUBO мог выбрать 3 точки, каждая по отдельности ≤max_travel_dist
    от СТАРТА раунда, но далеко друг от друга — тогда третья точка
    маршрута оказывалась вне досягаемости от текущей позиции дрона к
    моменту, когда до неё доходила очередь). Штраф за суммарную
    межточечную дистанцию заставляет QUBO предпочитать компактный,
    физически облетаемый за один заход набор точек.
    """
    n = len(cand_points)
    value_norm = (value - value.min()) / (np.ptp(value) + 1e-8)

    if current_pos is not None:
        energy = np.array([np.linalg.norm(p - current_pos) for p in cand_points])
        energy_norm = energy / max_travel_dist
    else:
        energy_norm = np.zeros(n)

    if risk is not None:
        risk_norm = (risk - risk.min()) / (np.ptp(risk) + 1e-8)
    else:
        risk_norm = np.zeros(n)

    Q = {}
    for i in range(n):
        Q[(i, i)] = (Q.get((i, i), 0.0) - value_norm[i]
                    + w_energy * energy_norm[i] + w_risk * risk_norm[i])

    # бюджетное ограничение (sum x_i - M)^2
    for i in range(n):
        Q[(i, i)] = Q.get((i, i), 0.0) + penalty_budget * (1 - 2 * M)
    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = Q.get((i, j), 0.0) + 2 * penalty_budget

    # штраф за избыточность близких пар + штраф за разбросанность маршрута
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cand_points[i] - cand_points[j])
            if d < redundancy_radius:
                Q[(i, j)] = Q.get((i, j), 0.0) + penalty_redundancy
            Q[(i, j)] = Q.get((i, j), 0.0) + w_transit * (d / max_travel_dist)

    return dimod.BinaryQuadraticModel.from_qubo(Q)


def solve(bqm, num_reads=200):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    best = sampleset.first.sample
    return np.array([best[i] for i in range(len(best))], dtype=bool)


def greedy_baseline(uncertainty, M):
    """Наивный top-M по ценности — бейзлайн без учёта избыточности."""
    idx = np.argsort(-uncertainty)[:M]
    mask = np.zeros(len(uncertainty), dtype=bool)
    mask[idx] = True
    return mask


def plot_result(env, cand_points, uncertainty, qubo_mask, greedy_mask, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, mask, title in [(axes[0], greedy_mask, "Жадный top-M (без QUBO)"),
                            (axes[1], qubo_mask, "QUBO (штраф за избыточность)")]:
        for (ox, oy), r in zip(env.obs_pos, env.obs_r):
            ax.add_patch(plt.Circle((ox, oy), r, color="gray", alpha=0.5))
        sizes = 30 + 200 * (uncertainty - uncertainty.min()) / (np.ptp(uncertainty) + 1e-8)
        ax.scatter(cand_points[~mask, 0], cand_points[~mask, 1],
                  s=sizes[~mask], c="lightgray", edgecolors="k", linewidths=0.5,
                  label="не выбрано")
        ax.scatter(cand_points[mask, 0], cand_points[mask, 1],
                  s=sizes[mask], c="tab:red", edgecolors="k", linewidths=0.8,
                  label="выбрано")
        ax.plot(*env.pos, "s", color="tab:blue", ms=10, label="дрон")
        ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
        ax.set_title(title); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Frontier selection: размер точки = неопределённость модели")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/openworld_jepa.pt")
    p.add_argument("--ensemble", type=str, default="checkpoints/openworld_ensemble.pt")
    p.add_argument("--n-candidates", type=int, default=25)
    p.add_argument("--m-select", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m-travel-dist", type=float, default=55.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    enc = Encoder(ck["latent_dim"]).to(device).eval(); enc.load_state_dict(ck["encoder"])
    ens = torch.load(args.ensemble, map_location=device, weights_only=True)
    members = []
    for sd in ens["members"]:
        m = Predictor(ens["latent_dim"]).to(device).eval()
        m.load_state_dict(sd)
        members.append(m)

    env = OpenWorldEnv(seed=args.seed)
    env.reset()

    rng = np.random.default_rng(args.seed)
    cand_points = []
    while len(cand_points) < args.n_candidates:
        p_ = rng.uniform(15, SIZE - 15, size=2)
        if not env._collides(p_) and np.linalg.norm(p_ - env.pos) > 10:
            cand_points.append(p_)
    cand_points = np.array(cand_points)

    print("считаю ценность и риск по кандидатам (ансамбль x horizon)...")
    value, risk = candidate_uncertainty(enc, members, env, cand_points, device,
                                        return_path_risk=True)
    print(f"ценность: min={value.min():.4f} max={value.max():.4f} | "
          f"риск: min={risk.min():.4f} max={risk.max():.4f}")

    bqm = build_qubo(cand_points, value, M=args.m_select, current_pos=env.pos,
                     risk=risk, max_travel_dist=args.m_travel_dist)
    qubo_mask = solve(bqm)
    greedy_mask = greedy_baseline(value, args.m_select)

    print(f"\nЖадный выбор ({greedy_mask.sum()} точек): "
          f"суммарная ценность = {value[greedy_mask].sum():.3f}")
    print(f"QUBO выбор ({qubo_mask.sum()} точек): "
          f"суммарная ценность = {value[qubo_mask].sum():.3f}")
    # средняя попарная дистанция внутри выбранного набора — мера "разброса"
    for name, mask in [("Жадный", greedy_mask), ("QUBO", qubo_mask)]:
        pts = cand_points[mask]
        if len(pts) > 1:
            dists = [np.linalg.norm(pts[i] - pts[j])
                    for i in range(len(pts)) for j in range(i + 1, len(pts))]
            n_close = sum(d < 20.0 for d in dists)   # совпадает с redundancy_radius
            max_pair = max(dists)
            print(f"{name}: средняя попарная дистанция = {np.mean(dists):.1f}, "
                  f"мин. дистанция = {min(dists):.1f}, "
                  f"макс. дистанция = {max_pair:.1f}, "
                  f"избыточных пар (< 20) = {n_close}/{len(dists)}")
            # прямая проверка бага из closed_loop_exploration: если весь набор
            # компактен (макс. попарная дистанция мала), последняя точка
            # маршрута не окажется неожиданно далеко после облёта первых
            if max_pair > args.m_travel_dist:
                print(f"    ВНИМАНИЕ: макс. дистанция {max_pair:.1f} превышает "
                      f"m_travel_dist={args.m_travel_dist} — маршрут не компактен")

    plot_result(env, cand_points, value, qubo_mask, greedy_mask,
               "frontier_selection.png")
