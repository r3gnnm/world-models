"""Многократный прогон frontier_qubo с разными seed кандидатных точек —
чтобы не повторить ошибку с landmark-sweep (вывод по одному прогону).

Усредняет: (1) разрыв по суммарной неопределённости QUBO vs жадный,
(2) число избыточных пар (расстояние < redundancy_radius) в каждом наборе.

Запуск:  python frontier_multiseed.py --seeds 0 1 2 3 4
"""
import argparse
import numpy as np
import torch

from frontier_qubo import (candidate_uncertainty, build_qubo, solve,
                           greedy_baseline)
from env_openworld import OpenWorldEnv, SIZE
from models import Encoder, Predictor


def n_close_pairs(pts, radius=20.0):
    if len(pts) < 2:
        return 0
    return sum(np.linalg.norm(pts[i] - pts[j]) < radius
              for i in range(len(pts)) for j in range(i + 1, len(pts)))


def max_pair_dist(pts):
    if len(pts) < 2:
        return 0.0
    return max(np.linalg.norm(pts[i] - pts[j])
              for i in range(len(pts)) for j in range(i + 1, len(pts)))


def run_one(enc, members, seed, n_candidates, m_select, device):
    env = OpenWorldEnv(seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)
    cand = []
    while len(cand) < n_candidates:
        p_ = rng.uniform(15, SIZE - 15, size=2)
        if not env._collides(p_) and np.linalg.norm(p_ - env.pos) > 10:
            cand.append(p_)
    cand = np.array(cand)

    value, risk = candidate_uncertainty(enc, members, env, cand, device,
                                        return_path_risk=True)
    bqm = build_qubo(cand, value, M=m_select, current_pos=env.pos, risk=risk)
    qubo_mask = solve(bqm)
    greedy_mask = greedy_baseline(value, m_select)

    return {
        "seed": seed,
        "u_greedy": float(value[greedy_mask].sum()),
        "u_qubo": float(value[qubo_mask].sum()),
        "close_greedy": n_close_pairs(cand[greedy_mask]),
        "close_qubo": n_close_pairs(cand[qubo_mask]),
        "maxd_greedy": max_pair_dist(cand[greedy_mask]),
        "maxd_qubo": max_pair_dist(cand[qubo_mask]),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/openworld_jepa.pt")
    p.add_argument("--ensemble", type=str, default="checkpoints/openworld_ensemble.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-candidates", type=int, default=25)
    p.add_argument("--m-select", type=int, default=6)
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

    rows = [run_one(enc, members, s, args.n_candidates, args.m_select, device)
           for s in args.seeds]

    u_ratio = [r["u_qubo"] / r["u_greedy"] for r in rows]
    close_g = [r["close_greedy"] for r in rows]
    close_q = [r["close_qubo"] for r in rows]

    print("\n" + "=" * 76)
    print(f"{'seed':>5}{'u_greedy':>11}{'u_qubo':>10}{'close_g':>9}{'close_q':>9}"
          f"{'maxd_g':>9}{'maxd_q':>9}")
    for r in rows:
        print(f"{r['seed']:>5}{r['u_greedy']:>11.3f}{r['u_qubo']:>10.3f}"
              f"{r['close_greedy']:>9}{r['close_qubo']:>9}"
              f"{r['maxd_greedy']:>9.1f}{r['maxd_qubo']:>9.1f}")
    print("-" * 76)
    print(f"среднее отношение u_qubo/u_greedy: {np.mean(u_ratio):.3f} "
          f"(±{np.std(u_ratio):.3f})")
    print(f"среднее число избыточных пар — жадный: {np.mean(close_g):.2f}, "
          f"QUBO: {np.mean(close_q):.2f}")
    maxd_g = [r["maxd_greedy"] for r in rows]
    maxd_q = [r["maxd_qubo"] for r in rows]
    over_g = sum(d > 55.0 for d in maxd_g)
    over_q = sum(d > 55.0 for d in maxd_q)
    print(f"средняя макс. попарная дистанция — жадный: {np.mean(maxd_g):.1f}, "
          f"QUBO: {np.mean(maxd_q):.1f}")
    print(f"маршрутов, превышающих разумную дальность (>55) — "
          f"жадный: {over_g}/{len(rows)}, QUBO: {over_q}/{len(rows)}")
    print("=" * 76)
    if np.mean(close_q) < np.mean(close_g):
        print("QUBO в среднем даёт МЕНЬШЕ избыточных пар — эффект воспроизводится.")
    else:
        print("QUBO НЕ даёт устойчивого снижения избыточности — нужно калибровать "
              "penalty_redundancy или это не робастный эффект.")
